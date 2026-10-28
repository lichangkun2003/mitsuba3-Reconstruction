#pragma once

#include <mitsuba/render/interaction.h>
#include <mitsuba/render/reconstruction_tree.h>
#include <mitsuba/core/object.h>
#include <array>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

namespace reconstruction { struct Result; }

/** Point-cloud initialized voxel ellipsoids, matching Falcor's
 * VoxelReconstructionNoLightTransport forward model. This is an independent
 * renderer: alpha is continuous transmittance coverage, not a validity mask.
 * Training uses the explicit Falcor appearance derivatives and geometry proxy;
 * it does not differentiate the discontinuous hard-hit test with ordinary AD.
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB VoxelReconstruction : public Object {
public:
    MI_IMPORT_TYPES()
    using FloatBuffer = mitsuba::DynamicBuffer<Float>;
    using UIntBuffer = mitsuba::DynamicBuffer<UInt32>;
    using IntBuffer = mitsuba::DynamicBuffer<Int32>;
    using SpatialAnchor = dr::uint64_array_t<UInt32>;

    /// Core resolution plus eight shell cells on each side at the finest level.
    static constexpr uint32_t max_core_resolution = 1024;
    static constexpr uint32_t max_grid_resolution = max_core_resolution + 16;
    // Flattened DDA cells use signed Int32 arithmetic before conversion to IDs.
    static_assert(uint64_t(max_grid_resolution) * max_grid_resolution * max_grid_resolution <= INT32_MAX);
    static_assert(uint64_t(3) * max_grid_resolution + 3 <= UINT32_MAX);

    /// Version of the alpha-independent RGB training contract exposed to Python.
    static constexpr uint32_t rgb_only_version = 1;
    /// Multiple independent parent proposals per empty cell; record capacity
    /// counts intersected cells, and overflow only invalidates unstored cells.
    static constexpr uint32_t rgb_growth_version = 3;
    /// Per-hit single-removal RGB gains on complete baseline paths.
    static constexpr uint32_t rgb_deletion_version = 1;
    /// Point-pair L1 depth distortion with exact fixed-hit continuous gradients.
    static constexpr uint32_t distortion_version = 1;
    /// Linear-RGB whole-image Gaussian SSIM with an explicit image adjoint.
    static constexpr uint32_t dssim_version = 1;
    /// GPU occupancy octree, including separately tagged hypothetical leaves.
    static constexpr uint32_t octree_version = 1;
    /// Device-resident compact Adam, including transactional moment updates.
    static constexpr uint32_t optimizer_device_version = 1;
    /// Pre-compacted gradient rows can feed Adam without dense reconstruction.
    static constexpr uint32_t compact_optimizer_version = 1;
    static constexpr uint32_t compact_gradient_version = 1;
    static constexpr uint32_t projective_version = 1;

    VoxelReconstruction(const std::string &filename, uint32_t resolution,
                        const std::array<float, 3> &grid_min,
                        const std::array<float, 3> &grid_max,
                        double opacity_threshold = .05,
                        uint32_t sh_count = 9, uint32_t opacity_sh_count = 9);

    /// Plain XYZ and sRGB triples already expressed in the renderer frame.
    VoxelReconstruction(const float *positions, const float *colors, size_t point_count,
                        uint32_t resolution, const std::array<float, 3> &grid_min,
                        const std::array<float, 3> &grid_max,
                        uint32_t sh_count = 9, uint32_t opacity_sh_count = 9,
                        uint32_t core_resolution = 0);

    VoxelReconstruction(uint32_t resolution,
                        const std::array<float, 3> &grid_min,
                        const std::array<float, 3> &grid_max,
                        const UIntBuffer &cell_indices,
                        const std::map<std::string, FloatBuffer> &parameters,
                        uint32_t sh_count = 9, uint32_t opacity_sh_count = 9,
                        uint32_t core_resolution = 0);

    /// Replace continuous parameters without changing occupancy or array sizes.
    void set_parameters(const std::map<std::string, FloatBuffer> &parameters);

    /// Training-only device transaction: retain every old parameter unless the
    /// scalar guard and every proposed value are finite/valid. Return the shared
    /// acceptance mask so the caller can commit Adam state with the same gate.
    /// Shape/key errors remain synchronous, but no device value is read on CPU.
    Mask set_parameters_guarded(const std::map<std::string, FloatBuffer> &parameters,
                                const Mask &allowed);

    /// Commit compact rows without scanning/replacing full parameter buffers.
    /// Appearance and geometry row IDs must each be strictly increasing and
    /// in bounds. Invalid IDs, nonfinite values, or a false scalar guard reject
    /// the whole transaction on device. The returned gate also commits Adam.
    Mask set_parameters_sparse(const UIntBuffer &appearance_indices,
                               const UIntBuffer &geometry_indices,
                               const std::map<std::string, FloatBuffer> &parameters,
                               const Mask &allowed);

    /// Compact active row IDs without returning their lengths to the host.
    /// Borrow moment arrays in place: no full-table snapshots/copies are made.
    /// A validation pass precedes a commit pass using the same scalar gate;
    /// fixed-size device batches bound temporary storage independently of SH
    /// table size. Per-view/group clocks and zero-gradient updates are retained.
    Mask optimizer_step_device(
        const std::map<std::string, FloatBuffer> &gradients,
        const UIntBuffer &appearance_valid, const UIntBuffer &geometry_valid,
        const UIntBuffer &background_carve_valid,
        const std::map<std::string, FloatBuffer *> &means,
        const std::map<std::string, FloatBuffer *> &squared_means,
        const std::map<std::string, UIntBuffer *> &steps,
        const std::map<std::string, double> &settings, const Mask &allowed,
        bool check_moments = true);

    /// Consume sorted global IDs with K-row gradients/counts directly. Model
    /// and moment storage remain full-sized; only the supplied valid rows are
    /// visited. The explicit device row_count keeps varying K replayable.
    /// Invalid IDs or a row-count mismatch reject the complete transaction.
    Mask optimizer_step_compact_device(
        const UIntBuffer &indices, const UInt32 &row_count,
        const std::map<std::string, FloatBuffer> &gradients,
        const UIntBuffer &appearance_valid, const UIntBuffer &geometry_valid,
        const UIntBuffer &background_carve_valid,
        const std::map<std::string, FloatBuffer *> &means,
        const std::map<std::string, FloatBuffer *> &squared_means,
        const std::map<std::string, UIntBuffer *> &steps,
        const std::map<std::string, double> &settings, const Mask &allowed,
        bool check_moments = true);

    /// Enter before frozen recording/replay. Detach genuine external aliases,
    /// then stash the unique owners' reference counts so tracing's temporary
    /// references do not force full parameter/moment copies. The caller must
    /// release every returned handle with jit_var_unstash_ref(), also on error.
    std::vector<uint64_t> optimizer_update_scope_begin(
        const std::map<std::string, FloatBuffer *> &means,
        const std::map<std::string, FloatBuffer *> &squared_means,
        const std::map<std::string, UIntBuffer *> &steps);

    /// Install immutable hypothetical proposals, allowing repeated target cells.
    /// Each input row retains its own evidence index. They never change trace(),
    /// actual parameter buffers, occupancy snapshots, or the training image/gradient.
    void set_growth_candidates(const UIntBuffer &cell_indices,
                               const std::map<std::string, FloatBuffer> &parameters);
    void clear_growth_candidates();
    uint32_t growth_candidate_count() const { return m_growth_count; }
    uint64_t topology_revision() const { return m_topology_revision; }
    /// Mapping is model metadata, never a trainable parameter. Changing it
    /// invalidates frozen recordings through topology_revision().
    void set_scene_mapping(const std::string &mapping);
    std::string native_scene_mapping() const { return m_scene_mapping; }

    struct TrainingResult {
        Color3f rgb, raw_rgb;
        Float alpha, raw_alpha, loss, rgb_loss, distortion_loss;
        Float dssim_loss;
        bool dssim_enabled = false;
        UInt32 hit_count;
        /// Fixed-capacity compact gradient layout; the required count may exceed
        /// capacity, in which case the caller must retry before any Adam update.
        bool compact_gradients = false;
        UIntBuffer voxel_ids;
        UInt32 voxel_count = UInt32(0);
        Mask compact_gradient_overflow = Mask(false);
        std::map<std::string, FloatBuffer> gradients;
        std::map<std::string, UIntBuffer> counters;
        /// Per-candidate single-insertion RGB loss gain and ray coverage.
        /// Incomplete paths and unstored overflow candidates contribute unknown.
        std::map<std::string, FloatBuffer> growth_sums;
        std::map<std::string, UIntBuffer> growth_counts;
        /// Optional per-voxel single-removal RGB gain and ray coverage.
        /// Truncated or invalid paths abstain without affecting other rays.
        std::map<std::string, FloatBuffer> deletion_sums;
        std::map<std::string, UIntBuffer> deletion_counts;
        /// Populated only by the diagnostic profile_stages setting.
        std::map<std::string, std::map<std::string, double>> stage_profile;
    };

    /// Per-pixel 1 - mean_RGB(SSIM), and the derivative of its pixel SUM.
    /// Images are complete row-major arrays. The 11-tap sigma-1.5 Gaussian
    /// uses zero padding and data_range=1, without clipping input linear RGB.
    /// Negative roundoff variances and SSIM outside [-1, 1] are clipped;
    /// the returned derivative includes these stabilization masks.
    static std::pair<Float, Color3f> dssim_image_loss_and_grad(
        const Color3f &rgb, const Color3f &reference,
        uint32_t image_width, uint32_t image_height);

    /// One sample per ray/view. Pixel contributions are summed into voxel
    /// gradients; the caller owns the optimizer and the per-view update clock.
    /// With settings["rgb_only"] != 0, reference_rgb is the composited target
    /// and reference_alpha is ignored (including its width and values). This
    /// mode uses RGB loss and the ungated sigmoid-distance geometry proxy;
    /// alpha supervision, background carving, and legacy deletion evidence
    /// are off. collect_rgb_deletion_evidence explicitly enables RGB-only
    /// single-removal evidence using the existing reverse traversal.
    /// Internal opacity, transmittance, and returned rendered alpha remain active.
    /// distortion_loss_weight optionally adds pairwise L1 separation of the
    /// recorded hard-hit contributions. Depth is the clipped ellipsoid-metric
    /// closest point, normalized by the fixed grid diagonal for legacy models.
    /// Projective models use world distance t/(t+core_half_extent), including
    /// its exact local-distance derivative. Environment color, unrecorded
    /// suffixes, near-miss surrogates, and extra tracing are not included.
    /// compact_gradient_capacity > 0 stores gradients and the three optimizer
    /// validity counters in sorted voxel_ids order, with a device voxel_count.
    /// Topology evidence retains full model indexing. Unused capacity is ignored;
    /// an overflow never writes outside the buffers and must not be optimized.
    /// dssim_loss_weight > 0 adds the whole-image linear-RGB loss above, using
    /// image_width/image_height and the same forward records. It requires
    /// RGB-only, complete row-major images and unit inverse_pdf. Invalid device
    /// weights produce nonfinite loss/gradients for the caller's finite guard.
    TrainingResult loss_and_grad(const Ray3f &ray, const Color3f &reference_rgb,
                                 const Float &reference_alpha,
                                 const Color3f &background,
                                 const Float &inverse_pdf = Float(1.f),
                                 const std::map<std::string, float> &settings = {}) const;

    /// Trace independent rays, returning linear RGB (including background),
    /// accumulated alpha, and contributing hit count. maxt bounds the ray.
    /// entry_offset_fraction reproduces Falcor's near/far segment entry nudge
    /// when set to 1e-6 for legacy models. Projective segments use local voxel
    /// units instead of an infinite far plane; zero disables the entry nudge.
    std::tuple<Color3f, Float, UInt32> trace(
        const Ray3f &ray, const Color3f &background,
        float transmittance_threshold = .01f, uint32_t max_hits = 16,
        float entry_offset_fraction = 0.f, bool use_octree = true) const;

    uint32_t resolution() const { return m_resolution; }
    /// Resolution across the uncompressed core; storage may include a shell.
    uint32_t core_resolution() const { return m_core_resolution; }
    uint32_t active_voxel_count() const { return m_active_count; }
    uint32_t sh_count() const { return m_sh_count; }
    uint32_t opacity_sh_count() const { return m_opacity_sh_count; }
    std::array<float, 3> grid_min() const { return m_min; }
    std::array<float, 3> grid_max() const { return m_max; }
    std::array<float, 3> voxel_size() const { return m_size; }
    const std::map<std::string, uint64_t> &statistics() const { return m_statistics; }

    const UIntBuffer &cell_indices() const { return m_cells; }
    const FloatBuffer &centers() const { return m_centers; }
    const FloatBuffer &log_scales() const { return m_log_scales; }
    const FloatBuffer &rotations() const { return m_rotations; }
    const FloatBuffer &radiance_sh() const { return m_radiance; }
    const FloatBuffer &opacity_sh() const { return m_opacity; }
    std::string to_string() const override;
    MI_DECLARE_CLASS(VoxelReconstruction)

private:
    struct ProjectiveSegment {
        Mask valid;
        Point3f origin;
        Vector3f direction;
        Float entry, exit, world_entry, h_start, h_rate, speed;
        UInt32 visited, chart;
        DRJIT_STRUCT(ProjectiveSegment, valid, origin, direction, entry, exit,
                     world_entry, h_start, h_rate, speed, visited, chart)
    };
    ProjectiveSegment next_projective_segment(const Point3f &origin,
        const Vector3f &direction, const Float &maxt, const UInt32 &visited) const;
    void initialize_projective_segment(const ProjectiveSegment &segment,
        Vector3i &cell, Vector3f &next, Float &entry, float entry_offset_fraction,
        bool use_octree) const;
    Vector3f projective_hit_direction(const Point3f &origin,
        const Vector3f &direction, const UInt32 &id, const Mask &active) const;
    Mask projective_core_ids(const UInt32 &id, const Mask &active) const;
    Mask projective_cell_inside(const Vector3i &cell, const UInt32 &chart) const;
    std::pair<Float,Float> projective_depth(const ProjectiveSegment &segment,
                                           const Float &distance) const;
    Mask optimizer_step_device_impl(
        const std::map<std::string, FloatBuffer> &gradients,
        const UIntBuffer &appearance_valid, const UIntBuffer &geometry_valid,
        const UIntBuffer &background_carve_valid,
        const std::map<std::string, FloatBuffer *> &means,
        const std::map<std::string, FloatBuffer *> &squared_means,
        const std::map<std::string, UIntBuffer *> &steps,
        const std::map<std::string, double> &settings, const Mask &allowed,
        bool check_moments, const UIntBuffer *compact_indices,
        const UInt32 *compact_count);
    void initialize_seeds(const reconstruction::Result &loaded);
    void rebuild_octree();
    void update_spatial_statistics();
    Vector3f octree_boundary_times(const Vector3i &cell, const Vector3i &step,
                                  const Point3f &origin, const Vector3f &inv_d) const;
    void advance_empty_block(Vector3i &cell, Vector3f &next, Float &entry,
                             UInt32 &iterations, const UInt32 &log_size,
                             const Vector3i &step, const Vector3f &step_t,
                             const Point3f &origin, const Vector3f &inv_d,
                             const Float &exit) const;
    void skip_empty_block(Vector3i &cell, Vector3f &next, Float &entry,
                          UInt32 &iterations, const Vector3i &step,
                          const Vector3f &step_t, const Point3f &origin,
                          const Vector3f &inv_d, const Float &exit,
                          Mask active, const UInt32 &log_size) const;

    uint32_t m_resolution, m_core_resolution, m_active_count, m_sh_count, m_opacity_sh_count;
    std::string m_scene_mapping = "bounded";
    std::array<float, 3> m_min, m_max, m_size;
    std::map<std::string, uint64_t> m_statistics;
    UIntBuffer m_cells;
    // Only occupied leaves and their ancestors are stored. A small independent
    // candidate overlay avoids rebuilding the normal tree at every evidence
    // window. Joint traversal intersects the two safe empty-block bounds.
    using SpatialTree = CompactReconstructionTree<UIntBuffer, IntBuffer>;
    SpatialTree m_tree, m_growth_tree;
    uint64_t m_topology_revision = 0;
    FloatBuffer m_centers, m_log_scales, m_rotations, m_radiance, m_opacity;
    uint32_t m_growth_count = 0;
    // Frozen-function traversal in this Dr.Jit version requires nonempty
    // buffers. These inert placeholders are never addressed when count is zero.
    // Cells are unique scatter destinations. Per-proposal linked lists preserve
    // original evidence IDs while sharing one path record for each target cell.
    UIntBuffer m_growth_cells = dr::zeros<UIntBuffer>(1),
               m_growth_next = dr::zeros<UIntBuffer>(1);
    FloatBuffer m_growth_centers = dr::zeros<FloatBuffer>(1),
                m_growth_log_scales = dr::zeros<FloatBuffer>(1),
                m_growth_rotations = dr::zeros<FloatBuffer>(1),
                m_growth_radiance = dr::zeros<FloatBuffer>(1),
                m_growth_opacity = dr::zeros<FloatBuffer>(1);
    MI_TRAVERSE_CB(Object, m_cells, m_tree, m_growth_tree, m_centers, m_log_scales,
                   m_rotations, m_radiance, m_opacity, m_growth_cells, m_growth_next,
                   m_growth_centers, m_growth_log_scales, m_growth_rotations,
                   m_growth_radiance, m_growth_opacity)
};

MI_EXTERN_CLASS(VoxelReconstruction)
NAMESPACE_END(mitsuba)
