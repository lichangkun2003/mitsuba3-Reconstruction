#pragma once

#include <mitsuba/render/interaction.h>
#include <mitsuba/core/object.h>
#include <array>
#include <map>
#include <string>
#include <tuple>

NAMESPACE_BEGIN(mitsuba)

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

    VoxelReconstruction(const std::string &filename, uint32_t resolution,
                        const std::array<float, 3> &grid_min,
                        const std::array<float, 3> &grid_max,
                        double opacity_threshold = .05,
                        uint32_t sh_count = 9, uint32_t opacity_sh_count = 9);

    VoxelReconstruction(uint32_t resolution,
                        const std::array<float, 3> &grid_min,
                        const std::array<float, 3> &grid_max,
                        const UIntBuffer &cell_indices,
                        const std::map<std::string, FloatBuffer> &parameters,
                        uint32_t sh_count = 9, uint32_t opacity_sh_count = 9);

    /// Replace continuous parameters without changing occupancy or array sizes.
    void set_parameters(const std::map<std::string, FloatBuffer> &parameters);

    struct TrainingResult {
        Color3f rgb, raw_rgb;
        Float alpha, raw_alpha, loss;
        UInt32 hit_count;
        std::map<std::string, FloatBuffer> gradients;
        std::map<std::string, UIntBuffer> counters;
    };

    /// One sample per ray/view. Pixel contributions are summed into voxel
    /// gradients; the caller owns the optimizer and the per-view update clock.
    TrainingResult loss_and_grad(const Ray3f &ray, const Color3f &reference_rgb,
                                 const Float &reference_alpha,
                                 const Color3f &background,
                                 const Float &inverse_pdf = Float(1.f),
                                 const std::map<std::string, float> &settings = {}) const;

    /// Trace independent rays, returning linear RGB (including background),
    /// accumulated alpha, and contributing hit count. maxt bounds the ray.
    /// entry_offset_fraction reproduces Falcor's near/far segment entry nudge
    /// when set to 1e-6; zero gives ordinary segment intersection semantics.
    std::tuple<Color3f, Float, UInt32> trace(
        const Ray3f &ray, const Color3f &background,
        float transmittance_threshold = .01f, uint32_t max_hits = 16,
        float entry_offset_fraction = 0.f) const;

    uint32_t resolution() const { return m_resolution; }
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
    uint32_t m_resolution, m_active_count, m_sh_count, m_opacity_sh_count;
    std::array<float, 3> m_min, m_max, m_size;
    std::map<std::string, uint64_t> m_statistics;
    IntBuffer m_index;
    UIntBuffer m_cells;
    FloatBuffer m_centers, m_log_scales, m_rotations, m_radiance, m_opacity;
    MI_TRAVERSE_CB(Object, m_index, m_cells, m_centers, m_log_scales,
                   m_rotations, m_radiance, m_opacity)
};

MI_EXTERN_CLASS(VoxelReconstruction)
NAMESPACE_END(mitsuba)
