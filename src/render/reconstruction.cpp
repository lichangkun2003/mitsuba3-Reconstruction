#include <mitsuba/render/reconstruction.h>
#include <drjit/if_stmt.h>
#include <drjit/while_loop.h>
#include "reconstruction_pointcloud.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>

NAMESPACE_BEGIN(mitsuba)

namespace {

/// Encode a completed linear pixel for image supervision. The extension below
/// zero and above one intentionally avoids clipping away corrective gradients.
/// Keep the inactive power branch finite at zero and for negative estimators.
template <typename Color>
Color reconstruction_linear_to_srgb(const Color &linear) {
    return dr::select(linear <= .0031308f, 12.92f * linear,
        1.055f * dr::pow(dr::maximum(linear, .0031308f), 1.f / 2.4f) - .055f);
}

template <typename Color>
Color reconstruction_srgb_derivative(const Color &linear, const Color &encoded) {
    // Reuse the encoded power instead of evaluating a second fractional power.
    return dr::select(linear <= .0031308f, Color(12.92f),
        (encoded + .055f) / (2.4f * dr::maximum(linear, .0031308f)));
}

/// Keep v5/v6 addresses canonical through depth 17. Only shell 9 can reach
/// depth 18 on the native world grid. Its v7 addresses use the otherwise
/// unused top nibble 0xf, a 6-bit local root and 54 Morton bits; the sentinel
/// is implicit. All three formats decode to a sentinel path and coarse root.
template <typename Cell>
std::array<Cell, 2> adaptive_address_parts(const Cell &cells) {
    auto extended = (cells & (uint64_t(1) << 62u)) != 0u;
    auto deepest = (cells >> 60u) == 0xfu;
    return {
        dr::select(deepest, (cells & ((uint64_t(1) << 54u) - 1u)) | (uint64_t(1) << 54u),
            dr::select(extended, cells & ((uint64_t(1) << 52u) - 1u),
                                 cells & uint64_t(0xffffffffu))),
        dr::select(deepest, uint64_t(504u) + ((cells >> 54u) & uint64_t(0x3fu)),
            dr::select(extended, (cells >> 52u) & uint64_t(0x3ffu),
                                 (cells >> 32u) & uint64_t(0x7fffffffu)))
    };
}

/// Diagnostic-only barriers and nested history regions. The disabled path must
/// neither change evaluation nor read a clock or alter the caller's JIT flags.
template <bool IsJit> class ReconstructionStageProfiler {
public:
    using Metrics = std::map<std::string, double>;
    using Profile = std::map<std::string, Metrics>;
    using Clock = std::chrono::steady_clock;

    explicit ReconstructionStageProfiler(bool enabled) : m_enabled(enabled) {
        if (m_enabled) {
            if constexpr (IsJit) {
                // Pending work belongs to the caller, not the first stage.
                dr::sync_thread();
                m_flag.emplace(JitFlag::KernelHistory, true);
            }
            begin();
        }
    }

    ReconstructionStageProfiler(const ReconstructionStageProfiler &) = delete;
    ReconstructionStageProfiler &operator=(const ReconstructionStageProfiler &) = delete;

    ~ReconstructionStageProfiler() {
        if constexpr (IsJit) {
            if (m_region_active)
                jit_kernel_history_free(jit_kernel_history_end(m_region));
        }
        // m_flag restores the previous flag, including an outer capture region.
    }

    bool enabled() const { return m_enabled; }

    void begin() {
        if (!m_enabled)
            return;
        if constexpr (IsJit) {
            m_region = jit_kernel_history_begin();
            m_region_active = true;
        }
        m_start = Clock::now();
    }

    void finish(const char *stage) {
        if (!m_enabled)
            return;
        if constexpr (IsJit)
            dr::sync_thread();
        double seconds = std::chrono::duration<double>(Clock::now() - m_start).count();
        Metrics metrics = zero_metrics();
        metrics["wall_seconds"] = seconds;
        if constexpr (IsJit) {
            using Snapshot = std::unique_ptr<KernelHistory, decltype(&jit_kernel_history_free)>;
            Snapshot history(jit_kernel_history_end(m_region), &jit_kernel_history_free);
            m_region_active = false;
            const size_t count = jit_kernel_history_size(history.get());
            metrics["operations"] = double(count);
            for (size_t i = 0; i < count; ++i) {
                auto query = [&](KernelHistoryField field) {
                    return jit_kernel_history_query(history.get(), i, field);
                };
                metrics["device_ms"] += double(query(KernelHistoryField::ExecutionTime)) * 1e-6;
                metrics["codegen_ms"] += double(query(KernelHistoryField::CodegenTime)) * 1e-6;
                metrics["backend_compile_ms"] += double(query(KernelHistoryField::BackendTime)) * 1e-6;
                if (query(KernelHistoryField::Type) == uint64_t(KernelType::JIT)) {
                    metrics["jit_kernels"] += 1.;
                    metrics[query(KernelHistoryField::CacheHit) ? "cache_hits" : "cache_misses"] += 1.;
                }
            }
        }
        profile[stage] = std::move(metrics);
    }

    void zero_stage(const char *stage) {
        if (m_enabled)
            profile[stage] = zero_metrics();
    }

    Profile profile;

private:
    static Metrics zero_metrics() {
        return {{"wall_seconds", 0.}, {"device_ms", 0.}, {"codegen_ms", 0.},
                {"backend_compile_ms", 0.}, {"operations", 0.}, {"jit_kernels", 0.},
                {"cache_hits", 0.}, {"cache_misses", 0.}};
    }

    bool m_enabled, m_region_active = false;
    uint64_t m_region = 0;
    Clock::time_point m_start{};
    std::optional<dr::scoped_set_flag> m_flag;
};

} // namespace

MI_VARIANT VoxelReconstruction<Float, Spectrum>::VoxelReconstruction(
    const std::string &filename, uint32_t resolution,
    const std::array<float, 3> &grid_min, const std::array<float, 3> &grid_max,
    double opacity_threshold, uint32_t sh_count, uint32_t opacity_sh_count)
    : m_resolution(resolution), m_core_resolution(resolution), m_active_count(0), m_sh_count(sh_count),
      m_active_sh_count(sh_count), m_opacity_sh_count(opacity_sh_count), m_active_opacity_sh_count(opacity_sh_count), m_min(grid_min), m_max(grid_max) {
    if (resolution < 1 || resolution > max_core_resolution)
        Throw("VoxelReconstruction: resolution must be in [1, 1024].");
    if (sh_count < 1 || sh_count > 16 || opacity_sh_count < 1 || opacity_sh_count > 16)
        Throw("VoxelReconstruction: SH coefficient counts must be in [1, 16].");
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(m_min[axis]) || !std::isfinite(m_max[axis]) ||
            m_max[axis] <= m_min[axis])
            Throw("VoxelReconstruction: grid bounds must be finite and increasing.");
        m_size[axis] = (m_max[axis] - m_min[axis]) / float(resolution);
    }
    const auto loaded = reconstruction::load(std::filesystem::u8path(filename),
        {resolution, resolution, resolution}, m_min, m_size, opacity_threshold);
    initialize_seeds(loaded);
}

MI_VARIANT VoxelReconstruction<Float, Spectrum>::VoxelReconstruction(
    const float *positions, const float *colors, size_t point_count,
    uint32_t resolution, const std::array<float, 3> &grid_min,
    const std::array<float, 3> &grid_max, uint32_t sh_count,
    uint32_t opacity_sh_count, uint32_t core_resolution, uint32_t background_shell_levels,
    uint32_t background_subdivision_level, bool background_use_ellipsoid, bool background_adaptive)
    : m_resolution(resolution), m_core_resolution(core_resolution ? core_resolution : resolution),
      m_active_count(0), m_sh_count(sh_count), m_active_sh_count(sh_count), m_opacity_sh_count(opacity_sh_count), m_active_opacity_sh_count(opacity_sh_count),
      m_min(grid_min), m_max(grid_max) {
    if (resolution < 1 || resolution > max_grid_resolution ||
        m_core_resolution > max_core_resolution || m_core_resolution != resolution)
        Throw("VoxelReconstruction.from_points(): core resolution must be in [1, 1024], "
              "with core_resolution equal to resolution; compressed layouts are unsupported.");
    if (sh_count < 1 || sh_count > 16 || opacity_sh_count < 1 || opacity_sh_count > 16)
        Throw("VoxelReconstruction.from_points(): SH coefficient counts must be in [1, 16].");
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(m_min[axis]) || !std::isfinite(m_max[axis]) ||
            m_max[axis] <= m_min[axis])
            Throw("VoxelReconstruction.from_points(): grid bounds must be finite and increasing.");
        m_size[axis] = (m_max[axis] - m_min[axis]) / float(resolution);
    }
    m_background_subdivision_level = background_subdivision_level;
    m_background_adaptive = background_adaptive;
    m_background_use_ellipsoid = background_use_ellipsoid;
    if (background_shell_levels) {
        m_background_shell_levels = background_shell_levels;
        m_scene_mapping = "world_shell";
    }
    initialize_seeds(reconstruction::load_points(positions, colors, point_count,
        {resolution, resolution, resolution}, m_min, m_size));
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::initialize_seeds(
    const reconstruction::Result &loaded) {
    if (loaded.seeds.empty())
        Throw("VoxelReconstruction: no occupied voxels; check point cloud and grid bounds.");
    m_active_count = (uint32_t) loaded.seeds.size();
    const auto &s = loaded.statistics;
    m_statistics = {{"input_points", s.inputPoints}, {"invalid_points", s.invalidPoints},
        {"outside_points", s.outsidePoints}, {"colored_points", s.coloredPoints},
        {"dropped_by_opacity", s.droppedByOpacity}, {"tested_blocks", s.testedBlocks},
        {"skipped_full_blocks", s.skippedFullBlocks}, {"tested_cells", s.testedCells},
        {"covered_cells", s.coveredCells}, {"max_covered_cells", s.maxCoveredCells}};

    const size_t n = m_active_count;
    const uint32_t sh_count = m_sh_count,
                   opacity_sh_count = m_opacity_sh_count;
    std::vector<uint64_t> cells(n);
    std::vector<float> centers(n * 3), scales(n * 3), rotations(n * 4),
        radiance(n * sh_count * 3, 0.f), opacity(n * opacity_sh_count, 0.f);
    constexpr float y00 = 0.28209479177387814f;
    for (size_t i = 0; i < n; ++i) {
        const auto &seed = loaded.seeds[i];
        cells[i] = seed.voxelIndex;
        for (size_t a = 0; a < 3; ++a) {
            centers[i * 3 + a] = seed.center[a];
            scales[i * 3 + a] = std::log(std::max(seed.scale[a], 1e-8f));
            radiance[i * sh_count * 3 + a] = seed.color[a] / y00;
        }
        for (size_t a = 0; a < 4; ++a)
            rotations[i * 4 + a] = seed.rotation[a];
    }
    m_cells = dr::load<CellBuffer>(cells.data(), cells.size());
    m_centers = dr::load<FloatBuffer>(centers.data(), centers.size());
    m_log_scales = dr::load<FloatBuffer>(scales.data(), scales.size());
    m_rotations = dr::load<FloatBuffer>(rotations.data(), rotations.size());
    m_radiance = dr::load<FloatBuffer>(radiance.data(), radiance.size());
    m_opacity = dr::load<FloatBuffer>(opacity.data(), opacity.size());
    rebuild_octree();
}

MI_VARIANT VoxelReconstruction<Float, Spectrum>::VoxelReconstruction(
    uint32_t resolution, const std::array<float, 3> &grid_min,
    const std::array<float, 3> &grid_max, const CellBuffer &cells,
    const std::map<std::string, FloatBuffer> &parameters,
    uint32_t sh_count, uint32_t opacity_sh_count, uint32_t core_resolution, uint32_t background_shell_levels,
    uint32_t background_subdivision_level, bool background_use_ellipsoid, bool background_adaptive)
    : m_resolution(resolution), m_core_resolution(core_resolution ? core_resolution : resolution),
      m_active_count((uint32_t) cells.size()),
      m_sh_count(sh_count), m_active_sh_count(sh_count), m_opacity_sh_count(opacity_sh_count), m_active_opacity_sh_count(opacity_sh_count),
      m_min(grid_min), m_max(grid_max) {
    if (!resolution || resolution > max_grid_resolution || cells.size() > size_t(INT32_MAX) ||
        m_core_resolution > max_core_resolution || m_core_resolution != resolution ||
        sh_count < 1 || sh_count > 16 || opacity_sh_count < 1 || opacity_sh_count > 16)
        Throw("VoxelReconstruction.from_state(): invalid grid, occupancy, or SH counts.");
    for (size_t a = 0; a < 3; ++a) {
        if (!std::isfinite(m_min[a]) || !std::isfinite(m_max[a]) || m_max[a] <= m_min[a])
            Throw("VoxelReconstruction.from_state(): invalid bounds.");
        m_size[a] = (m_max[a] - m_min[a]) / float(resolution);
    }
    m_background_subdivision_level = background_subdivision_level;
    m_background_adaptive = background_adaptive;
    m_background_use_ellipsoid = background_use_ellipsoid;
    if (background_shell_levels) {
        m_background_shell_levels = background_shell_levels;
        m_scene_mapping = "world_shell";
    }
    for (const auto *key : {"centers", "log_scales", "rotations", "radiance_sh", "opacity_sh"})
        if (parameters.find(key) == parameters.end())
            Throw("VoxelReconstruction.from_state(): missing %s.", key);
    if (!m_active_count) {
        m_cells = dr::zeros<CellBuffer>(0);
        set_parameters(parameters);
        m_statistics = {{"restored_voxels", 0}};
        rebuild_octree();
        return;
    }
    m_cells = dr::detach(cells);
    set_parameters(parameters);
    m_statistics = {{"restored_voxels", m_active_count}};
    rebuild_octree();
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::update_spatial_statistics() {
    const uint64_t leaf_bytes = uint64_t(m_tree.leaf_ids.size() +
                                        m_growth_tree.leaf_ids.size()) * sizeof(int32_t);
    m_statistics["spatial_index_entries"] = uint64_t(m_tree.leaf_count) + m_growth_tree.leaf_count;
    m_statistics["spatial_index_allocated_bytes"] = leaf_bytes;
    m_statistics["octree_levels"] = m_tree.depth;
    m_statistics["octree_padded_resolution"] = m_tree.padded_resolution;
    m_statistics["octree_parent_nodes"] = uint64_t(m_tree.node_count) + m_growth_tree.node_count;
    m_statistics["octree_allocated_bytes"] = m_tree.allocated_bytes() +
                                             m_growth_tree.allocated_bytes() - leaf_bytes;
    m_statistics["octree_normal_parent_nodes"] = m_tree.node_count;
    m_statistics["octree_candidate_parent_nodes"] = m_growth_tree.node_count;
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::rebuild_octree() {
    // Payload rows preserve parameter/Adam order, irrespective of Morton order.
    // Validation and sorting only touch occupied leaves, never resolution^3.
    m_core_adaptive = m_cells.size() && dr::slice(dr::any(
        ((m_cells >> 63u) == 0u) & ((m_cells & (uint64_t(1) << 61u)) != 0u)));
    build_world_tree(m_tree, m_cells, IntBuffer(dr::arange<UIntBuffer>(m_active_count)));
    m_growth_tree.clear(root_resolution());
    update_spatial_statistics();
    ++m_topology_revision;
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Vector3f
VoxelReconstruction<Float, Spectrum>::octree_boundary_times(
    const Vector3i &cell, const Vector3i &step,
    const Point3f &origin, const Vector3f &inv_d) const {
    Vector3f result;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        Float face = Float(cell[axis]) + dr::select(step[axis] > 0, 1.f, 0.f);
        Float plane = dr::fmadd(face-float(core_offset()), tree_cell_size(axis), m_min[axis]);
        result[axis] = dr::select(step[axis] != 0,
            (plane - origin[axis]) * inv_d[axis], dr::Infinity<Float>);
    }
    return result;
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::skip_empty_block(
    Vector3i &cell, Vector3f &next, Float &entry, UInt32 &iterations,
    const Vector3i &step, const Vector3f &step_t, const Point3f &origin,
    const Vector3f &inv_d, const Float &exit,
    Mask active, const UInt32 &log_size) const {
    Mask skippable = active & (log_size > 0u);
    dr::tie(cell, next, entry, iterations) = dr::if_stmt(
        dr::make_tuple(cell, next, entry, iterations), skippable,
        [this, log_size, step, step_t, origin, inv_d, exit](
            Vector3i cell, Vector3f next, Float entry, UInt32 iterations) {
            advance_empty_block(cell, next, entry, iterations, log_size,
                                step, step_t, origin, inv_d, exit);
            return dr::make_tuple(cell, next, entry, iterations);
        }, [](Vector3i cell, Vector3f next, Float entry, UInt32 iterations) {
            return dr::make_tuple(cell, next, entry, iterations);
        }, "Voxel reconstruction octree block exit");
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::advance_empty_block(
    Vector3i &cell, Vector3f &next, Float &entry, UInt32 &iterations,
    const UInt32 &log_size, const Vector3i &step, const Vector3f &step_t,
    const Point3f &origin, const Vector3f &inv_d, const Float &exit) const {
    Int32 size = Int32(UInt32(1u) << log_size);
    Vector3i lo, hi, remaining;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        lo[axis] = Int32((UInt32(cell[axis]) >> log_size) << log_size);
        hi[axis] = dr::minimum(lo[axis] + size - 1, int(root_resolution()) - 1);
        remaining[axis] = dr::select(step[axis] > 0, hi[axis] - cell[axis],
                           dr::select(step[axis] < 0, cell[axis] - lo[axis], 0));
    }
    Vector3f block_exit = octree_boundary_times(cell + step * remaining, step, origin, inv_d);
    Float target = dr::minimum(exit, dr::min(block_exit));
    Mask jump = target > entry;
    // Count internal face events strictly BEFORE the block exit. All events
    // at the exit time stay pending for the ordinary x/y/z boundary ordering,
    // including zero-length neighboring leaves. This is a bulk calculation,
    // not a loop through empty fine cells.
    Vector3i consumed;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        Mask moving = jump & (step[axis] != 0);
        Float distance = dr::select(moving, target - next[axis], 0.f);
        Float delta = dr::select(moving, step_t[axis], 1.f);
        Float estimate = distance / delta;
        estimate = dr::select(dr::isfinite(estimate), estimate, 0.f);
        consumed[axis] = Int32(dr::clip(dr::ceil(estimate),
                                       0.f, Float(remaining[axis])));
    }
    Vector3f last = octree_boundary_times(cell + step * (consumed - 1), step, origin, inv_d);
    consumed -= dr::select((consumed > 0) & (last >= target), 1, 0);
    Vector3f following = octree_boundary_times(cell + step * consumed, step, origin, inv_d);
    consumed += dr::select(jump & (consumed < remaining) & (following < target), 1, 0);
    last = octree_boundary_times(cell + step * (consumed - 1), step, origin, inv_d);
    following = octree_boundary_times(cell + step * consumed, step, origin, inv_d);
    Mask valid = dr::all(((consumed == 0) | (last < target)) &
                        ((consumed == remaining) | (following >= target)));
    // With far origins, several grid planes can round to the same ray time.
    // A failed estimate conservatively leaves the ENTIRE ray state untouched;
    // the normal one-face rule then handles it without skipping a live leaf.
    consumed = dr::select(jump & valid, consumed, Vector3i(0));
    for (uint32_t axis = 0; axis < 3; ++axis)
        entry = dr::maximum(entry, dr::select(consumed[axis] > 0, last[axis], entry));
    cell += step * consumed;
    next = dr::select(jump & valid, dr::maximum(entry, following), next);
    iterations += UInt32(dr::sum(consumed));
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_parameters(
    const std::map<std::string, FloatBuffer> &parameters) {
    const std::map<std::string, size_t> sizes = {
        {"centers", size_t(m_active_count)*3}, {"log_scales", size_t(m_active_count)*3},
        {"rotations", size_t(m_active_count)*4},
        {"radiance_sh", size_t(m_active_count)*m_sh_count*3},
        {"opacity_sh", size_t(m_active_count)*m_opacity_sh_count}};
    // Validate the complete transaction before changing any parameter. Queue
    // the reductions together and read one flag on the normal GPU path, rather
    // than synchronizing after each of the five parameter groups.
    Mask all_finite = true;
    for (const auto &[key, values] : parameters) {
        auto expected = sizes.find(key);
        if (expected == sizes.end() || values.size() != expected->second)
            Throw("VoxelReconstruction.set_parameters(): unknown key or wrong size: %s.", key.c_str());
        if (values.size())
            all_finite &= dr::all(dr::isfinite(values));
    }
    if (!dr::slice(all_finite)) {
        // Retain the precise offending group in errors. Extra synchronization
        // is confined to the rejected transaction; no parameter has changed.
        for (const auto &[key, values] : parameters)
            if (values.size() && !dr::slice(dr::all(dr::isfinite(values))))
                Throw("VoxelReconstruction.set_parameters(): non-finite %s.", key.c_str());
    }
    for (const auto &[key, values] : parameters) {
        if (key == "centers") m_centers = dr::detach(values);
        else if (key == "log_scales") m_log_scales = dr::detach(values);
        else if (key == "rotations") m_rotations = dr::detach(values);
        else if (key == "radiance_sh") m_radiance = dr::detach(values);
        else if (key == "opacity_sh") m_opacity = dr::detach(values);
    }
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Mask
VoxelReconstruction<Float, Spectrum>::set_parameters_guarded(
    const std::map<std::string, FloatBuffer> &parameters, const Mask &allowed) {
    if (dr::width(allowed) != 1)
        Throw("VoxelReconstruction.set_parameters_guarded(): allowed must be a scalar mask.");
    const std::map<std::string, std::pair<size_t, FloatBuffer *>> destinations = {
        {"centers", {size_t(m_active_count)*3, &m_centers}},
        {"log_scales", {size_t(m_active_count)*3, &m_log_scales}},
        {"rotations", {size_t(m_active_count)*4, &m_rotations}},
        {"radiance_sh", {size_t(m_active_count)*m_sh_count*3, &m_radiance}},
        {"opacity_sh", {size_t(m_active_count)*m_opacity_sh_count, &m_opacity}}};
    Mask accepted = allowed;
    for (const auto &[key, values] : parameters) {
        auto target = destinations.find(key);
        if (target == destinations.end() || values.size() != target->second.first)
            Throw("VoxelReconstruction.set_parameters_guarded(): unknown key or wrong size: %s.",
                  key.c_str());
        if (values.size())
            accepted &= dr::all(dr::isfinite(values));
    }
    // Both the parameters and the caller's Adam buffers use this SAME scalar
    // predicate. Selection happens on device and rejects the entire update,
    // including finite groups beside an invalid group. The caller latches a
    // rejected update and reports it at its next explicit reporting boundary.
    for (const auto &[key, values] : parameters) {
        if (values.size()) {
            FloatBuffer &target = *destinations.at(key).second;
            target = dr::select(accepted, dr::detach(values), target);
        }
    }
    return accepted;
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Mask
VoxelReconstruction<Float, Spectrum>::set_parameters_sparse(
    const UIntBuffer &appearance_indices, const UIntBuffer &geometry_indices,
    const std::map<std::string, FloatBuffer> &parameters, const Mask &allowed) {
    if (dr::width(allowed) != 1)
        Throw("VoxelReconstruction.set_parameters_sparse(): allowed must be a scalar mask.");
    struct Destination {
        uint32_t width;
        FloatBuffer *buffer;
        const UIntBuffer *rows;
    };
    // Keep pointers here: copying target buffers would force a full-size copy
    // on write even when only a few compact rows participate in this update.
    const std::map<std::string, Destination> destinations = {
        {"centers", {3, &m_centers, &geometry_indices}},
        {"log_scales", {3, &m_log_scales, &geometry_indices}},
        {"rotations", {4, &m_rotations, &geometry_indices}},
        {"radiance_sh", {3 * m_sh_count, &m_radiance, &appearance_indices}},
        {"opacity_sh", {m_opacity_sh_count, &m_opacity, &appearance_indices}}};
    for (const auto &[key, values] : parameters) {
        auto target = destinations.find(key);
        if (target == destinations.end() ||
            values.size() != target->second.rows->size() * target->second.width)
            Throw("VoxelReconstruction.set_parameters_sparse(): unknown key or wrong compact size: %s.",
                  key.c_str());
    }

    Mask accepted = allowed;
    for (const UIntBuffer *rows : {&appearance_indices, &geometry_indices}) {
        if (!rows->size())
            continue;
        UIntBuffer ordinal = dr::arange<UIntBuffer>(rows->size());
        UIntBuffer previous = dr::gather<UIntBuffer>(*rows, ordinal - 1, ordinal > 0);
        accepted &= dr::all((*rows < m_active_count) &
                            ((ordinal == 0) | (*rows > previous)));
    }
    for (const auto &[key, values] : parameters) {
        if (values.size())
            accepted &= dr::all(dr::isfinite(values));
        dr::schedule(values);
    }
    // Resolve the complete guard and every compact proposal before mutating
    // any destination. This both makes rejection atomic across all groups and
    // breaks lazy gather dependencies on the old model buffers. No CPU readback
    // or full target selection/finite scan is needed.
    dr::schedule(accepted);
    dr::eval();
    for (const auto &[key, values] : parameters) {
        if (!values.size())
            continue;
        const Destination &target = destinations.at(key);
        // Dr.Jit rejects even a fully masked scatter into an empty target.
        // Any nonempty proposal for an empty model already has accepted=false.
        if (!target.buffer->size())
            continue;
        UIntBuffer component = dr::arange<UIntBuffer>(values.size());
        UIntBuffer address = dr::gather<UIntBuffer>(*target.rows, component / target.width)
                           * target.width + component % target.width;
        dr::scatter(*target.buffer, dr::detach(values), address, accepted);
    }
    return accepted;
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Mask
VoxelReconstruction<Float, Spectrum>::optimizer_step_device(
    const std::map<std::string, FloatBuffer> &gradients,
    const UIntBuffer &appearance_valid, const UIntBuffer &geometry_valid,
    const UIntBuffer &background_carve_valid,
    const std::map<std::string, FloatBuffer *> &means,
    const std::map<std::string, FloatBuffer *> &squared_means,
    const std::map<std::string, UIntBuffer *> &steps,
    const std::map<std::string, double> &settings, const Mask &allowed,
    bool check_moments) {
    return optimizer_step_device_impl(gradients, appearance_valid, geometry_valid,
        background_carve_valid, means, squared_means, steps, settings, allowed,
        check_moments, nullptr, nullptr);
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Mask
VoxelReconstruction<Float, Spectrum>::optimizer_step_compact_device(
    const UIntBuffer &indices, const UInt32 &row_count,
    const std::map<std::string, FloatBuffer> &gradients,
    const UIntBuffer &appearance_valid, const UIntBuffer &geometry_valid,
    const UIntBuffer &background_carve_valid,
    const std::map<std::string, FloatBuffer *> &means,
    const std::map<std::string, FloatBuffer *> &squared_means,
    const std::map<std::string, UIntBuffer *> &steps,
    const std::map<std::string, double> &settings, const Mask &allowed,
    bool check_moments) {
    return optimizer_step_device_impl(gradients, appearance_valid, geometry_valid,
        background_carve_valid, means, squared_means, steps, settings, allowed,
        check_moments, &indices, &row_count);
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Mask
VoxelReconstruction<Float, Spectrum>::optimizer_step_device_impl(
    const std::map<std::string, FloatBuffer> &gradients,
    const UIntBuffer &appearance_valid, const UIntBuffer &geometry_valid,
    const UIntBuffer &background_carve_valid,
    const std::map<std::string, FloatBuffer *> &means,
    const std::map<std::string, FloatBuffer *> &squared_means,
    const std::map<std::string, UIntBuffer *> &steps,
    const std::map<std::string, double> &settings, const Mask &allowed,
    bool check_moments, const UIntBuffer *compact_indices,
    const UInt32 *compact_count) {
    if constexpr (!dr::is_jit_v<Float>) {
        Throw("VoxelReconstruction.optimizer_step_device(): requires a JIT variant.");
    } else {
        auto setting = [&](const char *name) {
            auto found = settings.find(name);
            if (found == settings.end() || !std::isfinite(found->second))
                Throw("VoxelReconstruction.optimizer_step_device(): invalid setting %s.", name);
            return float(found->second);
        };
        const float beta1 = setting("beta1"), beta2 = setting("beta2"),
                    epsilon = setting("epsilon"), clip = setting("gradient_clip"),
                    gradient_factor = setting("gradient_factor"),
                    carve_scale = std::max(setting("background_carve_multiplier"), 1.f) - 1.f,
                    max_scale = setting("max_scale_voxels"),
                    rotation_limit = setting("rotation_step_limit");
        // Python evaluates these scalar differences in double precision before
        // converting them to Float. Match that order (not 1.f - float(beta)).
        const float weight1 = float(1. - settings.at("beta1")),
                    weight2 = float(1. - settings.at("beta2"));
        const bool direction_guard = setting("geometry_direction_guard") != 0.f;
        struct Group {
            const char *name;
            uint32_t size;
            bool geometry;
            FloatBuffer *parameter;
            float rate, limit;
        };
        const std::array<Group, 5> groups = {{
            {"radiance_sh", m_sh_count * 3, false, &m_radiance,
                setting("lr_radiance"), setting("radiance_step_limit")},
            {"opacity_sh", m_opacity_sh_count, false, &m_opacity,
                setting("lr_opacity"), setting("opacity_step_limit")},
            {"centers", 3, true, &m_centers,
                setting("lr_center"), setting("center_step_limit")},
            {"log_scales", 3, true, &m_log_scales,
                setting("lr_log_scale"), setting("log_scale_step_limit")},
            {"rotation_tangent", 3, true, &m_rotations,
                setting("lr_rotation"), rotation_limit}
        }};
        const size_t input_rows = compact_indices ? compact_indices->size() : m_active_count;
        const uint32_t radiance_degrees = uint32_t(std::ceil(std::sqrt(double(m_sh_count))));
        const uint32_t opacity_degrees = uint32_t(std::ceil(std::sqrt(double(m_opacity_sh_count))));
        if (dr::width(allowed) != 1 || appearance_valid.size() != input_rows ||
            geometry_valid.size() != input_rows || background_carve_valid.size() != input_rows ||
            (compact_indices && (!compact_count || dr::width(*compact_count) != 1)))
            Throw("VoxelReconstruction.optimizer_step_device(): invalid mask/count sizes.");
        for (const Group &group : groups) {
            const size_t size = size_t(m_active_count) * group.size;
            auto mean = means.find(group.name), square = squared_means.find(group.name);
            auto clock = steps.find(group.name);
            auto gradient = gradients.find(group.name);
            if (mean == means.end() || square == squared_means.end() || clock == steps.end() ||
                !mean->second || !square->second || !clock->second ||
                mean->second->size() != size || square->second->size() != size ||
                (clock->second->size() != m_active_count &&
                 !(std::string(group.name) == "radiance_sh" &&
                   clock->second->size() == size_t(m_active_count) * radiance_degrees) &&
                 !(std::string(group.name) == "opacity_sh" &&
                   clock->second->size() == size_t(m_active_count) * opacity_degrees)) ||
                (group.rate > 0.f && (gradient == gradients.end() ||
                                     gradient->second.size() != input_rows * group.size)))
                Throw("VoxelReconstruction.optimizer_step_device(): invalid group %s.", group.name);
        }
        Mask accepted = allowed;
        if (compact_indices) {
            if (!input_rows)
                return accepted & (*compact_count == 0u);
            UIntBuffer ordinal = dr::arange<UIntBuffer>(input_rows);
            auto present = ordinal < *compact_count;
            auto in_bounds = *compact_indices < m_active_count;
            UIntBuffer previous = dr::gather<UIntBuffer>(*compact_indices, ordinal - 1u,
                                                        present & (ordinal > 0u));
            accepted &= (!present | (in_bounds & ((ordinal == 0u) |
                                                  (*compact_indices > previous)))).all_async_();
            // Do not capture host input_rows in this comparison: frozen calls
            // can receive a different K. This IDs-only kernel infers its width
            // from the compact input, independently of the full model buffers.
            // Keep this reduction dependent on runtime IDs, not just arange():
            // a literal capacity would be captured by freeze when K changes.
            // Only the valid prefix participates, so arbitrary tail IDs are
            // ignored while an over-capacity count still rejects atomically.
            UInt32 actual_rows = dr::sum(UIntBuffer(present & in_bounds));
            accepted &= *compact_count == actual_rows;
            dr::eval(accepted);
        }
        if (!m_active_count)
            return accepted;

        // Inclusive scan gives stable IDs and the device-side list length.
        // Unlike compress(), neither list length is fetched by the CPU. The
        // spare tail of each list is never addressed by the bounded loops.
        auto compact = [&](const UIntBuffer &counts, bool enabled) {
            if (!enabled)
                return std::make_pair(dr::zeros<UIntBuffer>(1), UInt32(0));
            UIntBuffer ordinal = dr::arange<UIntBuffer>(m_active_count);
            auto active = (counts > 0u) & allowed;
            UIntBuffer prefix = dr::prefix_sum(UIntBuffer(active), false);
            UInt32 count = dr::gather<UInt32>(prefix, UInt32(m_active_count - 1));
            UIntBuffer rows = dr::zeros<UIntBuffer>(m_active_count);
            dr::scatter(rows, ordinal, prefix - 1u, active);
            dr::eval(rows, count);
            return std::make_pair(std::move(rows), std::move(count));
        };
        std::pair<UIntBuffer, UInt32> appearance, geometry;
        if (!compact_indices) {
            appearance = compact(appearance_valid, groups[0].rate > 0.f || groups[1].rate > 0.f);
            geometry = compact(geometry_valid, groups[2].rate > 0.f || groups[3].rate > 0.f ||
                                               groups[4].rate > 0.f);
        }
        const uint32_t batch_size = std::min(m_active_count, uint32_t(32768));
        const UInt32 lanes = dr::arange<UInt32>(batch_size);
        const auto clean = [](const Float &value) {
            return dr::select(dr::isfinite(value), value, 0.f);
        };
        // Never fall back to an evaluated loop: that would read the loop's
        // termination condition on the CPU at every chunk.
        dr::scoped_set_flag symbolic_loops(JitFlag::SymbolicLoops, true);

        // The transaction is validated first, then recomputed and committed.
        // This avoids both full-table rollback copies and N*SH proposal arrays.
        // All groups share the final scalar gate, including late-chunk errors.
        for (unsigned pass = 0; pass < 2; ++pass) {
            if (pass == 1)
                dr::eval(accepted);
            for (const Group &group : groups) {
                if (group.rate <= 0.f)
                    continue;
                const bool rotation = std::string(group.name) == "rotation_tangent";
                const bool radiance = std::string(group.name) == "radiance_sh";
                const bool opacity = std::string(group.name) == "opacity_sh";
                const uint32_t active_coefficients = radiance ? m_active_sh_count : m_active_opacity_sh_count;
                const uint32_t degree_count = radiance ? radiance_degrees : opacity_degrees;
                const bool centers = std::string(group.name) == "centers";
                const bool scales = std::string(group.name) == "log_scales";
                const auto &compact_rows = group.geometry ? geometry : appearance;
                const UIntBuffer &rows = compact_indices ? *compact_indices : compact_rows.first;
                const UInt32 &count = compact_indices ? *compact_count : compact_rows.second;
                const UIntBuffer &validity = group.geometry ? geometry_valid : appearance_valid;
                const FloatBuffer &gradient = gradients.at(group.name);
                FloatBuffer &mean_target = *means.at(group.name),
                            &square_target = *squared_means.at(group.name);
                UIntBuffer &clock_target = *steps.at(group.name);
                const uint32_t component_count = radiance ? m_active_sh_count * 3 : (opacity ? m_active_opacity_sh_count : group.size);
                // Legacy callers retain one voxel clock. Progressive training
                // supplies voxel-major degree clocks so a newly enabled degree
                // starts its own Adam bias correction without altering DC age.
                const uint32_t clock_width = clock_target.size() == m_active_count ? 1 : degree_count;
                const uint32_t active_clocks = clock_width == 1 ? 1 :
                    uint32_t(std::ceil(std::sqrt(double(active_coefficients))));
                auto component_clock = [&](uint32_t component) {
                    return clock_width == 1 ? 0u : uint32_t(std::sqrt(double(component / (radiance ? 3u : 1u))));
                };
                auto clock_biases = [&](const UIntBuffer &buffer, const UInt32 &ids, const Mask &active) {
                    std::vector<std::tuple<UInt32, Float, Float>> result;
                    result.reserve(active_clocks);
                    for (uint32_t degree = 0; degree < active_clocks; ++degree) {
                        UInt32 next = dr::gather<UInt32>(buffer, ids * clock_width + degree, active) + 1u;
                        Float exponent(next);
                        result.emplace_back(next,
                            dr::maximum(1.f - dr::pow(Float(beta1), exponent), 1e-8f),
                            dr::maximum(1.f - dr::pow(Float(beta2), exponent), 1e-8f));
                    }
                    return result;
                };

                auto row_scale = [&](const UInt32 &ids, const Mask &active) {
                    if (!group.geometry)
                        return Float(1.f);
                    Float carve = Float(dr::gather<UInt32>(background_carve_valid, ids, active));
                    Float hits = Float(dr::gather<UInt32>(geometry_valid, ids, active));
                    return 1.f + carve_scale * dr::clip(carve / dr::maximum(hits, 1.f), 0.f, 1.f);
                };
                auto adam = [&](const UInt32 &gradient_index, const UInt32 &index, const Mask &active,
                                const FloatBuffer &mean_buffer, const FloatBuffer &square_buffer,
                                const Float &bias1, const Float &bias2, const Float &scale) {
                    Float original = clean(dr::gather<Float>(gradient, gradient_index, active) * gradient_factor);
                    Float grad = dr::clip(original, -clip, clip);
                    Float mean = beta1 * dr::gather<Float>(mean_buffer, index, active) + weight1 * grad;
                    Float square = beta2 * dr::gather<Float>(square_buffer, index, active) +
                                   weight2 * dr::square(grad);
                    Float delta = clean(group.rate * (mean / bias1) /
                        (dr::sqrt(dr::maximum(square / bias2, 0.f)) + epsilon));
                    if (group.geometry && direction_guard) {
                        Mask blocked = (original == 0.f) | (mean * original < 0.f);
                        mean = dr::select(blocked, 0.f, mean);
                        delta = dr::select(blocked, 0.f, delta);
                    }
                    return std::make_tuple(mean, square, delta * scale);
                };
                auto update_component = [&](const FloatBuffer &parameter, const UInt32 &ids,
                                            uint32_t component, const Float &delta, const Mask &active) {
                    Float old = dr::gather<Float>(parameter, ids * group.size + component, active);
                    Float result = old - dr::clip(delta, -group.limit, group.limit);
                    if (!group.geometry)
                        result = clean(result);
                    else if (centers) {
                        result = dr::clip(result,.01f,.99f);
                    }
                    else if (scales) {
                        Float size = leaf_size(ids, active)[component];
                        Float lower = dr::maximum(size * 1e-4f, 1e-8f);
                        Float upper = dr::maximum(size * max_scale, lower);
                        result = dr::clip(clean(result), dr::log(lower), dr::log(upper));
                    }
                    return result;
                };
                auto update_rotation = [&](const FloatBuffer &parameter, const UInt32 &ids,
                                           const std::array<Float, 3> &delta, const Mask &active) {
                    Vector3f tangent(-delta[0], -delta[1], -delta[2]);
                    Float length = dr::norm(tangent);
                    Float factor = dr::minimum(1.f, rotation_limit / dr::maximum(length, 1e-30f));
                    tangent *= factor;
                    length *= factor;
                    Float scale = dr::select(length > 1e-8f,
                        dr::sin(length * .5f) / dr::maximum(length, 1e-30f), 0.f);
                    Vector3f vector = tangent * scale;
                    Float scalar = dr::select(length > 1e-8f, dr::cos(length * .5f), 1.f);
                    Float old_w = dr::gather<Float>(parameter, ids * 4, active);
                    Vector3f old_v;
                    for (uint32_t c = 0; c < 3; ++c)
                        old_v[c] = dr::gather<Float>(parameter, ids * 4 + c + 1, active);
                    Float new_w = old_w * scalar - dr::dot(old_v, vector);
                    Vector3f new_v = old_w * vector + scalar * old_v + dr::cross(old_v, vector);
                    Float norm_squared = dr::square(new_w) + dr::dot(new_v, new_v);
                    Mask safe = dr::isfinite(norm_squared) & (norm_squared > 1e-20f);
                    Float inv_norm = dr::rsqrt(dr::select(safe, norm_squared, 1.f));
                    return std::array<Float, 4>{
                        dr::select(safe, new_w * inv_norm, 1.f),
                        dr::select(safe, new_v.x() * inv_norm, 0.f),
                        dr::select(safe, new_v.y() * inv_norm, 0.f),
                        dr::select(safe, new_v.z() * inv_norm, 0.f)};
                };

                if (pass == 0) {
                    struct ValidateState {
                        UInt32 ordinal;
                        Mask finite;
                        DRJIT_STRUCT(ValidateState, ordinal, finite)
                    } state{lanes, dr::full<Mask>(true, batch_size)};
                    dr::tie(state) = dr::while_loop(dr::make_tuple(std::move(state)),
                        [&](const ValidateState &s) { return (s.ordinal < count) & accepted; },
                        [&](ValidateState &s) {
                            Mask active = (s.ordinal < count) & accepted;
                            UInt32 ids = dr::gather<UInt32>(rows, s.ordinal, active);
                            UInt32 input_row = compact_indices ? s.ordinal : ids;
                            if (compact_indices)
                                active &= dr::gather<UInt32>(validity, input_row, active) > 0u;
                            // Legacy occupied-cell backgrounds retain fixed geometry.
                            if (group.geometry && m_background_shell_levels && !m_background_use_ellipsoid)
                                active &= core_ids(ids, active);
                            auto clocks = clock_biases(clock_target, ids, active);
                            Float scale = row_scale(input_row, active);
                            std::array<Float, 3> rotation_delta;
                            for (uint32_t component = 0; component < component_count; ++component) {
                                const auto &[next_clock, bias1, bias2] = clocks[component_clock(component)];
                                auto [mean, square, delta] = adam(input_row * group.size + component,
                                    ids * group.size + component,
                                    active, mean_target, square_target, bias1, bias2, scale);
                                if (check_moments)
                                    s.finite &= !active | (dr::isfinite(mean) & dr::isfinite(square));
                                if (rotation)
                                    rotation_delta[component] = delta;
                                else
                                    s.finite &= !active | dr::isfinite(update_component(
                                        *group.parameter, ids, component, delta, active));
                            }
                            if (rotation)
                                for (const Float &value : update_rotation(*group.parameter, ids, rotation_delta, active))
                                    s.finite &= !active | dr::isfinite(value);
                            s.ordinal += batch_size;
                        }, "Voxel reconstruction Adam validation");
                    // C++ dr::all() returns a host bool and synchronizes; the
                    // asynchronous primitive retains the reduction on device.
                    accepted &= state.finite.all_async_();
                } else {
                    // A length-one gather is optimized by Dr.Jit into a
                    // direct scalar input, rather than a pointer load. Writing
                    // that same scalar inside the kernel would then make its
                    // input dirty. Preserve only these four-byte read sources;
                    // ordinary multi-voxel tables need no extra storage/copy.
                    FloatBuffer scalar_mean, scalar_square, scalar_parameter;
                    UIntBuffer scalar_clock;
                    if (mean_target.size() == 1)
                        scalar_mean = FloatBuffer::steal(jit_var_copy_opaque(mean_target.index()));
                    if (square_target.size() == 1)
                        scalar_square = FloatBuffer::steal(jit_var_copy_opaque(square_target.index()));
                    if (group.parameter->size() == 1)
                        scalar_parameter = FloatBuffer::steal(jit_var_copy_opaque(group.parameter->index()));
                    if (clock_target.size() == 1)
                        scalar_clock = UIntBuffer::steal(jit_var_copy_opaque(clock_target.index()));
                    struct CommitState {
                        UInt32 ordinal;
                        FloatBuffer mean, square, parameter;
                        UIntBuffer clock;
                        DRJIT_STRUCT(CommitState, ordinal, mean, square, parameter, clock)
                    } state{lanes, std::move(mean_target), std::move(square_target),
                            std::move(*group.parameter), std::move(clock_target)};
                    // Move owners into/out of the loop state. Holding an extra
                    // array copy here would force full-table COW on every view.
                    dr::tie(state) = dr::while_loop(dr::make_tuple(std::move(state)),
                        [&](const CommitState &s) { return (s.ordinal < count) & accepted; },
                        [&](CommitState &s) {
                            Mask active = (s.ordinal < count) & accepted;
                            UInt32 ids = dr::gather<UInt32>(rows, s.ordinal, active);
                            UInt32 input_row = compact_indices ? s.ordinal : ids;
                            if (compact_indices)
                                active &= dr::gather<UInt32>(validity, input_row, active) > 0u;
                            if (group.geometry && m_background_shell_levels && !m_background_use_ellipsoid)
                                active &= core_ids(ids, active);
                            auto clocks = clock_biases(scalar_clock.size() ? scalar_clock : s.clock, ids, active);
                            Float scale = row_scale(input_row, active);
                            std::array<Float, 3> rotation_delta;
                            std::vector<Float> next_means, next_squares, next_parameters;
                            next_means.reserve(component_count);
                            next_squares.reserve(component_count);
                            next_parameters.reserve(rotation ? 4 : component_count);
                            for (uint32_t component = 0; component < component_count; ++component) {
                                const auto &[next_clock, bias1, bias2] = clocks[component_clock(component)];
                                UInt32 index = ids * group.size + component;
                                auto [mean, square, delta] = adam(input_row * group.size + component, index, active,
                                    scalar_mean.size() ? scalar_mean : s.mean,
                                    scalar_square.size() ? scalar_square : s.square, bias1, bias2, scale);
                                next_means.push_back(std::move(mean));
                                next_squares.push_back(std::move(square));
                                if (rotation)
                                    rotation_delta[component] = delta;
                                else
                                    next_parameters.push_back(update_component(
                                        scalar_parameter.size() ? scalar_parameter : s.parameter,
                                        ids, component, delta, active));
                            }
                            if (rotation) {
                                auto values = update_rotation(s.parameter, ids, rotation_delta, active);
                                for (uint32_t component = 0; component < 4; ++component)
                                    next_parameters.push_back(std::move(values[component]));
                            }
                            // Read all old components before starting writes.
                            // Compact IDs are unique, and different coefficient
                            // addresses are disjoint, so NoConflicts is exact.
                            for (uint32_t component = 0; component < component_count; ++component) {
                                UInt32 index = ids * group.size + component;
                                dr::scatter(s.mean, dr::detach(next_means[component]), index, active, ReduceMode::NoConflicts);
                                dr::scatter(s.square, dr::detach(next_squares[component]), index, active, ReduceMode::NoConflicts);
                            }
                            for (uint32_t component = 0; component < next_parameters.size(); ++component)
                                dr::scatter(s.parameter, dr::detach(next_parameters[component]),
                                    ids * (rotation ? 4u : group.size) + component, active, ReduceMode::NoConflicts);
                            for (uint32_t degree = 0; degree < active_clocks; ++degree)
                                dr::scatter(s.clock, std::get<0>(clocks[degree]), ids * clock_width + degree,
                                            active, ReduceMode::NoConflicts);
                            s.ordinal += batch_size;
                        }, "Voxel reconstruction Adam commit");
                    mean_target = std::move(state.mean);
                    square_target = std::move(state.square);
                    *group.parameter = std::move(state.parameter);
                    clock_target = std::move(state.clock);
                    dr::schedule(mean_target, square_target, *group.parameter, clock_target);
                }
            }
        }
        // Buffers above have already been scheduled. eval(accepted) alone can
        // be a no-op when the scalar gate is evaluated, leaving scatter side
        // effects pending when the next renderer records gathers from them.
        dr::schedule(accepted);
        dr::eval();
        return accepted;
    }
}

MI_VARIANT std::vector<uint64_t>
VoxelReconstruction<Float, Spectrum>::optimizer_update_scope_begin(
    const std::map<std::string, FloatBuffer *> &means,
    const std::map<std::string, FloatBuffer *> &squared_means,
    const std::map<std::string, UIntBuffer *> &steps) {
    std::vector<uint64_t> handles;
    if constexpr (!dr::is_jit_v<Float>) {
        Throw("VoxelReconstruction.optimizer_update_scope_begin(): requires a JIT variant.");
    } else {
        std::vector<FloatBuffer *> floats{&m_radiance, &m_opacity, &m_centers, &m_log_scales, &m_rotations};
        std::vector<UIntBuffer *> clocks;
        for (const auto &[name, value] : means) {
            if (!value)
                Throw("VoxelReconstruction.optimizer_update_scope_begin(): null moment %s.", name.c_str());
            floats.push_back(value);
        }
        for (const auto &[name, value] : squared_means) {
            if (!value)
                Throw("VoxelReconstruction.optimizer_update_scope_begin(): null moment %s.", name.c_str());
            floats.push_back(value);
        }
        for (const auto &[name, value] : steps) {
            if (!value)
                Throw("VoxelReconstruction.optimizer_update_scope_begin(): null clock %s.", name.c_str());
            clocks.push_back(value);
        }
        for (FloatBuffer *value : floats)
            dr::schedule(*value);
        for (UIntBuffer *value : clocks)
            dr::schedule(*value);
        dr::eval();
        auto make_unique = [](auto &value) {
            using Buffer = std::decay_t<decltype(value)>;
            if (!value.size())
                return;
            // Literals may share their JIT node even before any Python alias
            // exists. Materialize each writable buffer independently once.
            if (jit_var_ref(value.index()) > 1 || value.state() != VarState::Evaluated)
                value = Buffer::steal(jit_var_copy_opaque(value.index()));
        };
        for (FloatBuffer *value : floats)
            make_unique(*value);
        for (UIntBuffer *value : clocks)
            make_unique(*value);
        // Genuine user snapshots/moment aliases have now retained their old
        // allocation. Frozen tracing may add references after this point, but
        // those bookkeeping references do not require new device copies.
        handles.reserve(floats.size() + clocks.size());
        try {
            for (FloatBuffer *value : floats)
                if (value->size())
                    handles.push_back(jit_var_stash_ref(value->index()));
            for (UIntBuffer *value : clocks)
                if (value->size())
                    handles.push_back(jit_var_stash_ref(value->index()));
        } catch (...) {
            for (uint64_t handle : handles)
                jit_var_unstash_ref(handle);
            throw;
        }
    }
    return handles;
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::clear_growth_candidates() {
    if (m_growth_count) {
        m_growth_tree.clear(root_resolution());
        update_spatial_statistics();
        ++m_topology_revision;
    }
    m_growth_count = 0;
    m_growth_cells = dr::zeros<CellBuffer>(1);
    m_growth_next = dr::zeros<UIntBuffer>(1);
    m_growth_centers = m_growth_log_scales = m_growth_rotations =
        m_growth_radiance = m_growth_opacity = dr::zeros<FloatBuffer>(1);
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_growth_candidates(
    const CellBuffer &cell_indices, const std::map<std::string, FloatBuffer> &parameters) {
    const size_t count = cell_indices.size();
    if (count > size_t(INT32_MAX) - 1)
        Throw("VoxelReconstruction.set_growth_candidates(): too many candidates.");
    const std::map<std::string, size_t> sizes = {
        {"centers", count*3}, {"log_scales", count*3}, {"rotations", count*4},
        {"radiance_sh", count*m_sh_count*3}, {"opacity_sh", count*m_opacity_sh_count}};
    if (parameters.size() != sizes.size())
        Throw("VoxelReconstruction.set_growth_candidates(): requires all five parameter groups.");
    Mask finite = true;
    for (const auto &[name, size] : sizes) {
        auto it = parameters.find(name);
        if (it == parameters.end() || it->second.size() != size)
            Throw("VoxelReconstruction.set_growth_candidates(): missing or wrong size: %s.", name.c_str());
        if (size)
            finite &= dr::all(dr::isfinite(it->second));
    }
    if (!dr::slice(finite))
        Throw("VoxelReconstruction.set_growth_candidates(): parameters must be finite.");
    // Check the complete transaction before restoring/replacing the old overlay.
    // Negative entries are hypothetical cells and remain genuinely unoccupied.
    // A target cell may have several parent-specific shapes and appearances.
    // Keep all original row IDs, chaining siblings in input order. Only the
    // unique cell heads are scattered into the overlay, avoiding scatter races.
    validate_cells(cell_indices);
    CellBuffer cells(cell_indices);
    if (count && m_active_count) {
        UIntBuffer node(m_tree.root), level(m_tree.depth);
        dr::uint64_array_t<UIntBuffer> anchor(0);
        auto decoded=decode_cells(cells);
        auto [ids, unused_skip] = m_tree.template lookup<UIntBuffer, IntBuffer>(
            IntBuffer(decoded[0]), IntBuffer(decoded[1]), IntBuffer(decoded[2]), node, level, anchor,
            dr::mask_t<UIntBuffer>(true));
        auto occupied = ids >= 0;
        if (dr::slice(dr::any(occupied)))
            Throw("VoxelReconstruction.set_growth_candidates(): target cells must be empty.");
    }
    // Keep independent buffer references alive even when the caller reuses its
    // dictionary. All accesses are detached, and no gradients target proposals.
    CellBuffer new_cells = dr::zeros<CellBuffer>(1);
    UIntBuffer new_heads = dr::zeros<UIntBuffer>(1), new_next = dr::zeros<UIntBuffer>(1);
    if (count) {
        UIntBuffer rows = dr::arange<UIntBuffer>(count),
                   order = reconstruction_tree_detail::order<UIntBuffer>(cells,64u);
        CellBuffer ordered_cells = dr::gather<CellBuffer>(cells, order),
                   previous_cell = dr::gather<CellBuffer>(ordered_cells, rows - 1, rows > 0);
        auto group_start = (rows == 0) | (ordered_cells != previous_cell);
        // Scatter heads into ORIGINAL row order before compacting: even the
        // order of unique cells stays identical to the previous CPU builder.
        UIntBuffer head_flags = dr::zeros<UIntBuffer>(count);
        dr::scatter(head_flags, UIntBuffer(group_start), order);
        new_heads = dr::compress(head_flags != 0);
        new_cells = dr::gather<CellBuffer>(cells, new_heads);
        auto has_following = rows + 1 < uint32_t(count);
        CellBuffer following_cell = dr::gather<CellBuffer>(ordered_cells, rows + 1, has_following);
        UIntBuffer following_row = dr::gather<UIntBuffer>(order, rows + 1, has_following);
        new_next = dr::empty<UIntBuffer>(count);
        dr::scatter(new_next,
                    dr::select(has_following & (following_cell == ordered_cells),
                               following_row, UINT32_MAX), order);
        dr::eval(new_cells, new_heads, new_next);
    }
    // Construct the replacement before mutating the installed overlay. A
    // failed transaction must leave both the old candidates and replay intact.
    SpatialTree new_tree;
    if (count)
        build_world_tree(new_tree, CellBuffer(new_cells), -2 - IntBuffer(new_heads));
    if (count && m_active_count && (m_background_shell_levels || m_core_adaptive)) {
        // The first lookup rejects a proposal contained by an existing leaf.
        // This converse check rejects a coarse proposal containing a finer leaf
        // away from its minimum corner. Dyadic overlap always has one of these
        // two containment directions; no corner sampling is sufficient.
        auto decoded=decode_cells(m_cells);
        UIntBuffer node(new_tree.root), level(new_tree.depth);
        dr::uint64_array_t<UIntBuffer> anchor(0);
        auto [ids, unused_skip]=new_tree.template lookup<UIntBuffer,IntBuffer>(
            IntBuffer(decoded[0]),IntBuffer(decoded[1]),IntBuffer(decoded[2]),
            node,level,anchor,dr::mask_t<UIntBuffer>(true));
        if(dr::slice(dr::any(ids<=-2)))
            Throw("VoxelReconstruction.set_growth_candidates(): target overlaps an occupied descendant.");
    }
    clear_growth_candidates();
    if (!count)
        return;
    m_growth_cells = std::move(cells);
    m_growth_next = std::move(new_next);
    m_growth_centers = dr::detach(parameters.at("centers"));
    m_growth_log_scales = dr::detach(parameters.at("log_scales"));
    m_growth_rotations = dr::detach(parameters.at("rotations"));
    m_growth_radiance = dr::detach(parameters.at("radiance_sh"));
    m_growth_opacity = dr::detach(parameters.at("opacity_sh"));
    m_growth_count = uint32_t(count);
    m_growth_tree = std::move(new_tree);
    update_spatial_statistics();
    ++m_topology_revision;
}

// Falcor's real SH convention: positive y,z,x at degree one. Do not substitute
// a library SH basis without checking its signs and coefficient ordering.
template <typename T> static T reconstruction_sh(uint32_t i, const Vector<T, 3> &p) {
    const T &x = p.x(), &y = p.y(), &z = p.z();
    constexpr float inv_sqrt_pi = .5641895835477563f;
    switch (i) {
        case 0: return T(.5f * inv_sqrt_pi);
        case 1: return y * (.5f * std::sqrt(3.f) * inv_sqrt_pi);
        case 2: return z * (.5f * std::sqrt(3.f) * inv_sqrt_pi);
        case 3: return x * (.5f * std::sqrt(3.f) * inv_sqrt_pi);
        case 4: return x*y * (.5f * std::sqrt(15.f) * inv_sqrt_pi);
        case 5: return y*z * (.5f * std::sqrt(15.f) * inv_sqrt_pi);
        case 6: return (3.f*z*z-1.f) * (.25f * std::sqrt(5.f) * inv_sqrt_pi);
        case 7: return x*z * (.5f * std::sqrt(15.f) * inv_sqrt_pi);
        case 8: return (x*x-y*y) * (.25f * std::sqrt(15.f) * inv_sqrt_pi);
        case 9: return y*(3.f*x*x-y*y) * (.125f * std::sqrt(70.f) * inv_sqrt_pi);
        case 10: return x*y*z * (.5f * std::sqrt(105.f) * inv_sqrt_pi);
        case 11: return y*(5.f*z*z-1.f) * (.125f * std::sqrt(42.f) * inv_sqrt_pi);
        case 12: return z*(5.f*z*z-3.f) * (.25f * std::sqrt(7.f) * inv_sqrt_pi);
        case 13: return x*(5.f*z*z-1.f) * (.125f * std::sqrt(42.f) * inv_sqrt_pi);
        case 14: return z*(x*x-y*y) * (.25f * std::sqrt(105.f) * inv_sqrt_pi);
        case 15: return x*(x*x-3.f*y*y) * (.125f * std::sqrt(70.f) * inv_sqrt_pi);
        default: return T(0.f);
    }
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_active_sh_count(uint32_t count) {
    if (!count || count > m_sh_count)
        Throw("VoxelReconstruction.set_active_sh_count(): count must be in [1, %u].", m_sh_count);
    if (count != m_active_sh_count) {
        m_active_sh_count = count;
        // Frozen rendering captures the active coefficient loop bound.
        ++m_topology_revision;
    }
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_active_opacity_sh_count(uint32_t count) {
    if (!count || count > m_opacity_sh_count)
        Throw("VoxelReconstruction.set_active_opacity_sh_count(): count must be in [1, %u].", m_opacity_sh_count);
    if (count != m_active_opacity_sh_count) {
        m_active_opacity_sh_count = count;
        ++m_topology_revision;
    }
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_scene_mapping(const std::string &mapping) {
    if (mapping == "bounded") {
        set_world_shells(0);
    } else if (mapping == "world_shell") {
        if (!m_background_shell_levels)
            Throw("VoxelReconstruction: set world shell levels before selecting world_shell mapping.");
    } else {
        Throw("VoxelReconstruction: unsupported scene mapping '%s'; compressed background checkpoints are not compatible with world-space shells.", mapping.c_str());
    }
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_world_shells(
    uint32_t levels, uint32_t subdivision_level, bool adaptive) {
    if (levels == m_background_shell_levels && subdivision_level == m_background_subdivision_level &&
        adaptive == m_background_adaptive)
        return;
    const uint32_t previous = m_background_shell_levels,
                   previous_subdivision = m_background_subdivision_level;
    const bool previous_adaptive = m_background_adaptive;
    m_background_shell_levels = levels;
    m_background_subdivision_level = subdivision_level;
    m_background_adaptive = adaptive;
    try {
        SpatialTree replacement;
        build_world_tree(replacement, m_cells, IntBuffer(dr::arange<UIntBuffer>(m_active_count)));
        m_tree = std::move(replacement);
    } catch (...) {
        m_background_shell_levels = previous;
        m_background_subdivision_level = previous_subdivision;
        m_background_adaptive = previous_adaptive;
        throw;
    }
    m_scene_mapping = levels ? "world_shell" : "bounded";
    clear_growth_candidates();
    m_growth_tree.clear(root_resolution());
    update_spatial_statistics();
    ++m_topology_revision;
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_background_use_ellipsoid(bool enabled) {
    if (m_background_use_ellipsoid != enabled) {
        m_background_use_ellipsoid = enabled;
        // Frozen render/gradient/Adam functions capture this semantic choice.
        ++m_topology_revision;
    }
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::build_world_tree(
    SpatialTree &tree, const CellBuffer &cells, const IntBuffer &payload) const {
    if (!m_background_shell_levels && !m_core_adaptive) {
        if (m_background_subdivision_level || m_background_adaptive)
            Throw("VoxelReconstruction: background subdivision requires world shells.");
        const uint64_t core_count = uint64_t(m_resolution)*m_resolution*m_resolution;
        if (cells.size() && !dr::slice(dr::all(cells < core_count)))
            Throw("VoxelReconstruction: invalid bounded cell ID.");
        tree.build(UIntBuffer(cells), payload, m_resolution);
        return;
    }
    if (m_background_shell_levels > 10 || m_resolution < 2 ||
        (m_resolution & (m_resolution - 1)) || m_core_resolution != m_resolution)
        Throw("VoxelReconstruction: world shells require 1..10 levels and a power-of-two core resolution in [2, 1024].");
    for (uint32_t j=1;j<3;++j)
        if (std::abs(m_size[j]-m_size[0]) > 1e-5f*m_size[0])
            Throw("VoxelReconstruction: world shells require a cubic core.");
    for (uint32_t j=0;j<3;++j)
        if (!std::isfinite(m_size[j]) || !std::isfinite(root_min(j)) || !std::isfinite(root_max(j)))
            Throw("VoxelReconstruction: world-shell root bounds must be finite.");
    uint32_t core_log=0;
    while ((1u<<core_log)<m_resolution) ++core_log;
    if (m_background_subdivision_level > (m_background_adaptive ? 9u : std::min(core_log-1,9u)))
        Throw("VoxelReconstruction: background subdivision must not exceed min(log2(core_resolution)-1, 9).");
    if (cells.size() > size_t(INT32_MAX))
        Throw("VoxelReconstruction: parameter rows exceed native index capacity.");
    if (!cells.size()) {
        tree.clear(root_resolution());
        return;
    }
    validate_cells(cells);
    auto decoded = decode_cells(cells);
    tree.build_mixed(UIntBuffer(decoded[0]),UIntBuffer(decoded[1]),UIntBuffer(decoded[2]),
                     UIntBuffer(decoded[3]),payload,root_resolution());
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::validate_cells(const CellBuffer &cells) const {
    const uint64_t core_count = uint64_t(m_resolution)*m_resolution*m_resolution;
    const uint64_t descendants = uint64_t(1) << (3u*m_background_subdivision_level);
    const uint64_t legacy_limit = core_count + uint64_t(56)*m_background_shell_levels*descendants;
    auto tagged = (cells >> 63u) != 0u;
    auto core_tagged = !tagged & ((cells & (uint64_t(1) << 61u)) != 0u);
    CellBuffer core_path = cells & uint64_t(0x7fffffffu);
    UIntBuffer core_top(63u-dr::lzcnt(dr::maximum(core_path,CellBuffer(uint64_t(1u)))));
    // A flat bounded model has no virtual core. Reject a tagged overlay rather
    // than interpreting it on the smaller grid (or changing installed bounds).
    auto valid_core = (m_core_adaptive || m_background_adaptive) &
        ((cells & ~((uint64_t(1) << 61u) | uint64_t(0x7fffffffu))) == 0u) &
        (core_top >= 3u) & (core_top <= 30u) & ((core_top % 3u) == 0u);
    auto extended = (cells & (uint64_t(1) << 62u)) != 0u;
    auto deepest = (cells >> 60u) == 0xfu;
    auto parts = adaptive_address_parts(cells);
    const CellBuffer &path = parts[0];
    UIntBuffer coarse(parts[1]);
    UIntBuffer top(63u-dr::lzcnt(dr::maximum(path,CellBuffer(uint64_t(1u)))));
    auto valid_tag = m_background_adaptive && m_background_shell_levels ?
        (path != 0u) & dr::select(deepest, (top == 54u) & (coarse < 560u),
            dr::select(extended, (top >= 30u) & (top <= 51u), top <= 27u)) &
        ((top % 3u) == 0u) & ((top / 3u) <= 9u + coarse / 56u) &
        (coarse < 56u*m_background_shell_levels) : dr::mask_t<UIntBuffer>(false);
    if (cells.size() && !dr::slice(dr::all(dr::select(tagged,valid_tag,dr::select(core_tagged,valid_core,cells < legacy_limit)))))
        Throw("VoxelReconstruction: invalid adaptive or legacy world-shell cell address.");
}

MI_VARIANT template <typename Cell>
dr::uint32_array_t<Cell> VoxelReconstruction<Float, Spectrum>::cell_log_size(const Cell &cells) const {
    using U=dr::uint32_array_t<Cell>;
    if (!m_background_shell_levels && !m_core_adaptive) return U(0u);
    const uint64_t core_count=uint64_t(m_resolution)*m_resolution*m_resolution;
    auto tagged=(cells>>63u)!=0u;
    auto core_tagged=!tagged & ((cells & (uint64_t(1)<<61u))!=0u);
    auto outside=(cells>=core_count)&!core_tagged;
    auto parts=adaptive_address_parts(cells);
    Cell core_path=cells & uint64_t(0x7fffffffu);
    U core_depth=U(63u-dr::lzcnt(dr::maximum(core_path,Cell(uint64_t(1u)))))/3u;
    U depth=U(63u-dr::lzcnt(dr::maximum(parts[0],Cell(uint64_t(1u)))))/3u;
    const uint64_t descendants=uint64_t(1)<<(3u*m_background_subdivision_level);
    U coarse(dr::select(tagged,parts[1],
        dr::select(outside,cells-core_count,Cell(uint64_t(0)))/descendants));
    uint32_t tree_log=0, core_log=0;
    while ((1u<<tree_log)<tree_core_resolution()) ++tree_log;
    while ((1u<<core_log)<m_resolution) ++core_log;
    return dr::select(core_tagged,U(tree_log)-core_depth,
        dr::select(outside,coarse/56u+tree_log-1u-
        dr::select(tagged,depth,U(m_background_subdivision_level)),U(tree_log-core_log)));
}

MI_VARIANT template <typename Cell>
std::array<dr::uint32_array_t<Cell>,4> VoxelReconstruction<Float, Spectrum>::decode_cells(const Cell &cells) const {
    using U=dr::uint32_array_t<Cell>;
    if (!m_background_shell_levels && !m_core_adaptive) {
        U core(cells);
        return {core%m_resolution,(core/m_resolution)%m_resolution,
                core/(m_resolution*m_resolution),U(0u)};
    }
    const uint64_t core_count=uint64_t(m_resolution)*m_resolution*m_resolution;
    const uint64_t descendants=uint64_t(1)<<(3u*m_background_subdivision_level);
    auto tagged=(cells>>63u)!=0u;
    auto core_tagged=!tagged & ((cells & (uint64_t(1)<<61u))!=0u);
    auto outside=(cells>=core_count)&!core_tagged;
    Cell address=dr::select(outside,cells-core_count,Cell(uint64_t(0)));
    auto parts=adaptive_address_parts(cells);
    Cell path=dr::select(core_tagged,cells & uint64_t(0x7fffffffu),parts[0]);
    U depth=U(63u-dr::lzcnt(dr::maximum(path,Cell(uint64_t(1u)))))/3u;
    U subdivision=dr::select(tagged|core_tagged,depth,U(m_background_subdivision_level));
    U coarse(dr::select(tagged,parts[1],address/descendants));
    Cell morton=dr::select(tagged|core_tagged,path^(Cell(uint64_t(1u))<<(3u*depth)),address%descendants);
    U shell=coarse/56u, flat=coarse%56u;
    for (uint32_t hole:{21u,22u,25u,26u,37u,38u,41u,42u}) flat+=U(flat>=hole);
    uint32_t tree_log=0;
    while ((1u<<tree_log)<tree_core_resolution()) ++tree_log;
    U log=cell_log_size(cells), block=U(1u)<<log;
    U coarse_block=U(1u)<<(shell+tree_log-1u);
    U low=(U(root_resolution())-(U(tree_core_resolution())<<(shell+1u)))/2u;
    U core(dr::select(outside|core_tagged,Cell(uint64_t(0)),cells));
    U core_xyz[3]={core%m_resolution,(core/m_resolution)%m_resolution,core/(m_resolution*m_resolution)};
    std::array<U,4> result;
    for(uint32_t axis=0;axis<3;++axis) {
        U child(0u);
        for(uint32_t bit=0;bit<18;++bit)
            child|=dr::select(bit<subdivision,U((morton>>(3u*bit+axis))&1u)<<bit,U(0u));
        result[axis]=dr::select(core_tagged,child*block+core_offset(),
            dr::select(outside,low+((flat>>(2u*axis))&3u)*coarse_block+child*block,
            core_xyz[axis]*(tree_core_resolution()/m_resolution)+core_offset()));
    }
    result[3]=log;
    return result;
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Mask
VoxelReconstruction<Float, Spectrum>::core_ids(const UInt32 &id, const Mask &active) const {
    if (!m_background_shell_levels) return active;
    CellIndex cells=dr::gather<CellIndex>(m_cells,id,active);
    return active & ((cells < uint64_t(m_resolution)*m_resolution*m_resolution) |
        (((cells>>63u)==0u) & ((cells & (uint64_t(1)<<61u))!=0u)));
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::UInt32
VoxelReconstruction<Float, Spectrum>::leaf_log_size(const UInt32 &id, const Mask &active) const {
    return cell_log_size(dr::gather<CellIndex>(m_cells,id,active));
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Vector3f
VoxelReconstruction<Float, Spectrum>::leaf_size(const UInt32 &id, const Mask &active) const {
    Float factor(UInt32(1u) << leaf_log_size(id, active));
    return Vector3f(tree_cell_size(0), tree_cell_size(1), tree_cell_size(2))*factor;
}

MI_VARIANT std::pair<Float, Float> VoxelReconstruction<Float, Spectrum>::world_depth(const Float &distance) const {
    const float half=.5f*(m_max[0]-m_min[0]);
    Float denominator=distance+half;
    return {distance/denominator, half/dr::square(denominator)};
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Vector3f
VoxelReconstruction<Float, Spectrum>::world_block_boundary_times(
    const Vector3i &cell,const Vector3i &step,const Point3f &origin,
    const Vector3f &inv_d,const UInt32 &log_size) const {
    Vector3f result;
    for (uint32_t a=0;a<3;++a) {
        Int32 low=Int32((UInt32(cell[a])>>log_size)<<log_size);
        Int32 face=low+dr::select(step[a]>0,Int32(UInt32(1u)<<log_size),0);
        Float plane=dr::fmadd(Float(face)-float(core_offset()),tree_cell_size(a),m_min[a]);
        result[a]=dr::select(step[a]!=0,(plane-origin[a])*inv_d[a],dr::Infinity<Float>);
    }
    return result;
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::advance_world_block(
    Vector3i &cell,Vector3f &next,Float &entry,UInt32 &iterations,
    const Vector3i &step,const Point3f &origin,const Vector3f &direction,
    const Vector3f &inv_d,const Float &exit,const UInt32 &log_size) const {
    Vector3f boundary=world_block_boundary_times(cell,step,origin,inv_d,log_size);
    Float target=dr::maximum(entry,dr::minimum(exit,dr::min(boundary)));
    Point3f point=dr::fmadd(direction,target,origin);
    for (uint32_t a=0;a<3;++a) {
        Int32 low=Int32((UInt32(cell[a])>>log_size)<<log_size);
        Int32 high=low+Int32(UInt32(1u)<<log_size)-1;
        // Floor in the core-local coordinate system before adding the integer
        // root offset. A depth-20 root otherwise rounds near-face core points
        // into the adjacent finest cell before floor() can classify them.
        Float position=(point[a]-m_min[a])/tree_cell_size(a);
        Int32 interior=Int32(dr::floor(dr::clip(position,
            Float(low-int(core_offset())),Float(high-int(core_offset())))))+int(core_offset());
        cell[a]=dr::select(boundary[a]<=target,
            dr::select(step[a]>0,high+1,low-1),interior);
    }
    entry=target;
    next=dr::maximum(entry,octree_boundary_times(cell,step,origin,inv_d));
    ++iterations;
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::advance_ray_cell(
    Vector3i &cell,Vector3f &next,Float &entry,UInt32 &iterations,
    const Vector3i &step,const Vector3f &step_t,const Point3f &origin,
    const Vector3f &direction,const Vector3f &inv_d,const Float &exit,
    const Float &end,const UInt32 &log_size,bool use_octree) const {
    // The adaptive tree uses a 1024-cell virtual core even before the foreground
    // reaches that resolution. Its foreground leaves must still visit x/y/z
    // tied faces one at a time, exactly like the original foreground unit grid.
    // Only background leaves and larger empty blocks use simultaneous exits.
    uint32_t foreground_log=0;
    while ((1u<<foreground_log)<tree_core_resolution()/m_resolution) ++foreground_log;
    Mask foreground_unit=(m_core_adaptive || m_background_adaptive) &
        (foreground_log>0u) & (log_size==foreground_log) & dr::all(cell>=int(core_offset())) &
        dr::all(cell<int(core_offset()+tree_core_resolution()));
    Mask coarse=(m_background_shell_levels!=0 || m_core_adaptive) & (log_size>0u) & !foreground_unit;
    dr::tie(cell,next,entry,iterations)=dr::if_stmt(
        dr::make_tuple(cell,next,entry,iterations),coarse,
        [this,step,origin,direction,inv_d,exit,log_size](Vector3i cell,
            Vector3f next,Float entry,UInt32 iterations) {
            advance_world_block(cell,next,entry,iterations,step,origin,
                                direction,inv_d,exit,log_size);
            return dr::make_tuple(cell,next,entry,iterations);
        },
        [this,step,step_t,origin,direction,inv_d,end,use_octree,foreground_unit,log_size](Vector3i cell,
            Vector3f next,Float entry,UInt32 iterations) {
            Mask cross_x=(next.x()<=next.y())&(next.x()<=next.z());
            Mask cross_y=!cross_x&(next.y()<=next.z()),cross_z=!cross_x&!cross_y;
            Mask crossings[3]={cross_x,cross_y,cross_z};
            Point3f point = dr::fmadd(direction,end,origin);
            for (uint32_t a=0;a<3;++a) {
                Int32 lower=Int32((UInt32(cell[a])>>log_size)<<log_size);
                Int32 neighbor=dr::select(step[a]>0,lower+Int32(UInt32(1u)<<log_size),lower-1);
                Int32 interior=cell[a];
                if (m_core_adaptive) {
                    // The adjacent leaf can be finer. An un-crossed axis may
                    // have moved across many virtual cells since entry, so its
                    // entry index cannot identify that next leaf. Keep tied
                    // faces inside the old block until their x/y/z turn.
                    Int32 upper=lower+Int32(UInt32(1u)<<log_size)-1;
                    Float position=(point[a]-m_min[a])/tree_cell_size(a);
                    Int32 refreshed=Int32(dr::floor(dr::clip(position,
                        Float(lower-int(core_offset())),Float(upper-int(core_offset())))))+int(core_offset());
                    interior=dr::select(foreground_unit,refreshed,interior);
                }
                cell[a]=dr::select(crossings[a],
                    dr::select(foreground_unit,neighbor,cell[a]+step[a]),interior);
            }
            if (use_octree || m_core_adaptive || m_background_adaptive)
                next=dr::maximum(end,dr::select(foreground_unit,
                    world_block_boundary_times(cell,step,origin,inv_d,log_size),
                    octree_boundary_times(cell,step,origin,inv_d)));
            else {
                next.x()=dr::select(cross_x,next.x()+step_t.x(),next.x());
                next.y()=dr::select(cross_y,next.y()+step_t.y(),next.y());
                next.z()=dr::select(cross_z,next.z()+step_t.z(),next.z());
            }
            entry=end;
            ++iterations;
            return dr::make_tuple(cell,next,entry,iterations);
        }, "Voxel reconstruction leaf exit");
}

MI_VARIANT std::tuple<typename VoxelReconstruction<Float, Spectrum>::Color3f,
                     Float, typename VoxelReconstruction<Float, Spectrum>::UInt32>
VoxelReconstruction<Float, Spectrum>::trace(const Ray3f &ray, const Color3f &background,
    float transmittance_threshold, uint32_t max_hits, float entry_offset_fraction,
    bool use_octree) const {
    if (!std::isfinite(transmittance_threshold) || transmittance_threshold < 0.f ||
        transmittance_threshold >= 1.f || max_hits == 0 ||
        !std::isfinite(entry_offset_fraction) || entry_offset_fraction < 0.f)
        Throw("VoxelReconstruction: invalid traversal settings.");
    if (!m_active_count) {
        size_t width = std::max({dr::width(ray.o), dr::width(ray.d),
                                 dr::width(ray.maxt), dr::width(background)});
        Float zero = dr::zeros<Float>(width);
        return {background + zero, zero, dr::zeros<UInt32>(width)};
    }
    Float length = dr::norm(ray.d);
    Mask active = dr::all(dr::isfinite(ray.o)) & dr::all(dr::isfinite(ray.d)) &
                  (length > 0.f) & (ray.maxt > 0.f);
    Vector3f d = ray.d / dr::select(active, length, 1.f);
    Vector3f inv_d = dr::rcp(dr::select(d != 0.f, d, Vector3f(1.f)));
    Float entry = 0.f, exit = ray.maxt * length;
    Vector3f step_t;
    Vector3i step;
    for (size_t a = 0; a < 3; ++a) {
        Mask parallel = d[a] == 0.f;
        Float inv = dr::rcp(dr::select(parallel, 1.f, d[a]));
        Float near = (root_min(a) - ray.o[a]) * inv;
        Float far = (root_max(a) - ray.o[a]) * inv;
        entry = dr::maximum(entry, dr::select(parallel, -dr::Infinity<Float>, dr::minimum(near, far)));
        exit = dr::minimum(exit, dr::select(parallel, dr::Infinity<Float>, dr::maximum(near, far)));
        // Grid ownership is half-open: a parallel ray on the maximum face
        // has floor(cell) == resolution in Falcor and never enters the grid.
        active &= !parallel | ((ray.o[a] >= root_min(a)) & (ray.o[a] < root_max(a)));
        step[a] = dr::select(d[a] > 0.f, 1, dr::select(d[a] < 0.f, -1, 0));
        step_t[a] = dr::select(parallel, dr::Infinity<Float>, dr::abs(tree_cell_size(a) * inv));
    }
    if (entry_offset_fraction != 0.f) {
        if (m_background_shell_levels)
            entry += entry_offset_fraction*m_size[0];
        else {
            active &= dr::isfinite(ray.maxt);
            entry += entry_offset_fraction * ray.maxt * length;
        }
    }
    active &= entry <= exit;
    Point3f p = ray.o + d * dr::select(active, entry, 0.f);
    Vector3i cell;
    Vector3f next;
    for (size_t a = 0; a < 3; ++a) {
        Float grid_pos = (p[a] - m_min[a]) / tree_cell_size(a);
        grid_pos = dr::select(dr::isfinite(grid_pos), grid_pos, 0.f);
        cell[a] = Int32(dr::floor(dr::clip(grid_pos,-float(core_offset()),
            float(root_resolution()-1-core_offset()))))+int(core_offset());
        Float boundary = m_min[a] + (Float(cell[a]-int(core_offset())) + dr::select(step[a] > 0, 1.f, 0.f)) * tree_cell_size(a);
        next[a] = dr::select(step[a] == 0, dr::Infinity<Float>,
                            (boundary - ray.o[a]) / dr::select(step[a] == 0, 1.f, d[a]));
        next[a] = dr::maximum(next[a], entry);
    }
    if (use_octree)
        next = dr::maximum(entry, octree_boundary_times(cell, step, ray.o, inv_d));
    const bool world_shell=m_scene_mapping=="world_shell";
    const bool mixed_tree=world_shell || m_core_adaptive;
    const Vector3f world_d=d;
    RaySegment first{ray.o,d,exit};
    struct LoopState {
        Mask active;
        Vector3i cell;
        Vector3f next;
        Float entry, transmittance;
        Color3f color;
        UInt32 hits, iterations, tree_node, tree_level;
        SpatialAnchor tree_anchor;
        RaySegment segment;
        DRJIT_STRUCT(LoopState, active, cell, next, entry, transmittance, color, hits, iterations,
                     tree_node, tree_level, tree_anchor, segment)
    } ls{active, cell, next, entry, Float(1.f), Color3f(0.f), UInt32(0), UInt32(0),
         UInt32(m_tree.root), UInt32(m_tree.depth), SpatialAnchor(0), first};

    dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
        [](const LoopState &s) { return s.active; },
        [&](LoopState &s) {
            const Vector3f d=s.segment.direction;
            const Vector3f inv_d=dr::rcp(dr::select(d!=0.f,d,Vector3f(1.f)));
            const Vector3i step=dr::select(d>0.f,Vector3i(1),dr::select(d<0.f,Vector3i(-1),Vector3i(0)));
            const Vector3f step_t=dr::select(d==0.f,Vector3f(dr::Infinity<Float>),
                dr::abs(Vector3f(tree_cell_size(0),tree_cell_size(1),tree_cell_size(2))*inv_d));
            const Float exit=s.segment.exit;
            auto [id_signed, empty_log] = m_tree.template lookup<UInt32, Int32>(
                s.cell.x(), s.cell.y(), s.cell.z(), s.tree_node, s.tree_level, s.tree_anchor, s.active);
            Mask occupied = s.active & (id_signed >= 0);
            UInt32 id = UInt32(dr::maximum(id_signed, 0));
            if (use_octree && !mixed_tree) {
                Mask empty = s.active & !occupied;
                s = dr::if_stmt(dr::make_tuple(s, empty, empty_log), empty,
                    [this, step, step_t, origin = s.segment.origin, inv_d, exit](
                        LoopState s, const Mask &empty, const UInt32 &empty_log) {
                        skip_empty_block(s.cell, s.next, s.entry, s.iterations,
                                         step, step_t, origin, inv_d, exit, empty, empty_log);
                        return s;
                    }, [](LoopState s, const Mask &, const UInt32 &) { return s; },
                    "Voxel reconstruction octree trace skip");
            }
            if (mixed_tree)
                s.next=dr::maximum(s.entry,world_block_boundary_times(
                    s.cell,step,s.segment.origin,inv_d,empty_log));
            Float end = dr::minimum(dr::minimum(s.next.x(), dr::minimum(s.next.y(), s.next.z())), exit);
            end = dr::maximum(end, s.entry);
            const bool voxel_background = world_shell && !m_background_use_ellipsoid;
            Mask ellipsoid = voxel_background ? core_ids(id, occupied) : occupied;
            auto intersects = [this](const UInt32 &id, const Vector3i &cell,
                                    const Point3f &origin, const Vector3f &d,
                                    const Float &entry, const Float &end, const Mask &occupied) -> Mask {
                Vector3f local, scale, segment = d * (end - entry);
                UInt32 log_size = leaf_log_size(id, occupied);
                Float block(UInt32(1u) << log_size);
                for (size_t a = 0; a < 3; ++a) {
                    Float c = dr::gather<Float>(m_centers, id * 3 + uint32_t(a), occupied);
                    Float log_scale = dr::gather<Float>(m_log_scales, id * 3 + uint32_t(a), occupied);
                    // Subtract the cell origin first to retain precision for small ellipsoids.
                    Int32 lower = Int32((UInt32(cell[a]) >> log_size) << log_size);
                    local[a] = origin[a] + entry*d[a] -
                               (m_min[a] + (Float(lower-int(core_offset())) + c*block) * tree_cell_size(a));
                    scale[a] = dr::exp(dr::clip(log_scale, -30.f, 30.f));
                }
                Float qw = dr::gather<Float>(m_rotations, id*4, occupied);
                Vector3f qv;
                for (size_t a = 0; a < 3; ++a)
                    qv[a] = dr::gather<Float>(m_rotations, id*4 + uint32_t(a+1), occupied);
                Float qnorm = dr::sqrt(qw*qw + dr::squared_norm(qv));
                Mask valid_q = dr::isfinite(qnorm) & (qnorm > 1e-10f);
                qw = dr::select(valid_q, qw / qnorm, 1.f);
                qv = dr::select(valid_q, -qv / qnorm, Vector3f(0.f)); // inverse quaternion
                auto to_unit = [&](const Vector3f &v) {
                    return (v + 2.f * dr::cross(qv, dr::cross(qv, v) + qw*v)) / scale;
                };
                Vector3f u = to_unit(local), v = to_unit(segment);
                Float a = dr::squared_norm(v), b = 2.f * dr::dot(u, v), c = dr::squared_norm(u) - 1.f;
                Float discriminant = b*b - 4.f*a*c;
                Float root = dr::sqrt(dr::maximum(discriminant, 0.f));
                Float inv_a2 = .5f / dr::select(a > 1e-20f, a, 1.f);
                Float t0 = dr::maximum(0.f, (-b-root)*inv_a2);
                Float t1 = dr::minimum(1.f, (-b+root)*inv_a2);
                return occupied & dr::select(a <= 1e-20f, c <= 0.f,
                                             (discriminant >= 0.f) & (t0 <= t1));
            };
            Mask hit = voxel_background ? dr::if_stmt(
                dr::make_tuple(id, s.cell, s.segment.origin, d, s.entry, end, ellipsoid),
                ellipsoid, intersects,
                [](const UInt32 &, const Vector3i &, const Point3f &, const Vector3f &,
                   const Float &, const Float &, const Mask &) { return Mask(false); },
                "Voxel reconstruction core trace intersection") :
                intersects(id, s.cell, s.segment.origin, d, s.entry, end, ellipsoid);
            // DDA may visit tied faces with zero segment length. These are not
            // occupied-volume hits, while the legacy ellipsoid rule is retained.
            if (voxel_background)
                hit |= occupied & !ellipsoid & (end > s.entry);
            Color3f emission(0.f);
            Float logit = 0.f;
            for (uint32_t k = 0; k < std::max(m_active_sh_count, m_active_opacity_sh_count); ++k) {
                Float sh = reconstruction_sh(k, Vector3f(-world_d));
                if (k < m_active_sh_count)
                    for (uint32_t channel = 0; channel < 3; ++channel)
                        emission[channel] += sh * dr::gather<Float>(m_radiance,
                            (id*m_sh_count+k)*3+channel, hit);
                if (k < m_active_opacity_sh_count)
                    logit += sh * dr::gather<Float>(m_opacity, id*m_opacity_sh_count+k, hit);
            }
            Float alpha = dr::select(hit, dr::rcp(1.f + dr::exp(-logit)), 0.f);
            s.color += s.transmittance * alpha * dr::maximum(emission, 0.f);
            s.transmittance *= 1.f - alpha;
            s.hits += dr::select(hit, 1u, 0u);
            advance_ray_cell(s.cell,s.next,s.entry,s.iterations,step,step_t,
                s.segment.origin,d,inv_d,exit,end,empty_log,use_octree);
            Mask in_segment=dr::all(s.cell>=0)&dr::all(s.cell<int(root_resolution()))&(s.entry<exit);
            Mask budget=s.iterations<=3*root_resolution()+3;
            s.active &= in_segment & budget;
            s.active &= (s.transmittance>transmittance_threshold)&(s.hits<max_hits);
        }, "Voxel reconstruction DDA");
    return {ls.color + ls.transmittance*background, 1.f-ls.transmittance, ls.hits};
}

MI_VARIANT std::pair<Float, typename VoxelReconstruction<Float, Spectrum>::Color3f>
VoxelReconstruction<Float, Spectrum>::dssim_image_loss_and_grad(
    const Color3f &rgb, const Color3f &reference,
    uint32_t image_width, uint32_t image_height) {
    if constexpr (!dr::is_jit_v<Float>) {
        Throw("VoxelReconstruction.dssim_image_loss_and_grad(): complete images require a JIT variant.");
    } else {
        const uint64_t pixels = uint64_t(image_width) * image_height;
        if (!image_width || !image_height || pixels > INT32_MAX)
            Throw("VoxelReconstruction.dssim_image_loss_and_grad(): invalid image dimensions.");
        for (uint32_t c = 0; c < 3; ++c)
            if ((dr::width(rgb[c]) != 1 && dr::width(rgb[c]) != pixels) ||
                (dr::width(reference[c]) != 1 && dr::width(reference[c]) != pixels))
                Throw("VoxelReconstruction.dssim_image_loss_and_grad(): RGB buffers must cover one complete image.");
        using Value = dr::detached_t<Float>;
        using Index = dr::uint32_array_t<Value>;
        using SignedIndex = dr::int32_array_t<Value>;
        using ImageMask = dr::mask_t<Value>;
        const uint32_t count = uint32_t(pixels);
        Index lane = dr::arange<Index>(count);
        SignedIndex x = SignedIndex(lane % image_width), y = SignedIndex(lane / image_width);
        std::array<float, 11> kernel;
        double kernel_sum = 0.;
        for (int tap = -5; tap <= 5; ++tap)
            kernel_sum += std::exp(-double(tap * tap) / (2. * 1.5 * 1.5));
        for (int tap = -5; tap <= 5; ++tap)
            kernel[tap + 5] = float(std::exp(-double(tap * tap) / (2. * 1.5 * 1.5)) / kernel_sum);
        auto materialize = [](const auto &planes) {
            for (const auto &plane : planes)
                dr::schedule(plane);
            dr::eval();
        };
        // Detached image arithmetic is explicit: only dcolor is propagated into
        // the renderer's custom reverse, never an ordinary AD ray-tracing tape.
        std::array<Value, 3> image, target;
        for (uint32_t c = 0; c < 3; ++c) {
            image[c] = dr::detach<false>(rgb[c]) + dr::zeros<Value>(count);
            target[c] = dr::detach<false>(reference[c]) + dr::zeros<Value>(count);
        }
        materialize(image);
        materialize(target);
        auto blur_axis = [&](const auto &input, bool horizontal) {
            auto output = input;
            for (auto &plane : output)
                plane = dr::zeros<Value>(count);
            for (int tap = -5; tap <= 5; ++tap) {
                SignedIndex xx = x + (horizontal ? tap : 0);
                SignedIndex yy = y + (horizontal ? 0 : tap);
                ImageMask valid = (xx >= 0) & (xx < int(image_width)) &
                                  (yy >= 0) & (yy < int(image_height));
                Index index = Index(yy * int(image_width) + xx);
                for (size_t plane = 0; plane < input.size(); ++plane)
                    output[plane] += kernel[tap + 5] *
                        dr::gather<Value>(input[plane], index, valid);
            }
            materialize(output);
            return output;
        };
        std::array<Value, 15> moments;
        // Fuse the first moment pass: each neighboring x/y value is gathered
        // once, then reused for x, y, x*x, y*y and x*y.
        for (auto &plane : moments)
            plane = dr::zeros<Value>(count);
        for (int tap = -5; tap <= 5; ++tap) {
            SignedIndex xx = x + tap;
            ImageMask valid = (xx >= 0) & (xx < int(image_width));
            Index index = Index(y * int(image_width) + xx);
            for (uint32_t c = 0; c < 3; ++c) {
                Value a = dr::gather<Value>(image[c], index, valid);
                Value b = dr::gather<Value>(target[c], index, valid);
                const float k = kernel[tap + 5];
                moments[c * 5] += k * a;
                moments[c * 5 + 1] += k * b;
                moments[c * 5 + 2] += k * a * a;
                moments[c * 5 + 3] += k * b * b;
                moments[c * 5 + 4] += k * a * b;
            }
        }
        materialize(moments);
        moments = blur_axis(moments, false);
        std::array<Value, 9> adjoint;
        Value loss = dr::zeros<Value>(count);
        for (uint32_t c = 0; c < 3; ++c) {
            const Value &mu_x = moments[c * 5], &mu_y = moments[c * 5 + 1];
            Value raw_var_x = moments[c * 5 + 2] - dr::square(mu_x);
            Value raw_var_y = moments[c * 5 + 3] - dr::square(mu_y);
            Value var_x = dr::maximum(raw_var_x, 0.f), var_y = dr::maximum(raw_var_y, 0.f);
            Value covariance = moments[c * 5 + 4] - mu_x * mu_y;
            Value a = 2.f * mu_x * mu_y + .01f * .01f;
            Value b = 2.f * covariance + .03f * .03f;
            Value denominator_mu = dr::square(mu_x) + dr::square(mu_y) + .01f * .01f;
            Value denominator_var = var_x + var_y + .03f * .03f;
            Value inverse = dr::rcp(denominator_mu * denominator_var);
            Value score = a * b * inverse;
            ImageMask finite = dr::isfinite(score) & dr::isfinite(raw_var_x) & dr::isfinite(raw_var_y);
            ImageMask live = finite & (score > -1.f) & (score < 1.f);
            Value stabilized = dr::select(finite, dr::clip(score, -1.f, 1.f),
                                          std::numeric_limits<float>::quiet_NaN());
            loss += (1.f - stabilized) / 3.f;
            Value d_mu = dr::select(live, 2.f * mu_y * b * inverse -
                                   2.f * mu_x * score / denominator_mu, 0.f);
            Value d_var = dr::select(live & (raw_var_x > 0.f), -score / denominator_var, 0.f);
            Value d_cov = dr::select(live, 2.f * a * inverse, 0.f);
            adjoint[c * 3] = -(d_mu - 2.f * mu_x * d_var - mu_y * d_cov) / 3.f;
            adjoint[c * 3 + 1] = -d_var / 3.f;
            adjoint[c * 3 + 2] = -d_cov / 3.f;
        }
        dr::schedule(loss);
        materialize(adjoint);
        for (auto &plane : moments)
            plane = Value();
        // Symmetric zero padding makes the convolution self-adjoint. Applying
        // it to the three coefficient planes sums every overlapping window's
        // contribution without pixel scatter atomics or another ray traversal.
        adjoint = blur_axis(adjoint, true);
        adjoint = blur_axis(adjoint, false);
        Color3f dcolor;
        for (uint32_t c = 0; c < 3; ++c)
            dcolor[c] = Float(adjoint[c * 3] + 2.f * image[c] * adjoint[c * 3 + 1] +
                              target[c] * adjoint[c * 3 + 2]);
        dr::eval(dcolor);
        return {Float(loss), dcolor};
    }
}

// Port of Falcor NoLightTransport GradientPass/PathRecord semantics, source
// 398f634ec788bbb2e86a6f4d0ab78d41158d9905. The geometry terms below are
// deliberately surrogate derivatives of a hard forward model, not ordinary AD.
MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::TrainingResult
VoxelReconstruction<Float, Spectrum>::loss_and_grad(
    const Ray3f &ray, const Color3f &reference_rgb, const Float &reference_alpha,
    const Color3f &background, const Float &inverse_pdf,
    const std::map<std::string, float> &settings) const {
    std::map<std::string, float> cfg = {
        {"alpha_loss_weight", .5f}, {"foreground_alpha_min", .95f},
        {"background_alpha_max", 1e-4f}, {"geometry_tau_voxel_fraction", .12f},
        {"geometry_grad_clamp", 5.f}, {"alpha_geometry_weight", .1f},
        {"background_carve_weight", .01f}, {"geometry_two_sided_boundary", 0.f},
        {"geometry_boundary_band_tau", 1.f}, {"geometry_transition_only", 1.f},
        {"geometry_hit_expansion_band_tau", 0.f}, {"collect_deletion_evidence", 1.f},
        {"min_evidence_transmittance", .05f}, {"min_removal_loss_delta", 1e-4f},
        {"transmittance_threshold", .01f}, {"max_hits", 16.f},
        {"max_candidates", 16.f}, {"entry_offset_fraction", 0.f},
        {"profile_stages", 0.f}, {"profile_traversal", 0.f},
        {"use_octree", 1.f}, {"rgb_only", 0.f}, {"rgb_loss_srgb", 0.f},
        {"collect_rgb_deletion_evidence", 0.f},
        {"collect_adaptive_priority", 0.f},
        {"collect_growth_evidence", 0.f}, {"max_growth_hits", 8.f},
        {"distortion_loss_weight", 0.f}, {"compact_gradient_capacity", 0.f},
        {"dssim_loss_weight", 0.f}, {"image_width", 0.f}, {"image_height", 0.f},
        {"compact_path_records", 1.f}};
    for (const auto &[key, value] : settings) {
        if (cfg.find(key) == cfg.end() || !std::isfinite(value) || value < 0.f)
            Throw("VoxelReconstruction.loss_and_grad(): unknown/invalid setting %s.", key.c_str());
        cfg[key] = value;
    }
    const bool rgb_only = cfg["rgb_only"] != 0.f;
    const bool rgb_loss_srgb = cfg["rgb_loss_srgb"] != 0.f;
    if (cfg["rgb_loss_srgb"] != 0.f && cfg["rgb_loss_srgb"] != 1.f)
        Throw("VoxelReconstruction.loss_and_grad(): rgb_loss_srgb must be 0 or 1.");
    if (rgb_loss_srgb && !rgb_only)
        Throw("VoxelReconstruction.loss_and_grad(): sRGB image loss requires RGB-only mode.");
    // This is a storage/recomputation policy, not a different estimator. Keep
    // the legacy alpha path unchanged; its suffix alpha proxy is still live.
    const bool compact_path_records = rgb_only && cfg["compact_path_records"] != 0.f;
    const bool use_octree = cfg["use_octree"] != 0.f;
    const bool profile_traversal = cfg["profile_traversal"] != 0.f;
    const float distortion_weight = cfg["distortion_loss_weight"];
    const bool distortion_enabled = distortion_weight > 0.f;
    const float dssim_weight = cfg["dssim_loss_weight"];
    const bool dssim_enabled = dssim_weight > 0.f;
    uint32_t image_width = 0, image_height = 0;
    if (dssim_enabled) {
        if (!dr::is_jit_v<Float> || !rgb_only)
            Throw("VoxelReconstruction.loss_and_grad(): D-SSIM requires a JIT RGB-only full-image batch.");
        for (const char *name : {"image_width", "image_height"})
            if (cfg[name] < 1.f || double(cfg[name]) > double(INT32_MAX) ||
                std::floor(cfg[name]) != cfg[name])
                Throw("VoxelReconstruction.loss_and_grad(): D-SSIM requires positive integer image_width/image_height.");
        image_width = uint32_t(cfg["image_width"]);
        image_height = uint32_t(cfg["image_height"]);
    }
    const float requested_capacity = cfg["compact_gradient_capacity"];
    if (double(requested_capacity) > double(UINT32_MAX) ||
        std::floor(requested_capacity) != requested_capacity)
        Throw("VoxelReconstruction.loss_and_grad(): compact_gradient_capacity must be an integer in [0, UINT32_MAX].");
    const bool compact_gradients = requested_capacity > 0.f;
    const uint32_t gradient_rows = compact_gradients ?
        std::min(uint32_t(requested_capacity), m_active_count) : m_active_count;
    const bool collect_rgb_deletion = cfg["collect_rgb_deletion_evidence"] != 0.f;
    const bool collect_adaptive_priority = cfg["collect_adaptive_priority"] != 0.f;
    if (collect_rgb_deletion && !rgb_only)
        Throw("VoxelReconstruction.loss_and_grad(): RGB deletion evidence requires RGB-only mode.");
    const bool growth_requested = cfg["collect_growth_evidence"] != 0.f;
    const bool collect_growth = growth_requested && m_growth_count != 0;
    const uint32_t max_growth_hits = uint32_t(cfg["max_growth_hits"]);
    if ((growth_requested && !rgb_only) || !max_growth_hits || max_growth_hits > 32 ||
        cfg["max_growth_hits"] != float(max_growth_hits))
        Throw("VoxelReconstruction.loss_and_grad(): growth requires RGB-only mode and max_growth_hits in [1, 32].");
    const uint32_t max_hits = uint32_t(cfg["max_hits"]), max_candidates = uint32_t(cfg["max_candidates"]);
    if (!max_hits || max_hits > 128 || cfg["max_hits"] != float(max_hits) ||
        max_candidates > 128 || cfg["max_candidates"] != float(max_candidates) ||
        cfg["transmittance_threshold"] >= 1.f ||
        (!rgb_only && (cfg["foreground_alpha_min"] > 1.f ||
         cfg["background_alpha_max"] > cfg["foreground_alpha_min"] || cfg["min_evidence_transmittance"] > 1.f)))
        Throw("VoxelReconstruction.loss_and_grad(): invalid path capacity or thresholds.");
    size_t width = std::max({dr::width(ray.o), dr::width(ray.d), dr::width(ray.maxt),
                             dr::width(reference_rgb), dr::width(inverse_pdf), dr::width(background)});
    // RGB-only callers pass an already composited target. Even an empty or
    // differently sized compatibility alpha argument must not affect the batch.
    if (!rgb_only)
        width = std::max(width, dr::width(reference_alpha));
    if (dssim_enabled && (uint64_t(image_width) * image_height != width || width > INT32_MAX))
        Throw("VoxelReconstruction.loss_and_grad(): D-SSIM requires exactly image_width*image_height rays in row-major order.");
    const uint32_t hit_before_field = compact_path_records ? 4u : 8u;
    const uint32_t hit_depth_field = hit_before_field + 1u;
    const uint32_t hit_fields = hit_depth_field + (distortion_enabled ? 2u : 0u);
    if (width == 0 || width * std::max(max_hits, std::max(max_candidates,
        collect_growth ? max_growth_hits : 1u)) * hit_fields > UINT32_MAX)
        Throw("VoxelReconstruction.loss_and_grad(): ray batch is empty or too large.");
    ReconstructionStageProfiler<dr::is_jit_v<Float>> profiler(cfg["profile_stages"] != 0.f);
    // std::map is not traversed by dr::schedule(); schedule its buffers explicitly.
    auto evaluate_result = [](const TrainingResult &result) {
        if constexpr (dr::is_jit_v<Float>) {
            dr::schedule(result.rgb, result.raw_rgb, result.alpha, result.raw_alpha,
                         result.loss, result.rgb_loss, result.distortion_loss, result.hit_count);
            if (result.dssim_enabled)
                dr::schedule(result.dssim_loss);
            if (result.compact_gradients)
                dr::schedule(result.voxel_ids, result.voxel_count, result.compact_gradient_overflow);
            for (const auto &[key, value] : result.gradients)
                dr::schedule(value);
            for (const auto &[key, value] : result.counters)
                dr::schedule(value);
            for (const auto &[key, value] : result.growth_sums)
                dr::schedule(value);
            for (const auto &[key, value] : result.growth_counts)
                dr::schedule(value);
            for (const auto &[key, value] : result.deletion_sums)
                dr::schedule(value);
            for (const auto &[key, value] : result.deletion_counts)
                dr::schedule(value);
            dr::eval();
        }
    };
    if (!m_active_count && !collect_growth) {
        TrainingResult result;
        result.compact_gradients = compact_gradients;
        if (compact_gradients)
            result.voxel_ids = dr::empty<UIntBuffer>(0);
        result.raw_alpha = dr::zeros<Float>(width);
        result.raw_rgb = background + result.raw_alpha;
        Float weight = dr::select(inverse_pdf > 0.f, inverse_pdf, 1.f);
        result.rgb = Color3f(1.f) + (result.raw_rgb - 1.f)*weight;
        result.alpha = result.raw_alpha;
        Color3f target = rgb_only ? reference_rgb :
            reference_rgb*reference_alpha + Color3f(1.f)*(1.f-reference_alpha);
        Color3f loss_rgb = rgb_loss_srgb ? reconstruction_linear_to_srgb(result.rgb) : result.rgb;
        Color3f loss_target = rgb_loss_srgb ? reconstruction_linear_to_srgb(target) : target;
        result.rgb_loss = dr::squared_norm(loss_rgb-loss_target);
        result.distortion_loss = dr::zeros<Float>(width);
        result.loss = result.rgb_loss;
        if (!rgb_only)
            result.loss += cfg["alpha_loss_weight"]*dr::square(reference_alpha);
        result.hit_count = dr::zeros<UInt32>(width);
        for (const auto *key : {"centers", "log_scales", "rotation_tangent", "radiance_sh", "opacity_sh"})
            result.gradients[key] = dr::zeros<FloatBuffer>(0);
        for (const auto *key : {"appearance_valid", "geometry_valid", "background_carve_valid", "foreground_evidence", "background_evidence"})
            result.counters[key] = dr::zeros<UIntBuffer>(0);
        if (profile_traversal)
            for (const auto *key : {"traversal_steps", "traversal_skipped_cells",
                                   "traversal_leaf_visits", "traversal_tree_queries"})
                result.counters[key] = dr::zeros<UIntBuffer>(width);
        if (growth_requested) {
            result.growth_sums["growth_gain"] = dr::zeros<FloatBuffer>(0);
            result.growth_counts["growth_observed"] = dr::zeros<UIntBuffer>(0);
            result.growth_counts["growth_unknown"] = dr::zeros<UIntBuffer>(0);
        }
        if (collect_rgb_deletion) {
            result.deletion_sums["deletion_gain"] = dr::zeros<FloatBuffer>(0);
            result.deletion_counts["deletion_observed"] = dr::zeros<UIntBuffer>(0);
            result.deletion_counts["deletion_unknown"] = dr::zeros<UIntBuffer>(0);
        }
        if (collect_adaptive_priority)
            result.deletion_sums["adaptive_priority"] = dr::zeros<FloatBuffer>(0);
        if (dssim_enabled) {
            if (profiler.enabled()) {
                evaluate_result(result);
                profiler.finish("forward");
                profiler.begin();
            }
            result.dssim_enabled = true;
            auto image_loss = dssim_image_loss_and_grad(loss_rgb, loss_target, image_width, image_height);
            // A device guard preserves frozen replay and avoids synchronizing
            // every view just to validate the uniform sampling contract.
            result.dssim_loss = dr::select(dr::all(inverse_pdf == 1.f), image_loss.first,
                                           std::numeric_limits<float>::quiet_NaN());
            result.loss += dssim_weight * result.dssim_loss;
        }
        if (profiler.enabled()) {
            evaluate_result(result);
            profiler.finish(dssim_enabled ? "image_loss" : "forward");
            profiler.zero_stage("hit_backward");
            profiler.zero_stage("candidate_backward");
            result.stage_profile = std::move(profiler.profile);
        }
        return result;
    }
    UInt32 lane(0);
    if constexpr (dr::is_jit_v<Float>) lane = dr::arange<UInt32>(width);

    // Five compact hit floats: relative-world xStar[3], gmin, T_before.
    // Appearance is recomputed by the same function from unchanged parameters
    // during reverse traversal. The diagnostic full-record path additionally
    // retains RGB[3] and opacity, matching the original nine-field layout.
    // Each field contains hit-major rows of rays: field * hit_stride +
    // hit_index * ray_count + lane. Reverse traversal therefore reads adjacent
    // lanes contiguously. Distortion adds normalized closest-point depth and
    // an unclamped-depth flag only when explicitly enabled.
    const uint32_t ray_count = uint32_t(width), hit_stride = ray_count * max_hits;
    // Candidate slots are lane-local loop variables. A read after a scatter to
    // global memory cannot be evaluated inside a symbolic Dr.Jit loop.
    struct Candidate {
        Vector3f point;
        Float gmin, before, score;
        UInt32 id, front;
        DRJIT_STRUCT(Candidate, point, gmin, before, score, id, front)
    };
    using Candidates = std::vector<Candidate>;
    struct HitBuffers {
        FloatBuffer hit;
        UIntBuffer hit_id;
    };
    struct GrowthBuffers {
        FloatBuffer values;
        UIntBuffer ids, unknown;
    };
    struct Records {
        FloatBuffer hit;
        UIntBuffer hit_id;
        Candidates candidates;
        FloatBuffer growth_values;
        UIntBuffer growth_ids, growth_unknown;
        DRJIT_STRUCT(Records, hit, hit_id, candidates, growth_values, growth_ids, growth_unknown)
    } records{dr::zeros<FloatBuffer>(width*max_hits*hit_fields),
              dr::zeros<UIntBuffer>(width*max_hits),
              std::vector<Candidate>(max_candidates,
                  Candidate{Vector3f(0.f), Float(0.f), Float(0.f), Float(0.f), UInt32(0), UInt32(0)}),
              dr::zeros<FloatBuffer>(collect_growth ? width*max_growth_hits*6 : 1),
              dr::zeros<UIntBuffer>(collect_growth ? width*max_growth_hits : 1),
              dr::zeros<UIntBuffer>(collect_growth ? m_growth_count : 1)};
    const uint32_t growth_stride = collect_growth ? ray_count*max_growth_hits : 0;
    auto clean = [](const auto &value) { return dr::select(dr::isfinite(value), value, 0.f); };
    auto rotate = [](const Vector3f &v, const Float &w, const Vector3f &q) {
        return v + 2.f*dr::cross(q, dr::cross(q, v) + w*v);
    };
    auto growth_intersects = [this, rotate](const UInt32 &id,
        const Float &entry, const Float &end, const Point3f &origin,
        const Vector3f &direction, const Mask &active) {
        // Identical clipped-cell intersection for both the DDA record check and
        // the later proposal scoring. Siblings share a segment, not a shape.
        Vector3f relative, scale, segment = direction*(end-entry);
        auto cell=decode_cells(dr::gather<CellIndex>(m_growth_cells,id,active));
        Float block(UInt32(1u)<<cell[3]);
        for (uint32_t a=0; a<3; ++a) {
            Float center = dr::gather<Float>(m_growth_centers,id*3+a,active);
            relative[a] = origin[a]+entry*direction[a] -
                (m_min[a]+(Float(Int32(cell[a])-int(core_offset()))+center*block)*tree_cell_size(a));
            scale[a] = dr::exp(dr::clip(dr::gather<Float>(m_growth_log_scales,
                id*3+a,active),-30.f,30.f));
        }
        Float qw = dr::gather<Float>(m_growth_rotations,id*4,active);
        Vector3f qv;
        for (uint32_t a=0; a<3; ++a)
            qv[a] = dr::gather<Float>(m_growth_rotations,id*4+a+1,active);
        Float norm = dr::sqrt(qw*qw+dr::squared_norm(qv));
        Mask valid_q = dr::isfinite(norm)&(norm>1e-10f);
        qw = dr::select(valid_q,qw/dr::select(valid_q,norm,1.f),1.f);
        qv = dr::select(valid_q,qv/dr::select(valid_q,norm,1.f),Vector3f(0.f));
        Vector3f u = rotate(relative,qw,-qv)/scale, v = rotate(segment,qw,-qv)/scale;
        Float a = dr::squared_norm(v), b = 2.f*dr::dot(u,v), c = dr::squared_norm(u)-1.f;
        Float discriminant = b*b-4.f*a*c;
        Float root = dr::sqrt(dr::maximum(discriminant,0.f));
        Float inv = .5f/dr::select(a>1e-20f,a,1.f);
        Float t0 = dr::maximum(0.f,(-b-root)*inv), t1 = dr::minimum(1.f,(-b+root)*inv);
        return active & dr::select(a<=1e-20f,c<=0.f,
            (discriminant>=0.f)&(t0<=t1));
    };
    auto quaternion = [this](UInt32 id, Mask mask, Float &w, Vector3f &q) {
        w = dr::gather<Float>(m_rotations, id*4, mask);
        for (uint32_t a=0; a<3; ++a) q[a] = dr::gather<Float>(m_rotations, id*4+a+1, mask);
        Float norm = dr::sqrt(w*w+dr::squared_norm(q));
        Mask valid = dr::isfinite(norm) & (norm > 1e-10f);
        w = dr::select(valid, w/dr::select(valid,norm,1.f), 1.f);
        q = dr::select(valid, q/dr::select(valid,norm,1.f), Vector3f(0.f));
    };
    auto appearance = [this, clean](UInt32 id, const Vector3f &view, Mask mask, Color3f &rgb, Float &alpha) {
        rgb = Color3f(0.f);
        Float logit(0.f);
        for (uint32_t k=0; k<std::max(m_active_sh_count,m_active_opacity_sh_count); ++k) {
            Float sh = reconstruction_sh(k,view);
            if (k<m_active_sh_count)
                for (uint32_t a=0;a<3;++a)
                    rgb[a] += sh*dr::gather<Float>(m_radiance,(id*m_sh_count+k)*3+a,mask);
            if (k<m_active_opacity_sh_count)
                logit += sh*dr::gather<Float>(m_opacity,id*m_opacity_sh_count+k,mask);
        }
        rgb = dr::maximum(clean(rgb),0.f);
        alpha = dr::rcp(1.f+dr::exp(-logit));
    };

    Float direction_length = dr::norm(ray.d);
    Mask active = dr::all(dr::isfinite(ray.o)) & dr::all(dr::isfinite(ray.d)) &
                  (direction_length > 0.f) & (ray.maxt > 0.f);
    Vector3f d = ray.d / dr::select(active,direction_length,1.f);
    Vector3f inv_d = dr::rcp(dr::select(d != 0.f, d, Vector3f(1.f)));
    Float entry(0.f), exit = ray.maxt*direction_length;
    Vector3f step_t;
    Vector3i step;
    for (uint32_t a=0;a<3;++a) {
        Mask parallel=d[a]==0.f;
        Float inv=dr::rcp(dr::select(parallel,1.f,d[a]));
        Float near=(root_min(a)-ray.o[a])*inv, far=(root_max(a)-ray.o[a])*inv;
        entry=dr::maximum(entry,dr::select(parallel,-dr::Infinity<Float>,dr::minimum(near,far)));
        exit=dr::minimum(exit,dr::select(parallel,dr::Infinity<Float>,dr::maximum(near,far)));
        active &= !parallel | ((ray.o[a]>=root_min(a)) & (ray.o[a]<root_max(a)));
        step[a]=dr::select(d[a]>0.f,1,dr::select(d[a]<0.f,-1,0));
        step_t[a]=dr::select(parallel,dr::Infinity<Float>,dr::abs(tree_cell_size(a)*inv));
    }
    if (cfg["entry_offset_fraction"]!=0.f) {
        if (m_background_shell_levels)
            entry += cfg["entry_offset_fraction"]*m_size[0];
        else {
            active &= dr::isfinite(ray.maxt);
            entry += cfg["entry_offset_fraction"]*ray.maxt*direction_length;
        }
    }
    active &= entry<=exit;
    Point3f p=ray.o+d*dr::select(active,entry,0.f);
    Vector3i cell;
    Vector3f next;
    for (uint32_t a=0;a<3;++a) {
        Float position=(p[a]-m_min[a])/tree_cell_size(a);
        position=dr::select(dr::isfinite(position),position,0.f);
        cell[a]=Int32(dr::floor(dr::clip(position,-float(core_offset()),
            float(root_resolution()-1-core_offset()))))+int(core_offset());
        Float boundary=m_min[a]+(Float(cell[a]-int(core_offset()))+dr::select(step[a]>0,1.f,0.f))*tree_cell_size(a);
        next[a]=dr::maximum(entry,dr::select(step[a]==0,dr::Infinity<Float>,
                          (boundary-ray.o[a])/dr::select(step[a]==0,1.f,d[a])));
    }
    if (use_octree)
        next = dr::maximum(entry, octree_boundary_times(cell, step, ray.o, inv_d));
    const bool world_shell=m_scene_mapping=="world_shell";
    const bool mixed_tree=world_shell || m_core_adaptive;
    const Vector3f world_d=d;
    RaySegment first{ray.o,d,exit};
    struct State {
        Mask active;
        Vector3i cell;
        Vector3f next;
        Float entry, transmittance;
        Color3f color;
        UInt32 hits,candidates,iterations,growth_hits;
        UInt32 traversal_steps, traversal_skipped, traversal_leaves, traversal_queries;
        UInt32 tree_node, tree_level, growth_node, growth_level;
        SpatialAnchor tree_anchor, growth_anchor;
        Float distortion_mass, distortion_moment, distortion;
        RaySegment segment;
        DRJIT_STRUCT(State,active,cell,next,entry,transmittance,color,hits,candidates,
                     iterations,growth_hits,traversal_steps,traversal_skipped,
                     traversal_leaves,traversal_queries,tree_node,tree_level,growth_node,growth_level,
                     tree_anchor,growth_anchor,distortion_mass,distortion_moment,distortion,segment)
    } state{active,cell,next,entry,Float(1.f),Color3f(0.f),UInt32(0),UInt32(0),
            UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),
            UInt32(m_tree.root),UInt32(m_tree.depth),UInt32(m_growth_tree.root),UInt32(m_growth_tree.depth),
            SpatialAnchor(0),SpatialAnchor(0),
            Float(0.f),Float(0.f),Float(0.f),first};
    const float diagonal=std::sqrt(dr::square(m_max[0]-m_min[0])+dr::square(m_max[1]-m_min[1])+dr::square(m_max[2]-m_min[2]));
    dr::tie(state,records)=dr::while_loop(dr::make_tuple(state,records),
        [](const State &s,const Records &) {return s.active;},
        [&](State &s,Records &r) {
            const Vector3f d=s.segment.direction;
            const Vector3f inv_d=dr::rcp(dr::select(d!=0.f,d,Vector3f(1.f)));
            const Vector3i step=dr::select(d>0.f,Vector3i(1),dr::select(d<0.f,Vector3i(-1),Vector3i(0)));
            const Vector3f step_t=dr::select(d==0.f,Vector3f(dr::Infinity<Float>),
                dr::abs(Vector3f(tree_cell_size(0),tree_cell_size(1),tree_cell_size(2))*inv_d));
            const Float exit=s.segment.exit;
            auto [signed_id, empty_log] = m_tree.template lookup<UInt32, Int32>(
                s.cell.x(), s.cell.y(), s.cell.z(), s.tree_node, s.tree_level, s.tree_anchor, s.active);
            Mask occupied=s.active & (signed_id>=0);
            UInt32 id=UInt32(dr::maximum(signed_id,0));
            if (collect_growth && m_growth_count) {
                auto [proposal_id, proposal_log] = m_growth_tree.template lookup<UInt32, Int32>(
                    s.cell.x(), s.cell.y(), s.cell.z(), s.growth_node, s.growth_level,
                    s.growth_anchor, s.active & !occupied);
                signed_id = dr::select(occupied, signed_id, proposal_id);
                // Aligned octree blocks containing the same cell are nested.
                // Their smaller empty block is empty in the union as well.
                empty_log = dr::select(occupied,empty_log,dr::minimum(empty_log, proposal_log));
            }
            Mask visit = occupied | (collect_growth && m_growth_count ?
                         s.active & (signed_id <= -2) : Mask(false));
            if (profile_traversal) {
                s.traversal_steps += dr::select(s.active, 1u, 0u);
                s.traversal_leaves += dr::select(visit, 1u, 0u);
            }
            if (use_octree && !mixed_tree) {
                Mask empty = s.active & !visit;
                s = dr::if_stmt(dr::make_tuple(s, empty, empty_log), empty,
                    [this, step, step_t, origin = s.segment.origin, inv_d, exit, profile_traversal](
                        State s, const Mask &empty, const UInt32 &empty_log) {
                        UInt32 before = s.iterations;
                        skip_empty_block(s.cell, s.next, s.entry, s.iterations,
                            step, step_t, origin, inv_d, exit, empty, empty_log);
                        if (profile_traversal) {
                            s.traversal_queries += dr::select(empty, 1u, 0u);
                            s.traversal_skipped += s.iterations - before;
                        }
                        return s;
                    }, [](State s, const Mask &, const UInt32 &) { return s; },
                    "Voxel reconstruction octree training skip");
            }
            if (mixed_tree)
                s.next=dr::maximum(s.entry,world_block_boundary_times(
                    s.cell,step,s.segment.origin,inv_d,empty_log));
            Float end=dr::maximum(s.entry,dr::minimum(dr::minimum(s.next.x(),dr::minimum(s.next.y(),s.next.z())),exit));
            if (collect_growth) {
                Mask proposal = s.active & (signed_id <= -2);
                UInt32 growth_id = UInt32(-dr::minimum(signed_id, Int32(-2)) - 2);
                auto buffers = std::make_shared<GrowthBuffers>(
                    GrowthBuffers{r.growth_values, r.growth_ids, r.growth_unknown});
                s = dr::if_stmt(dr::make_tuple(s, growth_id, end, proposal, s.segment.origin, d), proposal,
                    [this, growth_intersects, buffers, lane, ray_count, growth_stride, max_growth_hits](
                        State s, const UInt32 &head,
                        const Float &end, const Mask &proposal,
                        const Point3f &origin, const Vector3f &direction) {
                        // One record per intersected cell, irrespective of the
                        // number of parent alternatives. Walk all siblings so
                        // a miss by the first proposal cannot hide another hit.
                        UInt32 id = dr::select(proposal,head,UInt32(UINT32_MAX));
                        Mask any_hit(false), full = s.growth_hits>=max_growth_hits;
                        dr::tie(id,any_hit,buffers->unknown) = dr::while_loop(
                            dr::make_tuple(id,any_hit,buffers->unknown),
                            [](const UInt32 &id,const Mask &,const UIntBuffer &) {
                                return id!=UInt32(UINT32_MAX);
                            },
                            [&](UInt32 &id,Mask &any_hit,UIntBuffer &unknown) {
                                Mask valid = id!=UInt32(UINT32_MAX);
                                Mask hit = growth_intersects(id,s.entry,end,
                                    origin,direction,valid);
                                any_hit |= hit;
                                dr::scatter_reduce(ReduceOp::Add,unknown,UInt32(1),id,hit&full);
                                id = dr::gather<UInt32>(m_growth_next,id,valid);
                            },"Voxel reconstruction growth cell intersections");
                        Mask stored = any_hit & !full;
                        UInt32 slot = dr::minimum(s.growth_hits,max_growth_hits-1)*ray_count+lane;
                        dr::scatter(buffers->ids,head,slot,stored);
                        for (uint32_t j=0; j<3; ++j)
                            dr::scatter(buffers->values,s.color[j],j*growth_stride+slot,stored);
                        dr::scatter(buffers->values,s.transmittance,3*growth_stride+slot,stored);
                        dr::scatter(buffers->values,s.entry,4*growth_stride+slot,stored);
                        dr::scatter(buffers->values,end,5*growth_stride+slot,stored);
                        s.growth_hits = dr::minimum(s.growth_hits+dr::select(any_hit,1u,0u),max_growth_hits);
                        return s;
                    },
                    [](State s,const UInt32 &,const Float &,const Mask &,
                       const Point3f &,const Vector3f &) { return s; },
                    "Voxel reconstruction hypothetical growth hit");
                r.growth_values = buffers->values;
                r.growth_ids = buffers->ids;
                r.growth_unknown = buffers->unknown;
            }
            // Keep empty cells out of geometry, appearance, and candidate work.
            // Only lane-local values participate in conditional merges. Hit
            // buffers have different widths and are captured as scatter targets.
            // Shared ownership also keeps the targets alive for callback replay.
            auto hit_buffers=std::make_shared<HitBuffers>(HitBuffers{r.hit,r.hit_id});
            if (m_active_count) dr::tie(s,r.candidates)=dr::if_stmt(
                dr::make_tuple(s,r.candidates,id,end,occupied,s.segment.origin,d),occupied,
                [this,rotate,quaternion,appearance,ray_count,hit_stride,lane,
                 max_candidates,diagonal,hit_buffers,distortion_enabled,world_d,world_shell,
                 compact_path_records,hit_before_field,hit_depth_field,
                 grid_entry=entry](State s,Candidates candidates,const UInt32 &id,
                    const Float &end,const Mask &occupied,const Point3f &origin,
                    const Vector3f &direction) {
                    const bool voxel_background = world_shell && !m_background_use_ellipsoid;
                    Mask ellipsoid = voxel_background ? core_ids(id, occupied) : occupied;
                    auto intersect = [this, rotate, quaternion](const UInt32 &id,
                        const Vector3i &cell, const Point3f &origin, const Vector3f &direction,
                        const Float &entry, const Float &end, const Mask &occupied) {
                        Vector3f relative,scale,segment=direction*(end-entry);
                        UInt32 log_size = leaf_log_size(id, occupied);
                        Float block(UInt32(1u) << log_size);
                        for(uint32_t a=0;a<3;++a) {
                            Float center=dr::gather<Float>(m_centers,id*3+a,occupied);
                            Int32 lower = Int32((UInt32(cell[a]) >> log_size) << log_size);
                            relative[a]=origin[a]+entry*direction[a]-(m_min[a]+(Float(lower-int(core_offset()))+center*block)*tree_cell_size(a));
                            scale[a]=dr::exp(dr::clip(dr::gather<Float>(m_log_scales,id*3+a,occupied),-30.f,30.f));
                        }
                        Float qw;
                        Vector3f qv;
                        quaternion(id,occupied,qw,qv);
                        Vector3f u=rotate(relative,qw,-qv)/scale, v=rotate(segment,qw,-qv)/scale;
                        Float a=dr::squared_norm(v), b=2.f*dr::dot(u,v), c=dr::squared_norm(u)-1.f;
                        Float discriminant=b*b-4.f*a*c, root=dr::sqrt(dr::maximum(discriminant,0.f));
                        Float inv=.5f/dr::select(a>1e-20f,a,1.f);
                        Float t0=dr::maximum(0.f,(-b-root)*inv), t1=dr::minimum(1.f,(-b+root)*inv);
                        Mask hit=occupied & dr::select(a<=1e-20f,c<=0.f,(discriminant>=0.f)&(t0<=t1));
                        Float star=dr::select(a>1e-20f,dr::clip(-dr::dot(u,v)/dr::select(a>1e-20f,a,1.f),0.f,1.f),0.f);
                        Vector3f relative_star=relative+star*segment, unit_star=u+star*v;
                        Float gmin=dr::squared_norm(unit_star)-1.f;
                        return dr::make_tuple(relative_star, unit_star, scale, gmin, star, a, hit);
                    };
                    Vector3f relative_star, unit_star, scale;
                    Float gmin, star, a;
                    Mask hit;
                    dr::tie(relative_star, unit_star, scale, gmin, star, a, hit) =
                        voxel_background ? dr::if_stmt(
                            dr::make_tuple(id, s.cell, origin, direction, s.entry, end, ellipsoid),
                            ellipsoid, intersect,
                            [](const UInt32 &, const Vector3i &, const Point3f &, const Vector3f &,
                               const Float &, const Float &, const Mask &) {
                                // Fixed background depth is the world-leaf segment midpoint.
                                return dr::make_tuple(Vector3f(0.f), Vector3f(0.f), Vector3f(1.f),
                                    Float(0.f), Float(.5f), Float(0.f), Mask(false));
                            }, "Voxel reconstruction core training intersection") :
                            intersect(id, s.cell, origin, direction, s.entry, end, ellipsoid);
                    if (voxel_background)
                        hit |= occupied & !ellipsoid & (end > s.entry);
                    Float depth(0.f);
                    Float depth_chain(0.f);
                    if (distortion_enabled) {
                        Float distance=s.entry+star*(end-s.entry);
                        Mask free=ellipsoid&(a>1e-20f)&(star>0.f)&(star<1.f);
                        depth=(distance-grid_entry)/diagonal;
                        depth_chain=Float(free);
                        if (world_shell) {
                            auto mapped=world_depth(distance);
                            depth=mapped.first;
                            depth_chain=dr::select(free,mapped.second,0.f);
                        }
                    }
                    s=dr::if_stmt(
                        dr::make_tuple(s,id,relative_star,gmin,hit,direction,depth,depth_chain),hit,
                        [appearance,ray_count,hit_stride,lane,hit_buffers,distortion_enabled,world_d,
                         compact_path_records,hit_before_field,hit_depth_field](State s,
                            const UInt32 &id,const Vector3f &relative_star,
                            const Float &gmin,const Mask &hit,const Vector3f &direction,
                            const Float &depth,const Float &depth_chain) {
                            Color3f emission;
                            Float opacity;
                            appearance(id,Vector3f(-world_d),hit,emission,opacity);
                            UInt32 slot=s.hits*ray_count+lane;
                            dr::scatter(hit_buffers->hit_id,id,slot,hit);
                            for(uint32_t j=0;j<3;++j) {
                                dr::scatter(hit_buffers->hit,relative_star[j],j*hit_stride+slot,hit);
                                if (!compact_path_records)
                                    dr::scatter(hit_buffers->hit,emission[j],(4+j)*hit_stride+slot,hit);
                            }
                            dr::scatter(hit_buffers->hit,gmin,3*hit_stride+slot,hit);
                            if (!compact_path_records)
                                dr::scatter(hit_buffers->hit,opacity,7*hit_stride+slot,hit);
                            dr::scatter(hit_buffers->hit,s.transmittance,hit_before_field*hit_stride+slot,hit);
                            Float hit_alpha=dr::select(hit,opacity,0.f);
                            if (distortion_enabled) {
                                dr::scatter(hit_buffers->hit,depth,hit_depth_field*hit_stride+slot,hit);
                                dr::scatter(hit_buffers->hit,depth_chain,(hit_depth_field+1u)*hit_stride+slot,hit);
                                Float contribution = s.transmittance*hit_alpha;
                                s.distortion += 2.f*contribution*
                                    (depth*s.distortion_mass-s.distortion_moment);
                                s.distortion_mass += contribution;
                                s.distortion_moment += contribution*depth;
                            }
                            s.color += s.transmittance*hit_alpha*emission;
                            s.transmittance *= 1.f-hit_alpha;
                            s.hits += dr::select(hit,1u,0u);
                            return s;
                        },
                        [](State s,const UInt32 &,const Vector3f &,
                           const Float &,const Mask &,const Vector3f &,
                           const Float &,const Float &) {
                            return s;
                        },"Voxel reconstruction training hit");
                    if(max_candidates) {
                        Mask miss=ellipsoid & !hit & dr::isfinite(gmin) & (gmin>=0.f);
                        dr::tie(s,candidates)=dr::if_stmt(
                            dr::make_tuple(s,candidates,id,relative_star,unit_star,scale,gmin,
                                           star,end,miss),miss,
                            [this,max_candidates,diagonal,world_shell,compact_path_records](State s,Candidates candidates,
                                const UInt32 &id,const Vector3f &relative_star,
                                const Vector3f &unit_star,const Vector3f &scale,
                                const Float &gmin,const Float &star,const Float &end,
                                const Mask &miss) {
                                Float radius2=dr::squared_norm(unit_star);
                                Float metric=2.f*dr::norm(unit_star/dr::sqrt(dr::maximum(radius2,1e-30f))/scale);
                                Mask valid_distance=dr::isfinite(metric)&(metric>0.f)&(radius2>1e-12f);
                                Float distance=gmin/dr::select(valid_distance,metric,1.f);
                                Float depth=s.entry+star*(end-s.entry);
                                Float depth_score=world_shell ? world_depth(depth).first :
                                    dr::abs(depth)/std::max(diagonal,1e-6f);
                                Float score=dr::select(valid_distance,dr::maximum(distance,0.f)/dr::maximum(leaf_size(id,miss)[0],1e-6f)+
                                                      .05f*depth_score,1e30f);
                                // T is constant between hard hits. Reconstruct
                                // it from the next recorded hit (or final T)
                                // after traversal instead of retaining a plane
                                // per geometric candidate.
                                Candidate incoming{relative_star,gmin,
                                    compact_path_records ? Float(0.f) : s.transmittance,
                                    score,id,s.hits};
                                // A full list keeps earlier candidates on equal scores.
                                Mask insert=(s.candidates<max_candidates) |
                                            (score<candidates.back().score);
                                dr::tie(s.candidates,candidates)=dr::if_stmt(
                                    dr::make_tuple(s.candidates,candidates,incoming,miss),insert,
                                    [max_candidates](UInt32 count,Candidates candidates,
                                        Candidate incoming,const Mask &miss) {
                                        Mask carry=miss;
                                        // Keep the original stable insertion and tie order.
                                        for(uint32_t j=0;j<max_candidates;++j) {
                                            Candidate old=candidates[j];
                                            Mask old_valid=j<count;
                                            Mask replace=carry & (!old_valid | (incoming.score<old.score));
                                            candidates[j]=dr::select(replace,incoming,old);
                                            incoming=dr::select(replace,old,incoming);
                                            carry &= !replace | old_valid;
                                        }
                                        count=dr::minimum(count+dr::select(miss,1u,0u),max_candidates);
                                        return dr::make_tuple(count,candidates);
                                    },
                                    [](UInt32 count,Candidates candidates,const Candidate &,
                                       const Mask &) {
                                        return dr::make_tuple(count,candidates);
                                    },"Voxel reconstruction training candidate insertion");
                                return dr::make_tuple(s,candidates);
                            },
                            [](State s,Candidates candidates,const UInt32 &,const Vector3f &,
                               const Vector3f &,const Vector3f &,const Float &,
                               const Float &,const Float &,const Mask &) {
                                return dr::make_tuple(s,candidates);
                            },"Voxel reconstruction training miss");
                    }
                    return dr::make_tuple(s,candidates);
                },
                [](State s,Candidates candidates,const UInt32 &,const Float &,const Mask &,
                   const Point3f &,const Vector3f &) {
                    return dr::make_tuple(s,candidates);
                },"Voxel reconstruction training occupied cell");
            r.hit=hit_buffers->hit;
            r.hit_id=hit_buffers->hit_id;
            advance_ray_cell(s.cell,s.next,s.entry,s.iterations,step,step_t,
                s.segment.origin,d,inv_d,exit,end,empty_log,use_octree);
            Mask in_segment=dr::all(s.cell>=0)&dr::all(s.cell<int(root_resolution()))&(s.entry<exit);
            Mask budget=s.iterations<=3*root_resolution()+3;
            s.active &= in_segment & budget;
            s.active &= (s.transmittance>cfg["transmittance_threshold"])&(s.hits<max_hits);
        },"Voxel reconstruction training paths");
    if (profiler.enabled()) {
        if constexpr (dr::is_jit_v<Float>) {
            // Scheduling Records alone does not recurse into its std::vector.
            // Materialize every candidate field here so candidate construction
            // is measured with the DDA, not deferred into the backward stage.
            for (const auto &candidate : records.candidates)
                dr::schedule(candidate.point, candidate.gmin, candidate.before,
                             candidate.score, candidate.id, candidate.front);
            dr::schedule(records, state);
            dr::eval();
        }
        profiler.finish("forward");
        profiler.begin();
    } else {
        dr::eval(records,state);
    }
    if (compact_path_records) {
        // Stable insertion scores decide the exact same candidate ordering,
        // but have no consumer after traversal. Retain the original lazy
        // candidate evaluation: forcing all candidate planes here overlaps
        // their allocation with the whole-image D-SSIM workspace.
        for (auto &candidate : records.candidates)
            candidate.score = Float();
    }

    TrainingResult result;
    result.raw_rgb=state.color+state.transmittance*background;
    result.raw_alpha=1.f-state.transmittance;
    result.hit_count=state.hits;
    if (profile_traversal) {
        result.counters["traversal_steps"] = state.traversal_steps;
        result.counters["traversal_skipped_cells"] = state.traversal_skipped;
        result.counters["traversal_leaf_visits"] = state.traversal_leaves;
        result.counters["traversal_tree_queries"] = state.traversal_queries;
    }
    Float weight=dr::select(inverse_pdf>0.f,inverse_pdf,1.f);
    result.rgb=Color3f(1.f)+(result.raw_rgb-1.f)*weight;
    result.alpha=result.raw_alpha*weight;
    Color3f target=rgb_only ? reference_rgb :
        reference_rgb*reference_alpha+Color3f(1.f)*(1.f-reference_alpha);
    // Rendering, path prefixes/suffixes and references remain linear. Only the
    // complete, sampling-weighted image and target enter the selected loss space.
    Color3f loss_rgb=rgb_loss_srgb ? reconstruction_linear_to_srgb(result.rgb) : result.rgb;
    Color3f loss_target=rgb_loss_srgb ? reconstruction_linear_to_srgb(target) : target;
    Color3f delta=loss_rgb-loss_target,dcolor=2.f*delta;
    Float dalpha(0.f),deletion_dalpha(0.f);
    Color3f deletion_dcolor(0.f);
    Mask is_background(false),is_foreground(false);
    result.rgb_loss=dr::squared_norm(delta);
    result.distortion_loss=distortion_enabled ?
        dr::select(state.hits>1u,state.distortion*weight,0.f) :
        dr::zeros<Float>(width);
    result.loss=result.rgb_loss;
    if (!rgb_only) {
        dalpha=2.f*cfg["alpha_loss_weight"]*(result.alpha-reference_alpha);
        result.loss+=cfg["alpha_loss_weight"]*dr::square(result.alpha-reference_alpha);
        deletion_dcolor=2.f*(result.raw_rgb-target);
        deletion_dalpha=2.f*cfg["alpha_loss_weight"]*(result.raw_alpha-reference_alpha);
        is_background=reference_alpha<=cfg["background_alpha_max"];
        is_foreground=!is_background & (reference_alpha>=cfg["foreground_alpha_min"]);
    }
    if (distortion_enabled)
        result.loss += distortion_weight*result.distortion_loss;
    if (dssim_enabled) {
        result.dssim_enabled = true;
        auto image_loss = dssim_image_loss_and_grad(loss_rgb, loss_target, image_width, image_height);
        Mask uniform = dr::all(inverse_pdf == 1.f);
        result.dssim_loss = dr::select(uniform, image_loss.first,
                                       std::numeric_limits<float>::quiet_NaN());
        result.loss += dssim_weight * result.dssim_loss;
        dcolor += dssim_weight * image_loss.second;
        dcolor = dr::select(uniform, dcolor,
                            Color3f(std::numeric_limits<float>::quiet_NaN()));
        if (profiler.enabled()) {
            dr::eval(result.rgb_loss, result.distortion_loss, result.dssim_loss, result.loss, dcolor);
            profiler.finish("image_loss");
            profiler.begin();
        }
    }
    // The native backward is explicit, so apply the transfer Jacobian once to
    // the combined L2 + D-SSIM image adjoint before all appearance/geometry uses.
    // Distortion and alpha supervision retain their independent derivatives.
    if (rgb_loss_srgb)
        dcolor *= reconstruction_srgb_derivative(result.rgb, loss_rgb);
    if (growth_requested) {
        result.growth_sums["growth_gain"] = dr::zeros<FloatBuffer>(m_growth_count);
        result.growth_counts["growth_observed"] = dr::zeros<UIntBuffer>(m_growth_count);
        result.growth_counts["growth_unknown"] = collect_growth ?
            records.growth_unknown : dr::zeros<UIntBuffer>(m_growth_count);
    }
    if (collect_growth) {
        // No second forward pass: an insertion changes the final color by
        // alpha * (T_before * radiance - already-weighted baseline suffix).
        // Empty candidate cells cannot reorder existing clipped-cell hits.
        // A stopped path has no reliable full suffix, and inserting a hit can
        // itself trigger the cap/early exit. Be conservative and report unknown
        // for both cases instead of extending traversal or inventing evidence.
        Mask exited =
            (!dr::all(state.cell>=0) | !dr::all(state.cell<int(root_resolution())) | (state.entry>=exit));
        Mask complete = exited & (state.iterations<=3*root_resolution()+3) &
                        (state.hits+1u<max_hits) &
                        (weight==1.f) & dr::all(dr::isfinite(result.raw_rgb)) &
                        dr::all(dr::isfinite(reference_rgb));
        for (uint32_t k=0; k<max_growth_hits; ++k) {
            Mask stored = k<state.growth_hits;
            UInt32 slot = k*ray_count+lane;
            UInt32 head=dr::gather<UInt32>(records.growth_ids,slot,stored);
            UInt32 id=dr::select(stored,head,UInt32(UINT32_MAX));
            Color3f prefix;
            for (uint32_t j=0; j<3; ++j)
                prefix[j] = dr::gather<Float>(records.growth_values,j*growth_stride+slot,stored);
            Float before = dr::gather<Float>(records.growth_values,3*growth_stride+slot,stored);
            Float segment_entry = dr::gather<Float>(records.growth_values,4*growth_stride+slot,stored);
            Float segment_end = dr::gather<Float>(records.growth_values,5*growth_stride+slot,stored);
            // The saved segment and baseline prefix are shared by all proposals
            // in this cell. Rechecking their own shapes needs no additional DDA
            // or actual rendering, and an arbitrary number of siblings can be
            // evaluated without consuming the per-ray cell-record capacity.
            dr::tie(id,result.growth_sums["growth_gain"],result.growth_counts["growth_observed"],
                    result.growth_counts["growth_unknown"]) = dr::while_loop(
                dr::make_tuple(id,result.growth_sums["growth_gain"],
                    result.growth_counts["growth_observed"],result.growth_counts["growth_unknown"]),
                [](const UInt32 &id,const FloatBuffer &,const UIntBuffer &,const UIntBuffer &) {
                    return id!=UInt32(UINT32_MAX);
                },
                [&](UInt32 &id,FloatBuffer &gains,UIntBuffer &observed,UIntBuffer &unknown) {
                    Mask valid = id!=UInt32(UINT32_MAX);
                    Mask hit = growth_intersects(id,segment_entry,segment_end,ray.o,d,valid);
                    Color3f radiance(0.f);
                    Float logit(0.f);
                    for (uint32_t j=0; j<std::max(m_active_sh_count,m_active_opacity_sh_count); ++j) {
                        Float sh = reconstruction_sh(j,Vector3f(-d));
                        if (j<m_active_sh_count)
                            for (uint32_t channel=0; channel<3; ++channel)
                                radiance[channel] += sh*dr::gather<Float>(m_growth_radiance,
                                    (id*m_sh_count+j)*3+channel,hit);
                        if (j<m_active_opacity_sh_count)
                            logit += sh*dr::gather<Float>(m_growth_opacity,id*m_opacity_sh_count+j,hit);
                    }
                    radiance = dr::maximum(clean(radiance),0.f);
                    Float opacity = dr::rcp(1.f+dr::exp(-logit));
                    Color3f change = opacity*(before*radiance-(result.raw_rgb-prefix));
                    Color3f gain_change = change, gain_residual = result.raw_rgb-reference_rgb;
                    if (rgb_loss_srgb) {
                        // Only weight==1 baselines can be known evidence. The
                        // nonlinear transfer requires the full counterfactual,
                        // not its first-order Jacobian. Expand in encoded space
                        // to retain small gains without subtracting two losses.
                        gain_change = reconstruction_linear_to_srgb(result.raw_rgb+change)-loss_rgb;
                        gain_residual = delta;
                    }
                    Float gain = -2.f*dr::dot(gain_residual,gain_change)-dr::squared_norm(gain_change);
                    Mask known = hit & complete & dr::isfinite(gain) &
                        (state.transmittance*(1.f-opacity)>cfg["transmittance_threshold"]);
                    dr::scatter_reduce(ReduceOp::Add,gains,dr::detach(gain),id,known);
                    dr::scatter_reduce(ReduceOp::Add,observed,UInt32(1),id,known);
                    dr::scatter_reduce(ReduceOp::Add,unknown,UInt32(1),id,hit&!known);
                    id = dr::gather<UInt32>(m_growth_next,id,valid);
                },"Voxel reconstruction growth cell evidence");
        }
    }
    result.compact_gradients = compact_gradients;
    UIntBuffer gradient_slots = dr::zeros<UIntBuffer>(1);
    if (compact_gradients && m_active_count) {
        // Mark every recorded hit and geometric near-miss, including rows whose
        // numerical gradient is zero: their validity still advances Adam clocks.
        // One dense UInt32 prefix replaces all dense floating-point gradients.
        // No compress()/host count readback occurs inside a frozen native call.
        UIntBuffer touched = dr::zeros<UIntBuffer>(m_active_count);
        for (uint32_t k = 0; k < max_hits; ++k) {
            Mask valid = k < state.hits;
            UInt32 id = dr::gather<UInt32>(records.hit_id, k * ray_count + lane, valid);
            dr::scatter_reduce(ReduceOp::Max, touched, UInt32(1), id, valid);
        }
        for (uint32_t k = 0; k < max_candidates; ++k)
            dr::scatter_reduce(ReduceOp::Max, touched, UInt32(1), records.candidates[k].id,
                               k < state.candidates);
        dr::eval(touched);
        gradient_slots = dr::prefix_sum(touched, false);
        result.voxel_count = dr::gather<UInt32>(gradient_slots, UInt32(m_active_count - 1));
        result.compact_gradient_overflow = result.voxel_count > gradient_rows;
        result.voxel_ids = dr::full<UIntBuffer>(UINT32_MAX, gradient_rows);
        UIntBuffer ordinal = dr::arange<UIntBuffer>(m_active_count);
        auto stored = (touched != 0u) & (gradient_slots <= gradient_rows);
        dr::scatter(result.voxel_ids, ordinal, gradient_slots - 1u, stored, ReduceMode::NoConflicts);
        dr::eval(gradient_slots, result.voxel_ids, result.voxel_count, result.compact_gradient_overflow);
    } else if (compact_gradients) {
        result.voxel_ids = dr::empty<UIntBuffer>(0);
    }
    auto gradient_row = [compact_gradients, gradient_slots](const UInt32 &id, const Mask &active) {
        return compact_gradients ? dr::gather<UInt32>(gradient_slots, id, active) - 1u : id;
    };
    for(const auto &[name,size]:std::map<std::string,size_t>{{"centers",size_t(gradient_rows)*3},
        {"log_scales",size_t(gradient_rows)*3},{"rotation_tangent",size_t(gradient_rows)*3},
        {"radiance_sh",size_t(gradient_rows)*m_sh_count*3},{"opacity_sh",size_t(gradient_rows)*m_opacity_sh_count}})
        result.gradients[name]=dr::zeros<FloatBuffer>(size);
    for(const auto *key:{"appearance_valid","geometry_valid","background_carve_valid"})
        result.counters[key]=dr::zeros<UIntBuffer>(gradient_rows);
    for(const auto *key:{"foreground_evidence","background_evidence"})
        result.counters[key]=dr::zeros<UIntBuffer>(m_active_count);
    if (collect_adaptive_priority)
        result.deletion_sums["adaptive_priority"] = dr::zeros<FloatBuffer>(m_active_count);
    if (collect_rgb_deletion) {
        result.deletion_sums["deletion_gain"] = dr::zeros<FloatBuffer>(m_active_count);
        result.deletion_counts["deletion_observed"] = dr::zeros<UIntBuffer>(m_active_count);
        result.deletion_counts["deletion_unknown"] = dr::zeros<UIntBuffer>(m_active_count);
    }
    if (!m_active_count) {
        if (profiler.enabled()) {
            evaluate_result(result);
            profiler.finish("hit_backward");
            profiler.zero_stage("candidate_backward");
            result.stage_profile = std::move(profiler.profile);
        }
        return result;
    }
    using GradientMap = std::map<std::string, FloatBuffer>;
    using CounterMap = std::map<std::string, UIntBuffer>;
    auto add_geometry=[this,quaternion,rotate,clean,is_background,dcolor,dalpha,weight,cfg,rgb_only,
                       gradient_row,gradient_rows](
                          GradientMap &gradients,CounterMap &counters,
                          UInt32 id,const Vector3f &offset,Float gmin,Float before,
                          Float opacity,const Color3f &radiance,const Color3f &behind,
                          Float after,Mask valid,bool hard_hit) {
        const bool voxel_background = m_background_shell_levels != 0 && !m_background_use_ellipsoid;
        if (voxel_background)
            valid &= core_ids(id, valid);
        Vector3f size = leaf_size(id, valid);
        Float tau = dr::maximum(cfg.at("geometry_tau_voxel_fraction")*size[0],1e-8f);
        auto evaluate = [this,quaternion,rotate,clean,is_background,dcolor,dalpha,tau,size,cfg,rgb_only,hard_hit](
            const UInt32 &id,const Vector3f &offset,const Float &gmin,const Float &before,
            const Float &opacity,const Color3f &radiance,const Color3f &behind,
            const Float &after,Mask valid) {
            Float qw;Vector3f qv,scale,center;
            quaternion(id,valid,qw,qv);
            for(uint32_t j=0;j<3;++j) {
                scale[j]=dr::exp(dr::clip(dr::gather<Float>(m_log_scales,id*3+j,valid),-30.f,30.f));
                center[j]=dr::gather<Float>(m_centers,id*3+j,valid);
            }
            Vector3f principal=rotate(offset,qw,-qv),unit=principal/scale;
            Float radius2=dr::squared_norm(unit);
            Float metric=2.f*dr::norm(unit/dr::sqrt(dr::maximum(radius2,1e-30f))/scale);
            valid &= dr::isfinite(gmin)&dr::isfinite(radius2)&(radius2>1e-12f)&dr::isfinite(metric)&(metric>0.f);
            if(!rgb_only && !hard_hit) valid &= !is_background;
            Float distance_scale=dr::rcp(dr::select(valid,metric,1.f));
            Float distance=gmin*distance_scale,w=dr::rcp(1.f+dr::exp(distance/tau));
            Float derivative=dr::dot(dcolor,before*opacity*(radiance-behind))*(-w*(1.f-w)/tau);
            if(!rgb_only) {
                if(hard_hit) derivative=dr::select(is_background,-cfg.at("background_carve_weight")*w/tau,derivative);
                derivative += cfg.at("alpha_geometry_weight")*dalpha*before*opacity*after*(-w*(1.f-w)/tau);
            }
            Float dg=dr::clip(clean(derivative*distance_scale),-cfg.at("geometry_grad_clamp"),cfg.at("geometry_grad_clamp"));
            Mask contains(true);
            const bool two_sided=!rgb_only && cfg.at("geometry_two_sided_boundary")!=0.f;
            const bool transition_only=!rgb_only && cfg.at("geometry_transition_only")!=0.f;
            if(hard_hit && (two_sided || (transition_only && cfg.at("geometry_hit_expansion_band_tau")>0.f))) {
                for(uint32_t corner=0;corner<8;++corner) {
                    Vector3f corner_offset;
                    for(uint32_t j=0;j<3;++j) corner_offset[j]=(float((corner>>j)&1u)-center[j])*size[j];
                    contains &= dr::squared_norm(rotate(corner_offset,qw,-qv)/scale)<=1.f;
                }
            }
            if(two_sided) {
                Float band=tau*cfg.at("geometry_boundary_band_tau");
                Float t=dr::select(band>0.f,dr::clip(1.f-dr::abs(distance)/dr::select(band>0.f,band,1.f),0.f,1.f),0.f);
                Float taper=t*t*(3.f-2.f*t);
                dg*=hard_hit?dr::select(is_background,1.f,taper):taper;
                if(hard_hit) dg=dr::select((dg>0.f)&(is_background|contains),0.f,dg);
            } else if(transition_only) {
                if(hard_hit) {
                    Float band=tau*cfg.at("geometry_hit_expansion_band_tau");
                    Float t=dr::select(band>0.f,dr::clip(1.f+distance/dr::select(band>0.f,band,1.f),0.f,1.f),0.f);
                    Float taper=dr::select(!is_background & !contains,t*t*(3.f-2.f*t),0.f);
                    dg=dr::select(dg>0.f,dg*taper,dg);
                } else dg=dr::maximum(dg,0.f);
            }
            Vector3f weighted=principal/(scale*scale);
            Vector3f gc=-2.f*rotate(weighted,qw,qv),gs=-2.f*weighted*principal,gr=2.f*dr::cross(weighted,principal);
            return dr::make_tuple(gc, gs, gr, dg, valid);
        };
        Vector3f gc, gs, gr;
        Float dg;
        dr::tie(gc, gs, gr, dg, valid) = voxel_background ? dr::if_stmt(
            dr::make_tuple(id, offset, gmin, before, opacity, radiance, behind, after, valid),
            valid, evaluate,
            [](const UInt32 &,const Vector3f &,const Float &,const Float &,const Float &,
               const Color3f &,const Color3f &,const Float &,const Mask &) {
                return dr::make_tuple(Vector3f(0.f), Vector3f(0.f), Vector3f(0.f), Float(0.f), Mask(false));
            }, "Voxel reconstruction core geometry proxy") :
            evaluate(id, offset, gmin, before, opacity, radiance, behind, after, valid);
        UInt32 row = gradient_row(id, valid);
        Mask stored = valid & (row < gradient_rows);
        // Apply 1/pdf after the nonlinear proxy clamp and transition gates.
        for(uint32_t j=0;j<3;++j) {
            dr::scatter_reduce(ReduceOp::Add,gradients["centers"],clean(dg*gc[j]*size[j])*weight,row*3+j,stored);
            dr::scatter_reduce(ReduceOp::Add,gradients["log_scales"],clean(dg*gs[j])*weight,row*3+j,stored);
            dr::scatter_reduce(ReduceOp::Add,gradients["rotation_tangent"],clean(dg*gr[j])*weight,row*3+j,stored);
        }
        dr::scatter_reduce(ReduceOp::Add,counters["geometry_valid"],UInt32(1),row,stored);
        if(!rgb_only && hard_hit) dr::scatter_reduce(ReduceOp::Add,counters["background_carve_valid"],UInt32(1),row,stored&is_background);
        return valid;
    };

    // Hit records stay read-only inside the symbolic reverse loop. Separate
    // suffix storage avoids a gather-after-scatter evaluation at every hit.
    Mask deletion_complete(false);
    Color3f deletion_residual(0.f);
    if (collect_rgb_deletion) {
        // Removing an actual hit can reveal an untraced suffix after opacity
        // or capacity termination. Only a normally exited baseline certifies
        // that the recorded suffix is complete; no extra traversal is done.
        // Removal reduces hit count and increases T, so insertion's spare-hit
        // capacity and reduced-final-T checks do not apply.
        Mask exited =
            (!dr::all(state.cell>=0) | !dr::all(state.cell<int(root_resolution())) | (state.entry>=exit));
        deletion_complete = exited & (state.iterations<=3*root_resolution()+3) &
                            (weight==1.f) & dr::isfinite(inverse_pdf) &
                            dr::all(dr::isfinite(result.raw_rgb)) &
                            dr::all(dr::isfinite(reference_rgb));
        deletion_residual = rgb_loss_srgb ? delta : result.raw_rgb-reference_rgb;
    }
    struct ReverseState {
        Int32 hit;
        Color3f behind;
        Float after;
        FloatBuffer suffix;
        GradientMap gradients;
        CounterMap counters;
        GradientMap deletion_sums;
        CounterMap deletion_counts;
        Float distortion_mass, distortion_moment, distortion_weight_derivative;
        DRJIT_STRUCT(ReverseState,hit,behind,after,suffix,gradients,counters,
                     deletion_sums,deletion_counts,distortion_mass,
                     distortion_moment,distortion_weight_derivative)
    } reverse{dr::full<Int32>(int(max_hits)-1,width),background,Float(1.f),
              dr::zeros<FloatBuffer>(size_t(hit_stride)*(compact_path_records ? 3u : 4u)),
              std::move(result.gradients),std::move(result.counters),
              std::move(result.deletion_sums),std::move(result.deletion_counts),
              Float(0.f),Float(0.f),Float(0.f)};
    dr::tie(reverse)=dr::while_loop(dr::make_tuple(std::move(reverse)),
        [](const ReverseState &s) { return s.hit>=0; },
        [this,hit_records=records.hit,hit_ids=records.hit_id,hit_count=state.hits,
         ray_count,hit_stride,lane,clean,dcolor,dalpha,d,weight,is_background,
         is_foreground,deletion_dcolor,deletion_dalpha,cfg,add_geometry,rgb_only,
         collect_rgb_deletion,collect_adaptive_priority,deletion_complete,deletion_residual,distortion_enabled,
         rgb_loss_srgb,deletion_rgb=result.raw_rgb,deletion_encoded_rgb=loss_rgb,
         distortion_weight,diagonal,quaternion,rotate,world_shell,
         total_mass=state.distortion_mass,total_moment=state.distortion_moment,
         gradient_row,gradient_rows,appearance,compact_path_records,hit_before_field,
         hit_depth_field](ReverseState &s) {
            Mask valid=UInt32(s.hit)<hit_count;
            UInt32 slot=UInt32(s.hit)*ray_count+lane,id=dr::gather<UInt32>(hit_ids,slot,valid);
            UInt32 row = gradient_row(id, valid);
            Mask stored = valid & (row < gradient_rows);
            Vector3f offset;Color3f radiance;
            for(uint32_t j=0;j<3;++j) {
                offset[j]=dr::gather<Float>(hit_records,j*hit_stride+slot,valid);
                if (!compact_path_records)
                    radiance[j]=dr::gather<Float>(hit_records,(4+j)*hit_stride+slot,valid);
            }
            Float gmin=dr::gather<Float>(hit_records,3*hit_stride+slot,valid);
            Float opacity;
            if (compact_path_records)
                appearance(id,Vector3f(-d),valid,radiance,opacity);
            else
                opacity=dr::gather<Float>(hit_records,7*hit_stride+slot,valid);
            Float before=dr::gather<Float>(hit_records,hit_before_field*hit_stride+slot,valid);
            Color3f grad_rgb=clean(dcolor*before*opacity);
            grad_rgb=dr::select(radiance<=0.f,dr::minimum(grad_rgb,0.f),grad_rgb);
            if(!rgb_only) grad_rgb=dr::select(is_background,Color3f(0.f),grad_rgb);
            grad_rgb*=weight;
            Float grad_opacity=clean(dr::dot(dcolor,before*(radiance-s.behind))+dalpha*before*s.after);
            // Appearance sensitivity before the sigmoid chain. This measures
            // RGB/D-SSIM reconstruction error, not the distortion regularizer;
            // it reuses the current reverse suffix and needs no extra tracing.
            if (collect_adaptive_priority)
                dr::scatter_reduce(ReduceOp::Add,s.deletion_sums["adaptive_priority"],
                    dr::detach(clean(dr::abs(opacity*weight*grad_opacity))),id,valid);
            Float grad_logit=grad_opacity*weight*opacity*(1.f-opacity);
            Mask distortion_geometry_valid(false);
            if (distortion_enabled) {
                Float depth=dr::gather<Float>(hit_records,hit_depth_field*hit_stride+slot,valid);
                Float depth_chain=dr::gather<Float>(hit_records,(hit_depth_field+1u)*hit_stride+slot,valid);
                Mask depth_free=depth_chain>0.f;
                Float contribution=dr::select(valid,before*opacity,0.f);
                Float left_mass=total_mass-contribution-s.distortion_mass;
                Float left_moment=total_moment-contribution*depth-s.distortion_moment;
                Float derivative=2.f*(depth*left_mass-left_moment+
                                      s.distortion_moment-depth*s.distortion_mass);
                // A single point has identically zero distortion. Enforce
                // this contract explicitly despite possible fused-operation
                // rounding in total_moment - contribution*depth.
                derivative=dr::select(hit_count>1u,derivative,0.f);
                // q_i = dD/dw_i. Differentiating all later transmittances
                // yields this division-free sigmoid-logit derivative, including
                // alpha == 1 without a potentially singular 1/(1-alpha).
                grad_logit += distortion_weight*weight*clean(
                    derivative*contribution*(1.f-opacity)-
                    opacity*s.distortion_weight_derivative);
                Float depth_derivative=distortion_weight*weight*
                    (2.f*contribution*(left_mass-s.distortion_mass))/diagonal;
                if (world_shell)
                    depth_derivative=distortion_weight*weight*
                        (2.f*contribution*(left_mass-s.distortion_mass))*depth_chain;
                depth_derivative=dr::select(hit_count>1u,depth_derivative,0.f);

                // Interior closest point minimizes p^T diag(scale^-2) p.
                // Its implicit derivative is exact while hard-hit membership,
                // cell clipping, and DDA ordering stay fixed. Clipped endpoints
                // are fixed by the grid, so their geometric depth derivative is zero.
                const bool voxel_background = world_shell && !m_background_use_ellipsoid;
                Mask geometry_hit = voxel_background ? core_ids(id, valid) : valid;
                auto depth_geometry = [this,quaternion,rotate,d](const UInt32 &id,
                    const Vector3f &offset,const Float &depth_derivative,
                    const Mask &depth_free,const Mask &valid) {
                    Float qw;Vector3f qv,scale,log_scale;
                    quaternion(id,valid,qw,qv);
                    for(uint32_t j=0;j<3;++j) {
                        log_scale[j]=dr::gather<Float>(m_log_scales,id*3+j,valid);
                        scale[j]=dr::exp(dr::clip(log_scale[j],-30.f,30.f));
                    }
                    Vector3f local_direction=d;
                    Vector3f principal=rotate(offset,qw,-qv), direction=rotate(local_direction,qw,-qv);
                    Vector3f weighted_point=principal/(scale*scale);
                    Vector3f weighted_direction=direction/(scale*scale);
                    Float denominator=dr::dot(direction,weighted_direction);
                    Mask distortion_geometry_valid=valid&depth_free&dr::isfinite(depth_derivative)&
                        (depth_derivative!=0.f)&dr::isfinite(denominator)&(denominator>0.f);
                    Float factor=dr::select(distortion_geometry_valid,
                        depth_derivative/dr::select(distortion_geometry_valid,denominator,1.f),0.f);
                    Vector3f center_gradient=rotate(weighted_direction,qw,qv);
                    Vector3f scale_gradient=2.f*direction*weighted_point;
                    Vector3f rotation_gradient=dr::cross(direction,weighted_point)+
                                              dr::cross(principal,weighted_direction);
                    Float rotation_norm2=dr::square(dr::gather<Float>(m_rotations,id*4,valid));
                    for(uint32_t j=0;j<3;++j)
                        rotation_norm2 += dr::square(dr::gather<Float>(m_rotations,id*4+j+1,valid));
                    Mask rotation_valid=dr::isfinite(rotation_norm2)&(rotation_norm2>1e-20f);
                    return dr::make_tuple(center_gradient, scale_gradient, rotation_gradient,
                        log_scale, factor, rotation_valid, distortion_geometry_valid);
                };
                Vector3f center_gradient, scale_gradient, rotation_gradient, log_scale;
                Float factor;
                Mask rotation_valid;
                dr::tie(center_gradient, scale_gradient, rotation_gradient, log_scale,
                    factor, rotation_valid, distortion_geometry_valid) = voxel_background ? dr::if_stmt(
                    dr::make_tuple(id, offset, depth_derivative, depth_free, geometry_hit),
                    geometry_hit, depth_geometry,
                    [](const UInt32 &,const Vector3f &,const Float &,const Mask &,const Mask &) {
                        return dr::make_tuple(Vector3f(0.f), Vector3f(0.f), Vector3f(0.f),
                            Vector3f(0.f), Float(0.f), Mask(false), Mask(false));
                    }, "Voxel reconstruction core distortion geometry") :
                    depth_geometry(id, offset, depth_derivative, depth_free, geometry_hit);
                Vector3f size = leaf_size(id, geometry_hit);
                for(uint32_t j=0;j<3;++j) {
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["centers"],
                        clean(factor*center_gradient[j]*size[j]),row*3+j,distortion_geometry_valid&stored);
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["log_scales"],
                        clean(factor*scale_gradient[j]),row*3+j,
                        distortion_geometry_valid&stored&(log_scale[j]>-30.f)&(log_scale[j]<30.f));
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["rotation_tangent"],
                        clean(factor*rotation_gradient[j]),row*3+j,
                        distortion_geometry_valid&stored&rotation_valid);
                }
                s.distortion_mass += contribution;
                s.distortion_moment += contribution*depth;
                s.distortion_weight_derivative += contribution*derivative;
            }
            for(uint32_t k=0;k<std::max(m_active_sh_count,m_active_opacity_sh_count);++k) {
                Float sh=reconstruction_sh(k,Vector3f(-d));
                if(k<m_active_sh_count)
                    for(uint32_t j=0;j<3;++j)
                        dr::scatter_reduce(ReduceOp::Add,s.gradients["radiance_sh"],grad_rgb[j]*sh,(row*m_sh_count+k)*3+j,stored);
                if(k<m_active_opacity_sh_count)
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["opacity_sh"],grad_logit*sh,row*m_opacity_sh_count+k,stored);
            }
            dr::scatter_reduce(ReduceOp::Add,s.counters["appearance_valid"],UInt32(1),row,stored);
            if(!rgb_only && cfg.at("collect_deletion_evidence")!=0.f) {
                Color3f dc=before*opacity*(s.behind-radiance);
                Float da=-before*opacity*s.after;
                Float delta_loss=clean(dr::dot(deletion_dcolor,dc)+dr::squared_norm(dc)+deletion_dalpha*da+cfg.at("alpha_loss_weight")*da*da);
                Mask visible=valid & (before>=cfg.at("min_evidence_transmittance"));
                dr::scatter_reduce(ReduceOp::Or,s.counters["foreground_evidence"],UInt32(1),id,
                    visible&is_foreground&(delta_loss>cfg.at("min_removal_loss_delta")));
                dr::scatter_reduce(ReduceOp::Or,s.counters["background_evidence"],UInt32(1),id,
                    visible&is_background&(delta_loss<-cfg.at("min_removal_loss_delta")));
            }
            if (collect_rgb_deletion) {
                // Exact single-removal color change on a complete baseline:
                // C_without - C = T_before * alpha * (suffix - radiance).
                // Use the existing reverse suffix directly, including when
                // alpha approaches one; never divide by (1-alpha).
                Color3f dc = before*opacity*(s.behind-radiance);
                Color3f gain_change = rgb_loss_srgb ?
                    reconstruction_linear_to_srgb(deletion_rgb+dc)-deletion_encoded_rgb : dc;
                Float gain = -2.f*dr::dot(deletion_residual,gain_change)-dr::squared_norm(gain_change);
                // Projective background hits use the same complete-ray RGB
                // removal test as core hits. Truncated suffixes remain unknown.
                Mask eligible=valid;
                Mask known = eligible & deletion_complete & dr::isfinite(before) &
                             dr::isfinite(opacity) & dr::all(dr::isfinite(radiance)) &
                             dr::all(dr::isfinite(s.behind)) &
                             dr::all(dr::isfinite(dc)) & dr::isfinite(gain);
                dr::scatter_reduce(ReduceOp::Add,s.deletion_sums["deletion_gain"],
                    dr::detach(gain),id,known);
                dr::scatter_reduce(ReduceOp::Add,s.deletion_counts["deletion_observed"],
                    UInt32(1),id,known);
                dr::scatter_reduce(ReduceOp::Add,s.deletion_counts["deletion_unknown"],
                    UInt32(1),id,eligible&!known);
            }
            Mask proxy_geometry_valid=add_geometry(s.gradients,s.counters,id,offset,gmin,
                before,opacity,radiance,s.behind,s.after,valid,true);
            if (distortion_enabled)
                dr::scatter_reduce(ReduceOp::Add,s.counters["geometry_valid"],UInt32(1),
                    row,distortion_geometry_valid&!proxy_geometry_valid&stored);
            s.behind=dr::select(valid,clean(opacity*radiance+(1.f-opacity)*s.behind),s.behind);
            s.after=dr::select(valid,dr::clip(clean(s.after*(1.f-opacity)),0.f,1.f),s.after);
            // Each ray writes each slot once; the suffix planes are disjoint.
            // Candidates read these suffixes only after reverse traversal ends.
            for(uint32_t j=0;j<3;++j)
                dr::scatter(s.suffix,s.behind[j],j*hit_stride+slot,valid,ReduceMode::NoConflicts);
            if (!compact_path_records)
                dr::scatter(s.suffix,s.after,3*hit_stride+slot,valid,ReduceMode::NoConflicts);
            s.hit-=1;
        },"Voxel reconstruction reverse paths");
    result.gradients=std::move(reverse.gradients);
    result.counters=std::move(reverse.counters);
    result.deletion_sums=std::move(reverse.deletion_sums);
    result.deletion_counts=std::move(reverse.deletion_counts);
    // Loop traversal includes maps, while schedule() needs their buffers
    // explicitly. Materialize all reverse outputs at the existing stage boundary.
    if constexpr (dr::is_jit_v<Float>) {
        dr::schedule(reverse.suffix);
        for(const auto &[name,value]:result.gradients) dr::schedule(value);
        for(const auto &[name,value]:result.counters) dr::schedule(value);
        for(const auto &[name,value]:result.deletion_sums) dr::schedule(value);
        for(const auto &[name,value]:result.deletion_counts) dr::schedule(value);
    }
    if (profiler.enabled()) {
        evaluate_result(result);
        profiler.finish("hit_backward");
        profiler.begin();
    } else if constexpr (dr::is_jit_v<Float>) {
        dr::eval();
    }
    for(uint32_t i=0;i<max_candidates;++i) {
        Mask valid=i<state.candidates;
        if(!rgb_only) valid &= !is_background;
        const Candidate &candidate=records.candidates[i];
        UInt32 id=candidate.id,front=candidate.front;
        UInt32 suffix=dr::minimum(front,max_hits-1)*ray_count+lane;
        Mask has_suffix=valid&(front<state.hits);
        Color3f suffix_color;
        for(uint32_t j=0;j<3;++j)
            suffix_color[j]=dr::select(has_suffix,dr::gather<Float>(reverse.suffix,j*hit_stride+suffix,has_suffix),background[j]);
        Float suffix_t(1.f);
        if (!compact_path_records)
            suffix_t=dr::select(has_suffix,dr::gather<Float>(reverse.suffix,3*hit_stride+suffix,has_suffix),1.f);
        Float before = candidate.before;
        if (compact_path_records)
            before = dr::select(has_suffix,
                dr::gather<Float>(records.hit,hit_before_field*hit_stride+suffix,has_suffix),
                state.transmittance);
        Color3f radiance;Float opacity;
        appearance(id,Vector3f(-d),valid,radiance,opacity);
        add_geometry(result.gradients,result.counters,id,candidate.point,candidate.gmin,before,opacity,radiance,suffix_color,suffix_t,valid,false);
    }
    if (profiler.enabled()) {
        evaluate_result(result);
        profiler.finish("candidate_backward");
        result.stage_profile = std::move(profiler.profile);
    }
    return result;
}

MI_VARIANT std::string VoxelReconstruction<Float, Spectrum>::to_string() const {
    std::ostringstream out;
    out << "VoxelReconstruction[resolution=" << m_resolution << ", active_voxels="
        << m_active_count << ", sh_count=" << m_sh_count << ", active_sh_count="
        << m_active_sh_count << ", opacity_sh_count="
        << m_opacity_sh_count << "]";
    return out.str();
}

MI_INSTANTIATE_CLASS(VoxelReconstruction)
NAMESPACE_END(mitsuba)
