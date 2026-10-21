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

/// Stable row permutation, using the same small-bucket radix kernels as
/// Dr.Jit's argsort(). No permutation or cell array crosses the device boundary.
template <typename UInt> UInt reconstruction_cell_order(const UInt &cells) {
    const size_t count = cells.size();
    if (count <= 1)
        return dr::arange<UInt>(count);
    if constexpr (dr::is_jit_v<UInt>) {
        using Index = dr::detached_t<UInt>;
        Index ordinal = dr::detach<false>(cells), order = dr::arange<Index>(count);
        constexpr uint32_t radix_bits = Index::Backend == JitBackend::LLVM ? 11u : 8u;
        for (uint32_t shift = 0; shift < 32; shift += radix_bits) {
            uint32_t bits = std::min(radix_bits, 32u - shift), buckets = 1u << bits;
            Index digit = (ordinal >> shift) & (buckets - 1);
            dr::eval(digit, ordinal, order);
            auto *permutation = (uint32_t *) jit_malloc(Index::Backend, count * sizeof(uint32_t));
            jit_block_mkperm(Index::Backend, digit.data(), uint32_t(count), uint32_t(count),
                             buckets, permutation, nullptr);
            Index perm = Index::map_(permutation, count, true);
            ordinal = dr::gather<Index>(ordinal, perm);
            order = dr::gather<Index>(order, perm);
        }
        return UInt(order);
    } else {
        std::vector<uint32_t> order(count);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(),
            [&](uint32_t a, uint32_t b) { return cells.entry(a) < cells.entry(b); });
        return dr::load<UInt>(order.data(), count);
    }
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
      m_opacity_sh_count(opacity_sh_count), m_min(grid_min), m_max(grid_max) {
    if (resolution < 1 || resolution > 1024)
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
    uint32_t opacity_sh_count, uint32_t core_resolution)
    : m_resolution(resolution), m_core_resolution(core_resolution ? core_resolution : resolution),
      m_active_count(0), m_sh_count(sh_count), m_opacity_sh_count(opacity_sh_count),
      m_min(grid_min), m_max(grid_max) {
    if (resolution < 1 || resolution > 1024 || m_core_resolution > resolution)
        Throw("VoxelReconstruction.from_points(): invalid storage or core resolution.");
    if (sh_count < 1 || sh_count > 16 || opacity_sh_count < 1 || opacity_sh_count > 16)
        Throw("VoxelReconstruction.from_points(): SH coefficient counts must be in [1, 16].");
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(m_min[axis]) || !std::isfinite(m_max[axis]) ||
            m_max[axis] <= m_min[axis])
            Throw("VoxelReconstruction.from_points(): grid bounds must be finite and increasing.");
        m_size[axis] = (m_max[axis] - m_min[axis]) / float(resolution);
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
    const uint32_t resolution = m_resolution, sh_count = m_sh_count,
                   opacity_sh_count = m_opacity_sh_count;
    std::vector<int32_t> index(size_t(resolution) * resolution * resolution, -1);
    std::vector<uint32_t> cells(n);
    std::vector<float> centers(n * 3), scales(n * 3), rotations(n * 4),
        radiance(n * sh_count * 3, 0.f), opacity(n * opacity_sh_count, 0.f);
    constexpr float y00 = 0.28209479177387814f;
    for (size_t i = 0; i < n; ++i) {
        const auto &seed = loaded.seeds[i];
        index.at(seed.voxelIndex) = (int32_t) i;
        cells[i] = seed.voxelIndex;
        for (size_t a = 0; a < 3; ++a) {
            centers[i * 3 + a] = seed.center[a];
            scales[i * 3 + a] = std::log(std::max(seed.scale[a], 1e-8f));
            radiance[i * sh_count * 3 + a] = seed.color[a] / y00;
        }
        for (size_t a = 0; a < 4; ++a)
            rotations[i * 4 + a] = seed.rotation[a];
    }
    m_index = dr::load<IntBuffer>(index.data(), index.size());
    m_cells = dr::load<UIntBuffer>(cells.data(), cells.size());
    m_centers = dr::load<FloatBuffer>(centers.data(), centers.size());
    m_log_scales = dr::load<FloatBuffer>(scales.data(), scales.size());
    m_rotations = dr::load<FloatBuffer>(rotations.data(), rotations.size());
    m_radiance = dr::load<FloatBuffer>(radiance.data(), radiance.size());
    m_opacity = dr::load<FloatBuffer>(opacity.data(), opacity.size());
    rebuild_octree();
}

MI_VARIANT VoxelReconstruction<Float, Spectrum>::VoxelReconstruction(
    uint32_t resolution, const std::array<float, 3> &grid_min,
    const std::array<float, 3> &grid_max, const UIntBuffer &cells,
    const std::map<std::string, FloatBuffer> &parameters,
    uint32_t sh_count, uint32_t opacity_sh_count, uint32_t core_resolution)
    : m_resolution(resolution), m_core_resolution(core_resolution ? core_resolution : resolution),
      m_active_count((uint32_t) cells.size()),
      m_sh_count(sh_count), m_opacity_sh_count(opacity_sh_count),
      m_min(grid_min), m_max(grid_max) {
    if (!resolution || resolution > 1024 || m_core_resolution > resolution ||
        sh_count < 1 || sh_count > 16 || opacity_sh_count < 1 || opacity_sh_count > 16)
        Throw("VoxelReconstruction.from_state(): invalid grid, occupancy, or SH counts.");
    for (size_t a = 0; a < 3; ++a) {
        if (!std::isfinite(m_min[a]) || !std::isfinite(m_max[a]) || m_max[a] <= m_min[a])
            Throw("VoxelReconstruction.from_state(): invalid bounds.");
        m_size[a] = (m_max[a] - m_min[a]) / float(resolution);
    }
    for (const auto *key : {"centers", "log_scales", "rotations", "radiance_sh", "opacity_sh"})
        if (parameters.find(key) == parameters.end())
            Throw("VoxelReconstruction.from_state(): missing %s.", key);
    if (!m_active_count) {
        m_cells = dr::zeros<UIntBuffer>(0);
        m_index = dr::zeros<IntBuffer>(0);
        set_parameters(parameters);
        m_statistics = {{"restored_voxels", 0}};
        rebuild_octree();
        return;
    }
    const size_t total = size_t(resolution) * resolution * resolution;
    // Build occupancy in place on its backend. Atomic max resolves duplicate
    // writers deterministically; gathering back each assigned row identifies
    // every duplicate without allocating or reading a CPU dense grid.
    UIntBuffer rows = dr::arange<UIntBuffer>(m_active_count);
    auto in_bounds = cells < uint32_t(total);
    m_index = dr::full<IntBuffer>(-1, total);
    dr::scatter_reduce(ReduceOp::Max, m_index, IntBuffer(rows), cells, in_bounds);
    auto valid = in_bounds & (dr::gather<IntBuffer>(m_index, cells, in_bounds) == IntBuffer(rows));
    if (!dr::slice(dr::all(valid)))
        Throw("VoxelReconstruction.from_state(): cell indices must be unique and in bounds.");
    m_cells = dr::detach(cells);
    set_parameters(parameters);
    m_statistics = {{"restored_voxels", m_active_count}};
    rebuild_octree();
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::mark_octree(
    const UIntBuffer &cells, uint32_t bit_offset) {
    if (!cells.size() || !m_octree_levels)
        return;
    if (m_octree.size() != m_octree_nodes)
        m_octree = dr::zeros<UIntBuffer>(m_octree_nodes);
    UIntBuffer x = cells % m_resolution,
               y = (cells / m_resolution) % m_resolution,
               z = cells / (m_resolution * m_resolution);
    for (uint32_t level = 0; level < m_octree_levels; ++level) {
        uint32_t side = m_octree_resolution >> (level + 1);
        UIntBuffer px = x >> (level + 1), py = y >> (level + 1),
                   pz = z >> (level + 1);
        UIntBuffer address = m_octree_offsets[level] + (pz * side + py) * side + px;
        UIntBuffer child = ((x >> level) & 1u) |
                           (((y >> level) & 1u) << 1) |
                           (((z >> level) & 1u) << 2);
        dr::scatter_reduce(ReduceOp::Or, m_octree,
                           UIntBuffer(1u) << (child + bit_offset), address);
    }
    // Materialize once outside ray traversal/replay, without a CPU readback.
    dr::eval(m_octree);
    m_statistics["octree_allocated_bytes"] = uint64_t(m_octree.size()) * sizeof(uint32_t);
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::refresh_octree_skips() {
    if (!m_octree_levels || m_octree.size() != m_octree_nodes)
        return;
    // Propagate the largest empty ancestor top-down. Traversal needs only one
    // lookup in the finest parent level, rather than an ascent for every ray.
    for (uint32_t level = m_octree_levels; level-- > 0;) {
        uint32_t side = m_octree_resolution >> (level + 1);
        UIntBuffer index = dr::arange<UIntBuffer>(side * side * side);
        UIntBuffer address = index + m_octree_offsets[level];
        UIntBuffer bits = dr::gather<UIntBuffer>(m_octree, address) & 0xffffu;
        UIntBuffer parent_normal(0), parent_union(0);
        if (level + 1 < m_octree_levels) {
            UIntBuffer x = index % side, y = (index / side) % side,
                       z = index / (side * side);
            uint32_t parent_side = side >> 1;
            UIntBuffer parent_address = m_octree_offsets[level + 1] +
                (((z >> 1) * parent_side + (y >> 1)) * parent_side + (x >> 1));
            UIntBuffer parent = dr::gather<UIntBuffer>(m_octree, parent_address);
            parent_normal = (parent >> 16) & 0xfu;
            parent_union = (parent >> 20) & 0xfu;
        }
        UIntBuffer normal = dr::select((bits & 0xffu) == 0u,
                                      dr::maximum(parent_normal, level + 1), 0u);
        UIntBuffer combined = dr::select(bits == 0u,
                                        dr::maximum(parent_union, level + 1), 0u);
        dr::scatter(m_octree, bits | (normal << 16) | (combined << 20), address);
        dr::eval(m_octree);
    }
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::rebuild_octree() {
    m_octree_resolution = 1;
    m_octree_levels = m_octree_nodes = 0;
    while (m_octree_resolution < m_resolution)
        m_octree_resolution <<= 1;
    for (uint32_t side = m_octree_resolution >> 1; side; side >>= 1) {
        m_octree_offsets[m_octree_levels++] = m_octree_nodes;
        m_octree_nodes += side * side * side;
    }
    // Empty models keep a nonempty replay-compatible placeholder. Allocate the
    // real hierarchy lazily if hypothetical candidates are subsequently added.
    m_octree = dr::zeros<UIntBuffer>(m_active_count && m_octree_nodes ? m_octree_nodes : 1);
    mark_octree(m_cells, 0);
    refresh_octree_skips();
    ++m_topology_revision;
    m_statistics["octree_levels"] = m_octree_levels;
    m_statistics["octree_padded_resolution"] = m_octree_resolution;
    m_statistics["octree_parent_nodes"] = m_octree_nodes;
    m_statistics["octree_allocated_bytes"] = uint64_t(m_octree.size()) * sizeof(uint32_t);
}

MI_VARIANT typename VoxelReconstruction<Float, Spectrum>::Vector3f
VoxelReconstruction<Float, Spectrum>::octree_boundary_times(
    const Vector3i &cell, const Vector3i &step,
    const Point3f &origin, const Vector3f &inv_d) const {
    Vector3f result;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        Float face = Float(cell[axis]) + dr::select(step[axis] > 0, 1.f, 0.f);
        Float plane = dr::fmadd(face, m_size[axis], m_min[axis]);
        result[axis] = dr::select(step[axis] != 0,
            (plane - origin[axis]) * inv_d[axis], dr::Infinity<Float>);
    }
    return result;
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::skip_empty_block(
    Vector3i &cell, Vector3f &next, Float &entry, UInt32 &iterations,
    const Vector3i &step, const Vector3f &step_t, const Point3f &origin,
    const Vector3f &inv_d, const Float &exit,
    Mask active, bool include_growth, UInt32 &queries, bool profile) const {
    if (!m_octree_levels)
        return;
    uint32_t side = m_octree_resolution >> 1;
    UInt32 address = ((UInt32(cell.z()) >> 1) * side + (UInt32(cell.y()) >> 1)) *
                    side + (UInt32(cell.x()) >> 1);
    UInt32 node = dr::gather<UInt32>(m_octree, address, active);
    UInt32 log_size = (node >> (include_growth ? 20 : 16)) & 0xfu;
    if (profile)
        queries += dr::select(active, 1u, 0u);
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
        hi[axis] = dr::minimum(lo[axis] + size - 1, int(m_resolution) - 1);
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

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::clear_growth_candidates() {
    if (m_growth_count) {
        dr::scatter(m_index, dr::full<IntBuffer>(-1, m_growth_cells.size()), m_growth_cells);
        m_octree &= 0xffu;
        dr::eval(m_octree);
        refresh_octree_skips();
        ++m_topology_revision;
    }
    m_growth_count = 0;
    m_growth_cells = m_growth_next = dr::zeros<UIntBuffer>(1);
    m_growth_centers = m_growth_log_scales = m_growth_rotations =
        m_growth_radiance = m_growth_opacity = dr::zeros<FloatBuffer>(1);
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_growth_candidates(
    const UIntBuffer &cells, const std::map<std::string, FloatBuffer> &parameters) {
    const size_t count = cells.size();
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
    const size_t total = size_t(m_resolution)*m_resolution*m_resolution;
    // A target cell may have several parent-specific shapes and appearances.
    // Keep all original row IDs, chaining siblings in input order. Only the
    // unique cell heads are scattered into the overlay, avoiding scatter races.
    if (count && !dr::slice(dr::all(cells < uint32_t(total))))
        Throw("VoxelReconstruction.set_growth_candidates(): cells must be in bounds.");
    if (count && m_active_count) {
        auto occupied = dr::gather<IntBuffer>(m_index, cells) >= 0;
        if (dr::slice(dr::any(occupied)))
            Throw("VoxelReconstruction.set_growth_candidates(): target cells must be empty.");
    }
    // Keep independent buffer references alive even when the caller reuses its
    // dictionary. All accesses are detached, and no gradients target proposals.
    UIntBuffer new_cells = dr::zeros<UIntBuffer>(1), new_heads = dr::zeros<UIntBuffer>(1),
               new_next = dr::zeros<UIntBuffer>(1);
    if (count) {
        UIntBuffer rows = dr::arange<UIntBuffer>(count),
                   order = reconstruction_cell_order(cells),
                   ordered_cells = dr::gather<UIntBuffer>(cells, order),
                   previous_cell = dr::gather<UIntBuffer>(ordered_cells, rows - 1, rows > 0);
        auto group_start = (rows == 0) | (ordered_cells != previous_cell);
        // Scatter heads into ORIGINAL row order before compacting: even the
        // order of unique cells stays identical to the previous CPU builder.
        UIntBuffer head_flags = dr::zeros<UIntBuffer>(count);
        dr::scatter(head_flags, UIntBuffer(group_start), order);
        new_heads = dr::compress(head_flags != 0);
        new_cells = dr::gather<UIntBuffer>(cells, new_heads);
        auto has_following = rows + 1 < uint32_t(count);
        UIntBuffer following_cell = dr::gather<UIntBuffer>(ordered_cells, rows + 1, has_following),
                   following_row = dr::gather<UIntBuffer>(order, rows + 1, has_following);
        new_next = dr::empty<UIntBuffer>(count);
        dr::scatter(new_next,
                    dr::select(has_following & (following_cell == ordered_cells),
                               following_row, UINT32_MAX), order);
        dr::eval(new_cells, new_heads, new_next);
    }
    clear_growth_candidates();
    if (!count)
        return;
    if (count && !m_index.size())
        m_index = dr::full<IntBuffer>(-1, total);
    m_growth_cells = std::move(new_cells);
    m_growth_next = std::move(new_next);
    m_growth_centers = dr::detach(parameters.at("centers"));
    m_growth_log_scales = dr::detach(parameters.at("log_scales"));
    m_growth_rotations = dr::detach(parameters.at("rotations"));
    m_growth_radiance = dr::detach(parameters.at("radiance_sh"));
    m_growth_opacity = dr::detach(parameters.at("opacity_sh"));
    m_growth_count = uint32_t(count);
    dr::scatter(m_index, -2 - IntBuffer(new_heads), m_growth_cells);
    mark_octree(m_growth_cells, 8);
    refresh_octree_skips();
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
        Float near = (m_min[a] - ray.o[a]) * inv;
        Float far = (m_max[a] - ray.o[a]) * inv;
        entry = dr::maximum(entry, dr::select(parallel, -dr::Infinity<Float>, dr::minimum(near, far)));
        exit = dr::minimum(exit, dr::select(parallel, dr::Infinity<Float>, dr::maximum(near, far)));
        // Grid ownership is half-open: a parallel ray on the maximum face
        // has floor(cell) == resolution in Falcor and never enters the grid.
        active &= !parallel | ((ray.o[a] >= m_min[a]) & (ray.o[a] < m_max[a]));
        step[a] = dr::select(d[a] > 0.f, 1, dr::select(d[a] < 0.f, -1, 0));
        step_t[a] = dr::select(parallel, dr::Infinity<Float>, dr::abs(m_size[a] * inv));
    }
    if (entry_offset_fraction != 0.f) {
        active &= dr::isfinite(ray.maxt);
        entry += entry_offset_fraction * ray.maxt * length;
    }
    active &= entry <= exit;
    Point3f p = ray.o + d * dr::select(active, entry, 0.f);
    Vector3i cell;
    Vector3f next;
    for (size_t a = 0; a < 3; ++a) {
        Float grid_pos = (p[a] - m_min[a]) / m_size[a];
        grid_pos = dr::select(dr::isfinite(grid_pos), grid_pos, 0.f);
        cell[a] = Int32(dr::floor(dr::clip(grid_pos, 0.f, float(m_resolution - 1))));
        Float boundary = m_min[a] + (Float(cell[a]) + dr::select(step[a] > 0, 1.f, 0.f)) * m_size[a];
        next[a] = dr::select(step[a] == 0, dr::Infinity<Float>,
                            (boundary - ray.o[a]) / dr::select(step[a] == 0, 1.f, d[a]));
        next[a] = dr::maximum(next[a], entry);
    }
    if (use_octree)
        next = dr::maximum(entry, octree_boundary_times(cell, step, ray.o, inv_d));
    struct LoopState {
        Mask active;
        Vector3i cell;
        Vector3f next;
        Float entry, transmittance;
        Color3f color;
        UInt32 hits, iterations;
        DRJIT_STRUCT(LoopState, active, cell, next, entry, transmittance, color, hits, iterations)
    } ls{active, cell, next, entry, Float(1.f), Color3f(0.f), UInt32(0), UInt32(0)};

    dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
        [](const LoopState &s) { return s.active; },
        [&](LoopState &s) {
            UInt32 spatial = UInt32((s.cell.z() * int(m_resolution) + s.cell.y()) * int(m_resolution) + s.cell.x());
            Int32 id_signed = dr::gather<Int32>(m_index, spatial, s.active);
            Mask occupied = s.active & (id_signed >= 0);
            UInt32 id = UInt32(dr::maximum(id_signed, 0));
            if (use_octree) {
                Mask empty = s.active & !occupied;
                s = dr::if_stmt(dr::make_tuple(s, empty), empty,
                    [this, step, step_t, origin = ray.o, inv_d, exit](LoopState s, const Mask &empty) {
                        UInt32 queries(0);
                        skip_empty_block(s.cell, s.next, s.entry, s.iterations,
                                         step, step_t, origin, inv_d, exit, empty, false, queries, false);
                        return s;
                    }, [](LoopState s, const Mask &) { return s; },
                    "Voxel reconstruction octree trace skip");
            }
            Float end = dr::minimum(dr::minimum(s.next.x(), dr::minimum(s.next.y(), s.next.z())), exit);
            end = dr::maximum(end, s.entry);
            Vector3f local, scale, segment = d * (end - s.entry);
            for (size_t a = 0; a < 3; ++a) {
                Float c = dr::gather<Float>(m_centers, id * 3 + uint32_t(a), occupied);
                Float log_scale = dr::gather<Float>(m_log_scales, id * 3 + uint32_t(a), occupied);
                // Subtract the cell origin first to retain precision for small ellipsoids.
                local[a] = ray.o[a] + s.entry*d[a] -
                           (m_min[a] + (Float(s.cell[a]) + c) * m_size[a]);
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
            Mask hit = occupied & dr::select(a <= 1e-20f, c <= 0.f,
                                             (discriminant >= 0.f) & (t0 <= t1));
            Color3f emission(0.f);
            Float logit = 0.f;
            for (uint32_t k = 0; k < std::max(m_sh_count, m_opacity_sh_count); ++k) {
                Float sh = reconstruction_sh(k, Vector3f(-d));
                if (k < m_sh_count)
                    for (uint32_t channel = 0; channel < 3; ++channel)
                        emission[channel] += sh * dr::gather<Float>(m_radiance,
                            (id*m_sh_count+k)*3+channel, hit);
                if (k < m_opacity_sh_count)
                    logit += sh * dr::gather<Float>(m_opacity, id*m_opacity_sh_count+k, hit);
            }
            Float alpha = dr::select(hit, dr::rcp(1.f + dr::exp(-logit)), 0.f);
            s.color += s.transmittance * alpha * dr::maximum(emission, 0.f);
            s.transmittance *= 1.f - alpha;
            s.hits += dr::select(hit, 1u, 0u);
            // Match Falcor's x, then y, then z tie priority. Each iteration
            // crosses one face, including zero-length cells at exact edges.
            Mask cross_x = (s.next.x() <= s.next.y()) & (s.next.x() <= s.next.z());
            Mask cross_y = !cross_x & (s.next.y() <= s.next.z());
            Mask cross_z = !cross_x & !cross_y;
            s.cell.x() += dr::select(cross_x, step.x(), 0);
            s.cell.y() += dr::select(cross_y, step.y(), 0);
            s.cell.z() += dr::select(cross_z, step.z(), 0);
            if (use_octree)
                s.next = dr::maximum(end, octree_boundary_times(s.cell, step, ray.o, inv_d));
            else {
                s.next.x() = dr::select(cross_x, s.next.x() + step_t.x(), s.next.x());
                s.next.y() = dr::select(cross_y, s.next.y() + step_t.y(), s.next.y());
                s.next.z() = dr::select(cross_z, s.next.z() + step_t.z(), s.next.z());
            }
            s.entry = end;
            s.iterations += 1;
            s.active &= dr::all(s.cell >= 0) & dr::all(s.cell < int(m_resolution)) &
                        (s.entry < exit) & (s.transmittance > transmittance_threshold) &
                        (s.hits < max_hits) & (s.iterations <= 3*m_resolution+3);
        }, "Voxel reconstruction DDA");
    return {ls.color + ls.transmittance*background, 1.f-ls.transmittance, ls.hits};
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
        {"use_octree", 1.f}, {"rgb_only", 0.f},
        {"collect_rgb_deletion_evidence", 0.f},
        {"collect_growth_evidence", 0.f}, {"max_growth_hits", 8.f},
        {"distortion_loss_weight", 0.f}};
    for (const auto &[key, value] : settings) {
        if (cfg.find(key) == cfg.end() || !std::isfinite(value) || value < 0.f)
            Throw("VoxelReconstruction.loss_and_grad(): unknown/invalid setting %s.", key.c_str());
        cfg[key] = value;
    }
    const bool rgb_only = cfg["rgb_only"] != 0.f;
    const bool use_octree = cfg["use_octree"] != 0.f;
    const bool profile_traversal = cfg["profile_traversal"] != 0.f;
    const float distortion_weight = cfg["distortion_loss_weight"];
    const bool distortion_enabled = distortion_weight > 0.f;
    const bool collect_rgb_deletion = cfg["collect_rgb_deletion_evidence"] != 0.f;
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
    const uint32_t hit_fields = distortion_enabled ? 11u : 9u;
    if (width == 0 || width * std::max(max_hits, std::max(max_candidates,
        collect_growth ? max_growth_hits : 1u)) * hit_fields > UINT32_MAX)
        Throw("VoxelReconstruction.loss_and_grad(): ray batch is empty or too large.");
    ReconstructionStageProfiler<dr::is_jit_v<Float>> profiler(cfg["profile_stages"] != 0.f);
    // std::map is not traversed by dr::schedule(); schedule its buffers explicitly.
    auto evaluate_result = [](const TrainingResult &result) {
        if constexpr (dr::is_jit_v<Float>) {
            dr::schedule(result.rgb, result.raw_rgb, result.alpha, result.raw_alpha,
                         result.loss, result.rgb_loss, result.distortion_loss, result.hit_count);
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
        result.raw_alpha = dr::zeros<Float>(width);
        result.raw_rgb = background + result.raw_alpha;
        Float weight = dr::select(inverse_pdf > 0.f, inverse_pdf, 1.f);
        result.rgb = Color3f(1.f) + (result.raw_rgb - 1.f)*weight;
        result.alpha = result.raw_alpha;
        Color3f target = rgb_only ? reference_rgb :
            reference_rgb*reference_alpha + Color3f(1.f)*(1.f-reference_alpha);
        result.rgb_loss = dr::squared_norm(result.rgb-target);
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
        if (profiler.enabled()) {
            evaluate_result(result);
            profiler.finish("forward");
            profiler.zero_stage("hit_backward");
            profiler.zero_stage("candidate_backward");
            result.stage_profile = std::move(profiler.profile);
        }
        return result;
    }
    UInt32 lane(0);
    if constexpr (dr::is_jit_v<Float>) lane = dr::arange<UInt32>(width);

    // Nine hit floats: relative-world xStar[3], gmin, RGB[3], opacity, T_before.
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
    auto growth_intersects = [this, rotate](const UInt32 &id, const Vector3i &cell,
        const Float &entry, const Float &end, const Point3f &origin,
        const Vector3f &direction, const Mask &active) {
        // Identical clipped-cell intersection for both the DDA record check and
        // the later proposal scoring. Siblings share a segment, not a shape.
        Vector3f relative, scale, segment = direction*(end-entry);
        for (uint32_t a=0; a<3; ++a) {
            Float center = dr::gather<Float>(m_growth_centers,id*3+a,active);
            relative[a] = origin[a]+entry*direction[a] -
                (m_min[a]+(Float(cell[a])+center)*m_size[a]);
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
        for (uint32_t k=0; k<std::max(m_sh_count,m_opacity_sh_count); ++k) {
            Float sh = reconstruction_sh(k,view);
            if (k<m_sh_count)
                for (uint32_t a=0;a<3;++a)
                    rgb[a] += sh*dr::gather<Float>(m_radiance,(id*m_sh_count+k)*3+a,mask);
            if (k<m_opacity_sh_count)
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
        Float near=(m_min[a]-ray.o[a])*inv, far=(m_max[a]-ray.o[a])*inv;
        entry=dr::maximum(entry,dr::select(parallel,-dr::Infinity<Float>,dr::minimum(near,far)));
        exit=dr::minimum(exit,dr::select(parallel,dr::Infinity<Float>,dr::maximum(near,far)));
        active &= !parallel | ((ray.o[a]>=m_min[a]) & (ray.o[a]<m_max[a]));
        step[a]=dr::select(d[a]>0.f,1,dr::select(d[a]<0.f,-1,0));
        step_t[a]=dr::select(parallel,dr::Infinity<Float>,dr::abs(m_size[a]*inv));
    }
    if (cfg["entry_offset_fraction"]!=0.f) {
        active &= dr::isfinite(ray.maxt);
        entry += cfg["entry_offset_fraction"]*ray.maxt*direction_length;
    }
    active &= entry<=exit;
    Point3f p=ray.o+d*dr::select(active,entry,0.f);
    Vector3i cell;
    Vector3f next;
    for (uint32_t a=0;a<3;++a) {
        Float position=(p[a]-m_min[a])/m_size[a];
        position=dr::select(dr::isfinite(position),position,0.f);
        cell[a]=Int32(dr::floor(dr::clip(position,0.f,float(m_resolution-1))));
        Float boundary=m_min[a]+(Float(cell[a])+dr::select(step[a]>0,1.f,0.f))*m_size[a];
        next[a]=dr::maximum(entry,dr::select(step[a]==0,dr::Infinity<Float>,
                          (boundary-ray.o[a])/dr::select(step[a]==0,1.f,d[a])));
    }
    if (use_octree)
        next = dr::maximum(entry, octree_boundary_times(cell, step, ray.o, inv_d));
    struct State {
        Mask active;
        Vector3i cell;
        Vector3f next;
        Float entry, transmittance;
        Color3f color;
        UInt32 hits,candidates,iterations,growth_hits;
        UInt32 traversal_steps, traversal_skipped, traversal_leaves, traversal_queries;
        Float distortion_mass, distortion_moment, distortion;
        DRJIT_STRUCT(State,active,cell,next,entry,transmittance,color,hits,candidates,
                     iterations,growth_hits,traversal_steps,traversal_skipped,
                     traversal_leaves,traversal_queries,distortion_mass,distortion_moment,distortion)
    } state{active,cell,next,entry,Float(1.f),Color3f(0.f),UInt32(0),UInt32(0),
            UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),UInt32(0),
            Float(0.f),Float(0.f),Float(0.f)};
    const float diagonal=std::sqrt(dr::square(m_max[0]-m_min[0])+dr::square(m_max[1]-m_min[1])+dr::square(m_max[2]-m_min[2]));
    dr::tie(state,records)=dr::while_loop(dr::make_tuple(state,records),
        [](const State &s,const Records &) {return s.active;},
        [&](State &s,Records &r) {
            UInt32 spatial=UInt32((s.cell.z()*int(m_resolution)+s.cell.y())*int(m_resolution)+s.cell.x());
            Int32 signed_id=dr::gather<Int32>(m_index,spatial,s.active);
            Mask occupied=s.active & (signed_id>=0);
            UInt32 id=UInt32(dr::maximum(signed_id,0));
            Mask visit = occupied | (collect_growth && m_growth_count ?
                         s.active & (signed_id <= -2) : Mask(false));
            if (profile_traversal) {
                s.traversal_steps += dr::select(s.active, 1u, 0u);
                s.traversal_leaves += dr::select(visit, 1u, 0u);
            }
            if (use_octree) {
                Mask empty = s.active & !visit;
                s = dr::if_stmt(dr::make_tuple(s, empty), empty,
                    [this, step, step_t, origin = ray.o, inv_d, exit, collect_growth, profile_traversal](
                        State s, const Mask &empty) {
                        UInt32 before = s.iterations;
                        skip_empty_block(s.cell, s.next, s.entry, s.iterations,
                            step, step_t, origin, inv_d, exit, empty, collect_growth,
                            s.traversal_queries, profile_traversal);
                        if (profile_traversal)
                            s.traversal_skipped += s.iterations - before;
                        return s;
                    }, [](State s, const Mask &) { return s; },
                    "Voxel reconstruction octree training skip");
            }
            Float end=dr::maximum(s.entry,dr::minimum(dr::minimum(s.next.x(),dr::minimum(s.next.y(),s.next.z())),exit));
            if (collect_growth) {
                Mask proposal = s.active & (signed_id <= -2);
                UInt32 growth_id = UInt32(-dr::minimum(signed_id, Int32(-2)) - 2);
                auto buffers = std::make_shared<GrowthBuffers>(
                    GrowthBuffers{r.growth_values, r.growth_ids, r.growth_unknown});
                s = dr::if_stmt(dr::make_tuple(s, growth_id, spatial, end, proposal, ray.o, d), proposal,
                    [this, growth_intersects, buffers, lane, ray_count, growth_stride, max_growth_hits](
                        State s, const UInt32 &head, const UInt32 &spatial,
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
                                Mask hit = growth_intersects(id,s.cell,s.entry,end,
                                    origin,direction,valid);
                                any_hit |= hit;
                                dr::scatter_reduce(ReduceOp::Add,unknown,UInt32(1),id,hit&full);
                                id = dr::gather<UInt32>(m_growth_next,id,valid);
                            },"Voxel reconstruction growth cell intersections");
                        Mask stored = any_hit & !full;
                        UInt32 slot = dr::minimum(s.growth_hits,max_growth_hits-1)*ray_count+lane;
                        dr::scatter(buffers->ids,spatial,slot,stored);
                        for (uint32_t j=0; j<3; ++j)
                            dr::scatter(buffers->values,s.color[j],j*growth_stride+slot,stored);
                        dr::scatter(buffers->values,s.transmittance,3*growth_stride+slot,stored);
                        dr::scatter(buffers->values,s.entry,4*growth_stride+slot,stored);
                        dr::scatter(buffers->values,end,5*growth_stride+slot,stored);
                        s.growth_hits = dr::minimum(s.growth_hits+dr::select(any_hit,1u,0u),max_growth_hits);
                        return s;
                    },
                    [](State s,const UInt32 &,const UInt32 &,const Float &,const Mask &,
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
                dr::make_tuple(s,r.candidates,id,end,occupied,ray.o,d),occupied,
                [this,rotate,quaternion,appearance,ray_count,hit_stride,lane,
                 max_candidates,diagonal,hit_buffers,distortion_enabled,
                 grid_entry=entry](State s,Candidates candidates,const UInt32 &id,
                    const Float &end,const Mask &occupied,const Point3f &origin,
                    const Vector3f &direction) {
                    Vector3f relative,scale,segment=direction*(end-s.entry);
                    for(uint32_t a=0;a<3;++a) {
                        Float center=dr::gather<Float>(m_centers,id*3+a,occupied);
                        relative[a]=origin[a]+s.entry*direction[a]-(m_min[a]+(Float(s.cell[a])+center)*m_size[a]);
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
                    Float depth(0.f);
                    Mask depth_free(false);
                    if (distortion_enabled) {
                        depth = (s.entry+star*(end-s.entry)-grid_entry)/diagonal;
                        depth_free = (a>1e-20f)&(star>0.f)&(star<1.f);
                    }
                    s=dr::if_stmt(
                        dr::make_tuple(s,id,relative_star,gmin,hit,direction,depth,depth_free),hit,
                        [appearance,ray_count,hit_stride,lane,hit_buffers,distortion_enabled](State s,
                            const UInt32 &id,const Vector3f &relative_star,
                            const Float &gmin,const Mask &hit,const Vector3f &direction,
                            const Float &depth,const Mask &depth_free) {
                            Color3f emission;
                            Float opacity;
                            appearance(id,Vector3f(-direction),hit,emission,opacity);
                            UInt32 slot=s.hits*ray_count+lane;
                            dr::scatter(hit_buffers->hit_id,id,slot,hit);
                            for(uint32_t j=0;j<3;++j) {
                                dr::scatter(hit_buffers->hit,relative_star[j],j*hit_stride+slot,hit);
                                dr::scatter(hit_buffers->hit,emission[j],(4+j)*hit_stride+slot,hit);
                            }
                            dr::scatter(hit_buffers->hit,gmin,3*hit_stride+slot,hit);
                            dr::scatter(hit_buffers->hit,opacity,7*hit_stride+slot,hit);
                            dr::scatter(hit_buffers->hit,s.transmittance,8*hit_stride+slot,hit);
                            Float hit_alpha=dr::select(hit,opacity,0.f);
                            if (distortion_enabled) {
                                dr::scatter(hit_buffers->hit,depth,9*hit_stride+slot,hit);
                                dr::scatter(hit_buffers->hit,Float(depth_free),10*hit_stride+slot,hit);
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
                           const Float &,const Mask &) {
                            return s;
                        },"Voxel reconstruction training hit");
                    if(max_candidates) {
                        Mask miss=occupied & !hit & dr::isfinite(gmin) & (gmin>=0.f);
                        dr::tie(s,candidates)=dr::if_stmt(
                            dr::make_tuple(s,candidates,id,relative_star,unit_star,scale,gmin,
                                           star,end,miss),miss,
                            [this,max_candidates,diagonal](State s,Candidates candidates,
                                const UInt32 &id,const Vector3f &relative_star,
                                const Vector3f &unit_star,const Vector3f &scale,
                                const Float &gmin,const Float &star,const Float &end,
                                const Mask &miss) {
                                Float radius2=dr::squared_norm(unit_star);
                                Float metric=2.f*dr::norm(unit_star/dr::sqrt(dr::maximum(radius2,1e-30f))/scale);
                                Mask valid_distance=dr::isfinite(metric)&(metric>0.f)&(radius2>1e-12f);
                                Float distance=gmin/dr::select(valid_distance,metric,1.f);
                                Float depth=s.entry+star*(end-s.entry);
                                Float score=dr::select(valid_distance,dr::maximum(distance,0.f)/std::max(m_size[0],1e-6f)+
                                                      .05f*dr::abs(depth)/std::max(diagonal,1e-6f),1e30f);
                                Candidate incoming{relative_star,gmin,s.transmittance,score,id,s.hits};
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
            Mask cross_x=(s.next.x()<=s.next.y())&(s.next.x()<=s.next.z());
            Mask cross_y=!cross_x&(s.next.y()<=s.next.z()),cross_z=!cross_x&!cross_y;
            s.cell.x()+=dr::select(cross_x,step.x(),0);s.cell.y()+=dr::select(cross_y,step.y(),0);s.cell.z()+=dr::select(cross_z,step.z(),0);
            if (use_octree)
                s.next=dr::maximum(end,octree_boundary_times(s.cell,step,ray.o,inv_d));
            else {
                s.next.x()=dr::select(cross_x,s.next.x()+step_t.x(),s.next.x());
                s.next.y()=dr::select(cross_y,s.next.y()+step_t.y(),s.next.y());
                s.next.z()=dr::select(cross_z,s.next.z()+step_t.z(),s.next.z());
            }
            s.entry=end;s.iterations+=1;
            s.active &= dr::all(s.cell>=0)&dr::all(s.cell<int(m_resolution))&(s.entry<exit)&
                        (s.transmittance>cfg["transmittance_threshold"])&(s.hits<max_hits)&(s.iterations<=3*m_resolution+3);
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
    Color3f delta=result.rgb-target,dcolor=2.f*delta;
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
        Mask exited = !dr::all(state.cell>=0) | !dr::all(state.cell<int(m_resolution)) |
                      (state.entry>=exit);
        Mask complete = exited & (state.iterations<=3*m_resolution+3) &
                        (state.hits+1u<max_hits) &
                        (weight==1.f) & dr::all(dr::isfinite(result.raw_rgb)) &
                        dr::all(dr::isfinite(reference_rgb));
        for (uint32_t k=0; k<max_growth_hits; ++k) {
            Mask stored = k<state.growth_hits;
            UInt32 slot = k*ray_count+lane;
            UInt32 spatial = dr::gather<UInt32>(records.growth_ids,slot,stored);
            Int32 head = dr::gather<Int32>(m_index,spatial,stored);
            UInt32 id = dr::select(stored,UInt32(-head-2),UInt32(UINT32_MAX));
            Vector3i cell;
            cell.x() = Int32(spatial%m_resolution);
            cell.y() = Int32((spatial/m_resolution)%m_resolution);
            cell.z() = Int32(spatial/(m_resolution*m_resolution));
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
                    Mask hit = growth_intersects(id,cell,segment_entry,segment_end,ray.o,d,valid);
                    Color3f radiance(0.f);
                    Float logit(0.f);
                    for (uint32_t j=0; j<std::max(m_sh_count,m_opacity_sh_count); ++j) {
                        Float sh = reconstruction_sh(j,Vector3f(-d));
                        if (j<m_sh_count)
                            for (uint32_t channel=0; channel<3; ++channel)
                                radiance[channel] += sh*dr::gather<Float>(m_growth_radiance,
                                    (id*m_sh_count+j)*3+channel,hit);
                        if (j<m_opacity_sh_count)
                            logit += sh*dr::gather<Float>(m_growth_opacity,id*m_opacity_sh_count+j,hit);
                    }
                    radiance = dr::maximum(clean(radiance),0.f);
                    Float opacity = dr::rcp(1.f+dr::exp(-logit));
                    Color3f change = opacity*(before*radiance-(result.raw_rgb-prefix));
                    Float gain = -2.f*dr::dot(result.raw_rgb-reference_rgb,change)-dr::squared_norm(change);
                    Mask known = hit & complete & dr::isfinite(gain) &
                        (state.transmittance*(1.f-opacity)>cfg["transmittance_threshold"]);
                    dr::scatter_reduce(ReduceOp::Add,gains,dr::detach(gain),id,known);
                    dr::scatter_reduce(ReduceOp::Add,observed,UInt32(1),id,known);
                    dr::scatter_reduce(ReduceOp::Add,unknown,UInt32(1),id,hit&!known);
                    id = dr::gather<UInt32>(m_growth_next,id,valid);
                },"Voxel reconstruction growth cell evidence");
        }
    }
    for(const auto &[name,size]:std::map<std::string,size_t>{{"centers",size_t(m_active_count)*3},
        {"log_scales",size_t(m_active_count)*3},{"rotation_tangent",size_t(m_active_count)*3},
        {"radiance_sh",size_t(m_active_count)*m_sh_count*3},{"opacity_sh",size_t(m_active_count)*m_opacity_sh_count}})
        result.gradients[name]=dr::zeros<FloatBuffer>(size);
    for(const auto *key:{"appearance_valid","geometry_valid","background_carve_valid","foreground_evidence","background_evidence"})
        result.counters[key]=dr::zeros<UIntBuffer>(m_active_count);
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
    const float tau=std::max(cfg["geometry_tau_voxel_fraction"]*m_size[0],1e-8f);
    using GradientMap = std::map<std::string, FloatBuffer>;
    using CounterMap = std::map<std::string, UIntBuffer>;
    auto add_geometry=[this,quaternion,rotate,clean,is_background,dcolor,dalpha,weight,tau,cfg,rgb_only](
                          GradientMap &gradients,CounterMap &counters,
                          UInt32 id,const Vector3f &offset,Float gmin,Float before,
                          Float opacity,const Color3f &radiance,const Color3f &behind,
                          Float after,Mask valid,bool hard_hit) {
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
                for(uint32_t j=0;j<3;++j) corner_offset[j]=(float((corner>>j)&1u)-center[j])*m_size[j];
                contains &= dr::squared_norm(rotate(corner_offset,qw,-qv)/scale)<=1.f;
            }
        }
        if(two_sided) {
            float band=tau*cfg.at("geometry_boundary_band_tau");
            Float t=band>0.f?dr::clip(1.f-dr::abs(distance)/band,0.f,1.f):Float(0.f);
            Float taper=t*t*(3.f-2.f*t);
            dg*=hard_hit?dr::select(is_background,1.f,taper):taper;
            if(hard_hit) dg=dr::select((dg>0.f)&(is_background|contains),0.f,dg);
        } else if(transition_only) {
            if(hard_hit) {
                float band=tau*cfg.at("geometry_hit_expansion_band_tau");
                Float t=band>0.f?dr::clip(1.f+distance/band,0.f,1.f):Float(0.f);
                Float taper=dr::select(!is_background & !contains,t*t*(3.f-2.f*t),0.f);
                dg=dr::select(dg>0.f,dg*taper,dg);
            } else dg=dr::maximum(dg,0.f);
        }
        Vector3f weighted=principal/(scale*scale);
        Vector3f gc=-2.f*rotate(weighted,qw,qv),gs=-2.f*weighted*principal,gr=2.f*dr::cross(weighted,principal);
        // Apply 1/pdf after the nonlinear proxy clamp and transition gates.
        for(uint32_t j=0;j<3;++j) {
            dr::scatter_reduce(ReduceOp::Add,gradients["centers"],clean(dg*gc[j]*m_size[j])*weight,id*3+j,valid);
            dr::scatter_reduce(ReduceOp::Add,gradients["log_scales"],clean(dg*gs[j])*weight,id*3+j,valid);
            dr::scatter_reduce(ReduceOp::Add,gradients["rotation_tangent"],clean(dg*gr[j])*weight,id*3+j,valid);
        }
        dr::scatter_reduce(ReduceOp::Add,counters["geometry_valid"],UInt32(1),id,valid);
        if(!rgb_only && hard_hit) dr::scatter_reduce(ReduceOp::Add,counters["background_carve_valid"],UInt32(1),id,valid&is_background);
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
        Mask exited = !dr::all(state.cell>=0) | !dr::all(state.cell<int(m_resolution)) |
                      (state.entry>=exit);
        deletion_complete = exited & (state.iterations<=3*m_resolution+3) &
                            (weight==1.f) & dr::isfinite(inverse_pdf) &
                            dr::all(dr::isfinite(result.raw_rgb)) &
                            dr::all(dr::isfinite(reference_rgb));
        deletion_residual = result.raw_rgb-reference_rgb;
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
              dr::zeros<FloatBuffer>(size_t(hit_stride)*4),
              std::move(result.gradients),std::move(result.counters),
              std::move(result.deletion_sums),std::move(result.deletion_counts),
              Float(0.f),Float(0.f),Float(0.f)};
    dr::tie(reverse)=dr::while_loop(dr::make_tuple(std::move(reverse)),
        [](const ReverseState &s) { return s.hit>=0; },
        [this,hit_records=records.hit,hit_ids=records.hit_id,hit_count=state.hits,
         ray_count,hit_stride,lane,clean,dcolor,dalpha,d,weight,is_background,
         is_foreground,deletion_dcolor,deletion_dalpha,cfg,add_geometry,rgb_only,
         collect_rgb_deletion,deletion_complete,deletion_residual,distortion_enabled,
         distortion_weight,diagonal,quaternion,rotate,
         total_mass=state.distortion_mass,total_moment=state.distortion_moment](ReverseState &s) {
            Mask valid=UInt32(s.hit)<hit_count;
            UInt32 slot=UInt32(s.hit)*ray_count+lane,id=dr::gather<UInt32>(hit_ids,slot,valid);
            Vector3f offset;Color3f radiance;
            for(uint32_t j=0;j<3;++j) {
                offset[j]=dr::gather<Float>(hit_records,j*hit_stride+slot,valid);
                radiance[j]=dr::gather<Float>(hit_records,(4+j)*hit_stride+slot,valid);
            }
            Float gmin=dr::gather<Float>(hit_records,3*hit_stride+slot,valid);
            Float opacity=dr::gather<Float>(hit_records,7*hit_stride+slot,valid);
            Float before=dr::gather<Float>(hit_records,8*hit_stride+slot,valid);
            Color3f grad_rgb=clean(dcolor*before*opacity);
            grad_rgb=dr::select(radiance<=0.f,dr::minimum(grad_rgb,0.f),grad_rgb);
            if(!rgb_only) grad_rgb=dr::select(is_background,Color3f(0.f),grad_rgb);
            grad_rgb*=weight;
            Float grad_opacity=clean(dr::dot(dcolor,before*(radiance-s.behind))+dalpha*before*s.after);
            Float grad_logit=grad_opacity*weight*opacity*(1.f-opacity);
            Mask distortion_geometry_valid(false);
            if (distortion_enabled) {
                Float depth=dr::gather<Float>(hit_records,9*hit_stride+slot,valid);
                Mask depth_free=dr::gather<Float>(hit_records,10*hit_stride+slot,valid)>0.f;
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
                depth_derivative=dr::select(hit_count>1u,depth_derivative,0.f);

                // Interior closest point minimizes p^T diag(scale^-2) p.
                // Its implicit derivative is exact while hard-hit membership,
                // cell clipping, and DDA ordering stay fixed. Clipped endpoints
                // are fixed by the grid, so their geometric depth derivative is zero.
                Float qw;Vector3f qv,scale,log_scale;
                quaternion(id,valid,qw,qv);
                for(uint32_t j=0;j<3;++j) {
                    log_scale[j]=dr::gather<Float>(m_log_scales,id*3+j,valid);
                    scale[j]=dr::exp(dr::clip(log_scale[j],-30.f,30.f));
                }
                Vector3f principal=rotate(offset,qw,-qv), direction=rotate(d,qw,-qv);
                Vector3f weighted_point=principal/(scale*scale);
                Vector3f weighted_direction=direction/(scale*scale);
                Float denominator=dr::dot(direction,weighted_direction);
                distortion_geometry_valid=valid&depth_free&dr::isfinite(depth_derivative)&
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
                for(uint32_t j=0;j<3;++j) {
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["centers"],
                        clean(factor*center_gradient[j]*m_size[j]),id*3+j,distortion_geometry_valid);
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["log_scales"],
                        clean(factor*scale_gradient[j]),id*3+j,
                        distortion_geometry_valid&(log_scale[j]>-30.f)&(log_scale[j]<30.f));
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["rotation_tangent"],
                        clean(factor*rotation_gradient[j]),id*3+j,
                        distortion_geometry_valid&rotation_valid);
                }
                s.distortion_mass += contribution;
                s.distortion_moment += contribution*depth;
                s.distortion_weight_derivative += contribution*derivative;
            }
            for(uint32_t k=0;k<std::max(m_sh_count,m_opacity_sh_count);++k) {
                Float sh=reconstruction_sh(k,Vector3f(-d));
                if(k<m_sh_count)
                    for(uint32_t j=0;j<3;++j)
                        dr::scatter_reduce(ReduceOp::Add,s.gradients["radiance_sh"],grad_rgb[j]*sh,(id*m_sh_count+k)*3+j,valid);
                if(k<m_opacity_sh_count)
                    dr::scatter_reduce(ReduceOp::Add,s.gradients["opacity_sh"],grad_logit*sh,id*m_opacity_sh_count+k,valid);
            }
            dr::scatter_reduce(ReduceOp::Add,s.counters["appearance_valid"],UInt32(1),id,valid);
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
                Float gain = -2.f*dr::dot(deletion_residual,dc)-dr::squared_norm(dc);
                Mask known = valid & deletion_complete & dr::isfinite(before) &
                             dr::isfinite(opacity) & dr::all(dr::isfinite(radiance)) &
                             dr::all(dr::isfinite(s.behind)) &
                             dr::all(dr::isfinite(dc)) & dr::isfinite(gain);
                dr::scatter_reduce(ReduceOp::Add,s.deletion_sums["deletion_gain"],
                    dr::detach(gain),id,known);
                dr::scatter_reduce(ReduceOp::Add,s.deletion_counts["deletion_observed"],
                    UInt32(1),id,known);
                dr::scatter_reduce(ReduceOp::Add,s.deletion_counts["deletion_unknown"],
                    UInt32(1),id,valid&!known);
            }
            Mask proxy_geometry_valid=add_geometry(s.gradients,s.counters,id,offset,gmin,
                before,opacity,radiance,s.behind,s.after,valid,true);
            if (distortion_enabled)
                dr::scatter_reduce(ReduceOp::Add,s.counters["geometry_valid"],UInt32(1),
                    id,distortion_geometry_valid&!proxy_geometry_valid);
            s.behind=dr::select(valid,clean(opacity*radiance+(1.f-opacity)*s.behind),s.behind);
            s.after=dr::select(valid,dr::clip(clean(s.after*(1.f-opacity)),0.f,1.f),s.after);
            // Each ray writes each slot once; the four planes are disjoint.
            // Candidates read these suffixes only after reverse traversal ends.
            for(uint32_t j=0;j<3;++j)
                dr::scatter(s.suffix,s.behind[j],j*hit_stride+slot,valid,ReduceMode::NoConflicts);
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
        Float suffix_t=dr::select(has_suffix,dr::gather<Float>(reverse.suffix,3*hit_stride+suffix,has_suffix),1.f);
        Color3f radiance;Float opacity;
        appearance(id,Vector3f(-d),valid,radiance,opacity);
        add_geometry(result.gradients,result.counters,id,candidate.point,candidate.gmin,candidate.before,opacity,radiance,suffix_color,suffix_t,valid,false);
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
        << m_active_count << ", sh_count=" << m_sh_count << ", opacity_sh_count="
        << m_opacity_sh_count << "]";
    return out.str();
}

MI_INSTANTIATE_CLASS(VoxelReconstruction)
NAMESPACE_END(mitsuba)
