#include <mitsuba/render/reconstruction.h>
#include <drjit/while_loop.h>
#include "reconstruction_pointcloud.h"
#include <cmath>
#include <limits>
#include <sstream>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT VoxelReconstruction<Float, Spectrum>::VoxelReconstruction(
    const std::string &filename, uint32_t resolution,
    const std::array<float, 3> &grid_min, const std::array<float, 3> &grid_max,
    double opacity_threshold, uint32_t sh_count, uint32_t opacity_sh_count)
    : m_resolution(resolution), m_active_count(0), m_sh_count(sh_count),
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
    float transmittance_threshold, uint32_t max_hits, float entry_offset_fraction) const {
    if (!std::isfinite(transmittance_threshold) || transmittance_threshold < 0.f ||
        transmittance_threshold >= 1.f || max_hits == 0 ||
        !std::isfinite(entry_offset_fraction) || entry_offset_fraction < 0.f)
        Throw("VoxelReconstruction: invalid traversal settings.");
    Float length = dr::norm(ray.d);
    Mask active = dr::all(dr::isfinite(ray.o)) & dr::all(dr::isfinite(ray.d)) &
                  (length > 0.f) & (ray.maxt > 0.f);
    Vector3f d = ray.d / dr::select(active, length, 1.f);
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
            s.next.x() = dr::select(cross_x, s.next.x() + step_t.x(), s.next.x());
            s.next.y() = dr::select(cross_y, s.next.y() + step_t.y(), s.next.y());
            s.next.z() = dr::select(cross_z, s.next.z() + step_t.z(), s.next.z());
            s.entry = end;
            s.iterations += 1;
            s.active &= dr::all(s.cell >= 0) & dr::all(s.cell < int(m_resolution)) &
                        (s.entry < exit) & (s.transmittance > transmittance_threshold) &
                        (s.hits < max_hits) & (s.iterations <= 3*m_resolution+3);
        }, "Voxel reconstruction DDA");
    return {ls.color + ls.transmittance*background, 1.f-ls.transmittance, ls.hits};
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
