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

MI_VARIANT VoxelReconstruction<Float, Spectrum>::VoxelReconstruction(
    uint32_t resolution, const std::array<float, 3> &grid_min,
    const std::array<float, 3> &grid_max, const UIntBuffer &cells,
    const std::map<std::string, FloatBuffer> &parameters,
    uint32_t sh_count, uint32_t opacity_sh_count)
    : m_resolution(resolution), m_active_count((uint32_t) cells.size()),
      m_sh_count(sh_count), m_opacity_sh_count(opacity_sh_count),
      m_min(grid_min), m_max(grid_max) {
    if (!resolution || resolution > 1024 ||
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
        return;
    }
    std::vector<uint32_t> host_cells(m_active_count);
    dr::eval(cells);
    dr::store(host_cells.data(), cells);
    const size_t total = size_t(resolution) * resolution * resolution;
    std::vector<int32_t> index(total, -1);
    for (uint32_t i = 0; i < m_active_count; ++i) {
        uint32_t cell = host_cells[i];
        if (cell >= total || index[cell] >= 0)
            Throw("VoxelReconstruction.from_state(): cell indices must be unique and in bounds.");
        index[cell] = int32_t(i);
    }
    m_cells = dr::detach(cells);
    m_index = dr::load<IntBuffer>(index.data(), index.size());
    set_parameters(parameters);
    m_statistics = {{"restored_voxels", m_active_count}};
}

MI_VARIANT void VoxelReconstruction<Float, Spectrum>::set_parameters(
    const std::map<std::string, FloatBuffer> &parameters) {
    const std::map<std::string, size_t> sizes = {
        {"centers", size_t(m_active_count)*3}, {"log_scales", size_t(m_active_count)*3},
        {"rotations", size_t(m_active_count)*4},
        {"radiance_sh", size_t(m_active_count)*m_sh_count*3},
        {"opacity_sh", size_t(m_active_count)*m_opacity_sh_count}};
    // Validate the complete transaction before changing any parameter.
    for (const auto &[key, values] : parameters) {
        auto expected = sizes.find(key);
        if (expected == sizes.end() || values.size() != expected->second)
            Throw("VoxelReconstruction.set_parameters(): unknown key or wrong size: %s.", key.c_str());
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
        {"max_candidates", 16.f}, {"entry_offset_fraction", 0.f}};
    for (const auto &[key, value] : settings) {
        if (cfg.find(key) == cfg.end() || !std::isfinite(value) || value < 0.f)
            Throw("VoxelReconstruction.loss_and_grad(): unknown/invalid setting %s.", key.c_str());
        cfg[key] = value;
    }
    const uint32_t max_hits = uint32_t(cfg["max_hits"]), max_candidates = uint32_t(cfg["max_candidates"]);
    if (!max_hits || max_hits > 128 || cfg["max_hits"] != float(max_hits) ||
        max_candidates > 128 || cfg["max_candidates"] != float(max_candidates) ||
        cfg["transmittance_threshold"] >= 1.f || cfg["foreground_alpha_min"] > 1.f ||
        cfg["background_alpha_max"] > cfg["foreground_alpha_min"] || cfg["min_evidence_transmittance"] > 1.f)
        Throw("VoxelReconstruction.loss_and_grad(): invalid path capacity or thresholds.");
    const size_t width = std::max({dr::width(ray.o), dr::width(ray.d), dr::width(ray.maxt),
                                  dr::width(reference_rgb), dr::width(reference_alpha),
                                  dr::width(inverse_pdf), dr::width(background)});
    if (width == 0 || width * std::max(max_hits, std::max(max_candidates, 1u)) * 9 > UINT32_MAX)
        Throw("VoxelReconstruction.loss_and_grad(): ray batch is empty or too large.");
    if (!m_active_count) {
        TrainingResult result;
        result.raw_alpha = dr::zeros<Float>(width);
        result.raw_rgb = background + result.raw_alpha;
        Float weight = dr::select(inverse_pdf > 0.f, inverse_pdf, 1.f);
        result.rgb = Color3f(1.f) + (result.raw_rgb - 1.f)*weight;
        result.alpha = result.raw_alpha;
        Color3f target = reference_rgb*reference_alpha + Color3f(1.f)*(1.f-reference_alpha);
        result.loss = dr::squared_norm(result.rgb-target) +
                      cfg["alpha_loss_weight"]*dr::square(reference_alpha);
        result.hit_count = dr::zeros<UInt32>(width);
        for (const auto *key : {"centers", "log_scales", "rotation_tangent", "radiance_sh", "opacity_sh"})
            result.gradients[key] = dr::zeros<FloatBuffer>(0);
        for (const auto *key : {"appearance_valid", "geometry_valid", "background_carve_valid", "foreground_evidence", "background_evidence"})
            result.counters[key] = dr::zeros<UIntBuffer>(0);
        return result;
    }
    UInt32 lane(0);
    if constexpr (dr::is_jit_v<Float>) lane = dr::arange<UInt32>(width);

    // Nine hit floats: relative-world xStar[3], gmin, RGB[3], opacity, T_before.
    // Each field contains hit-major rows of rays: field * hit_stride +
    // hit_index * ray_count + lane. Reverse traversal therefore reads adjacent
    // lanes contiguously. The batch-size check above bounds all nine fields.
    const uint32_t ray_count = uint32_t(width), hit_stride = ray_count * max_hits;
    // Candidate slots are lane-local loop variables. A read after a scatter to
    // global memory cannot be evaluated inside a symbolic Dr.Jit loop.
    struct Candidate {
        Vector3f point;
        Float gmin, before, score;
        UInt32 id, front;
        DRJIT_STRUCT(Candidate, point, gmin, before, score, id, front)
    };
    struct Records {
        FloatBuffer hit;
        UIntBuffer hit_id;
        std::vector<Candidate> candidates;
        DRJIT_STRUCT(Records, hit, hit_id, candidates)
    } records{dr::zeros<FloatBuffer>(width*max_hits*9),
              dr::zeros<UIntBuffer>(width*max_hits),
              std::vector<Candidate>(max_candidates,
                  Candidate{Vector3f(0.f), Float(0.f), Float(0.f), Float(0.f), UInt32(0), UInt32(0)})};
    auto clean = [](const auto &value) { return dr::select(dr::isfinite(value), value, 0.f); };
    auto rotate = [](const Vector3f &v, const Float &w, const Vector3f &q) {
        return v + 2.f*dr::cross(q, dr::cross(q, v) + w*v);
    };
    auto quaternion = [&](UInt32 id, Mask mask, Float &w, Vector3f &q) {
        w = dr::gather<Float>(m_rotations, id*4, mask);
        for (uint32_t a=0; a<3; ++a) q[a] = dr::gather<Float>(m_rotations, id*4+a+1, mask);
        Float norm = dr::sqrt(w*w+dr::squared_norm(q));
        Mask valid = dr::isfinite(norm) & (norm > 1e-10f);
        w = dr::select(valid, w/dr::select(valid,norm,1.f), 1.f);
        q = dr::select(valid, q/dr::select(valid,norm,1.f), Vector3f(0.f));
    };
    auto appearance = [&](UInt32 id, const Vector3f &view, Mask mask, Color3f &rgb, Float &alpha) {
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
    struct State {
        Mask active;
        Vector3i cell;
        Vector3f next;
        Float entry, transmittance;
        Color3f color;
        UInt32 hits,candidates,iterations;
        DRJIT_STRUCT(State,active,cell,next,entry,transmittance,color,hits,candidates,iterations)
    } state{active,cell,next,entry,Float(1.f),Color3f(0.f),UInt32(0),UInt32(0),UInt32(0)};
    const float diagonal=std::sqrt(dr::square(m_max[0]-m_min[0])+dr::square(m_max[1]-m_min[1])+dr::square(m_max[2]-m_min[2]));
    dr::tie(state,records)=dr::while_loop(dr::make_tuple(state,records),
        [](const State &s,const Records &) {return s.active;},
        [&](State &s,Records &r) {
            UInt32 spatial=UInt32((s.cell.z()*int(m_resolution)+s.cell.y())*int(m_resolution)+s.cell.x());
            Int32 signed_id=dr::gather<Int32>(m_index,spatial,s.active);
            Mask occupied=s.active & (signed_id>=0);
            UInt32 id=UInt32(dr::maximum(signed_id,0));
            Float end=dr::maximum(s.entry,dr::minimum(dr::minimum(s.next.x(),dr::minimum(s.next.y(),s.next.z())),exit));
            Vector3f relative,scale,segment=d*(end-s.entry);
            for(uint32_t a=0;a<3;++a) {
                Float center=dr::gather<Float>(m_centers,id*3+a,occupied);
                relative[a]=ray.o[a]+s.entry*d[a]-(m_min[a]+(Float(s.cell[a])+center)*m_size[a]);
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
            Color3f emission;
            Float opacity;
            appearance(id,Vector3f(-d),hit,emission,opacity);
            UInt32 slot=s.hits*ray_count+lane;
            dr::scatter(r.hit_id,id,slot,hit);
            for(uint32_t j=0;j<3;++j) {
                dr::scatter(r.hit,relative_star[j],j*hit_stride+slot,hit);
                dr::scatter(r.hit,emission[j],(4+j)*hit_stride+slot,hit);
            }
            dr::scatter(r.hit,gmin,3*hit_stride+slot,hit);
            dr::scatter(r.hit,opacity,7*hit_stride+slot,hit);
            dr::scatter(r.hit,s.transmittance,8*hit_stride+slot,hit);
            if(max_candidates) {
                Mask miss=occupied & !hit & dr::isfinite(gmin) & (gmin>=0.f);
                Float radius2=dr::squared_norm(unit_star);
                Float metric=2.f*dr::norm(unit_star/dr::sqrt(dr::maximum(radius2,1e-30f))/scale);
                Mask valid_distance=dr::isfinite(metric)&(metric>0.f)&(radius2>1e-12f);
                Float distance=gmin/dr::select(valid_distance,metric,1.f);
                Float depth=s.entry+star*(end-s.entry);
                Float score=dr::select(valid_distance,dr::maximum(distance,0.f)/std::max(m_size[0],1e-6f)+
                                      .05f*dr::abs(depth)/std::max(diagonal,1e-6f),1e30f);
                Candidate incoming{relative_star,gmin,s.transmittance,score,id,s.hits};
                Mask carry=miss;
                // Stable insertion retains the same nearest candidates and tie order as Falcor.
                for(uint32_t j=0;j<max_candidates;++j) {
                    Candidate old=r.candidates[j];
                    Mask old_valid=j<s.candidates;
                    Mask replace=carry & (!old_valid | (incoming.score<old.score));
                    r.candidates[j]=dr::select(replace,incoming,old);
                    incoming=dr::select(replace,old,incoming);
                    carry &= !replace | old_valid;
                }
                s.candidates=dr::minimum(s.candidates+dr::select(miss,1u,0u),max_candidates);
            }
            Float hit_alpha=dr::select(hit,opacity,0.f);
            s.color += s.transmittance*hit_alpha*emission;
            s.transmittance *= 1.f-hit_alpha;
            s.hits += dr::select(hit,1u,0u);
            Mask cross_x=(s.next.x()<=s.next.y())&(s.next.x()<=s.next.z());
            Mask cross_y=!cross_x&(s.next.y()<=s.next.z()),cross_z=!cross_x&!cross_y;
            s.cell.x()+=dr::select(cross_x,step.x(),0);s.cell.y()+=dr::select(cross_y,step.y(),0);s.cell.z()+=dr::select(cross_z,step.z(),0);
            s.next.x()=dr::select(cross_x,s.next.x()+step_t.x(),s.next.x());
            s.next.y()=dr::select(cross_y,s.next.y()+step_t.y(),s.next.y());
            s.next.z()=dr::select(cross_z,s.next.z()+step_t.z(),s.next.z());
            s.entry=end;s.iterations+=1;
            s.active &= dr::all(s.cell>=0)&dr::all(s.cell<int(m_resolution))&(s.entry<exit)&
                        (s.transmittance>cfg["transmittance_threshold"])&(s.hits<max_hits)&(s.iterations<=3*m_resolution+3);
        },"Voxel reconstruction training paths");
    dr::eval(records,state);

    TrainingResult result;
    result.raw_rgb=state.color+state.transmittance*background;
    result.raw_alpha=1.f-state.transmittance;
    result.hit_count=state.hits;
    Float weight=dr::select(inverse_pdf>0.f,inverse_pdf,1.f);
    result.rgb=Color3f(1.f)+(result.raw_rgb-1.f)*weight;
    result.alpha=result.raw_alpha*weight;
    Color3f target=reference_rgb*reference_alpha+Color3f(1.f)*(1.f-reference_alpha);
    Color3f delta=result.rgb-target,dcolor=2.f*delta;
    Float dalpha=2.f*cfg["alpha_loss_weight"]*(result.alpha-reference_alpha);
    result.loss=dr::squared_norm(delta)+cfg["alpha_loss_weight"]*dr::square(result.alpha-reference_alpha);
    Color3f deletion_dcolor=2.f*(result.raw_rgb-target);
    Float deletion_dalpha=2.f*cfg["alpha_loss_weight"]*(result.raw_alpha-reference_alpha);
    Mask is_background=reference_alpha<=cfg["background_alpha_max"];
    Mask is_foreground=!is_background & (reference_alpha>=cfg["foreground_alpha_min"]);
    for(const auto &[name,size]:std::map<std::string,size_t>{{"centers",size_t(m_active_count)*3},
        {"log_scales",size_t(m_active_count)*3},{"rotation_tangent",size_t(m_active_count)*3},
        {"radiance_sh",size_t(m_active_count)*m_sh_count*3},{"opacity_sh",size_t(m_active_count)*m_opacity_sh_count}})
        result.gradients[name]=dr::zeros<FloatBuffer>(size);
    for(const auto *key:{"appearance_valid","geometry_valid","background_carve_valid","foreground_evidence","background_evidence"})
        result.counters[key]=dr::zeros<UIntBuffer>(m_active_count);
    const float tau=std::max(cfg["geometry_tau_voxel_fraction"]*m_size[0],1e-8f);
    auto add_geometry=[&](UInt32 id,const Vector3f &offset,Float gmin,Float before,
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
        if(!hard_hit) valid &= !is_background;
        Float distance_scale=dr::rcp(dr::select(valid,metric,1.f));
        Float distance=gmin*distance_scale,w=dr::rcp(1.f+dr::exp(distance/tau));
        Float derivative=dr::dot(dcolor,before*opacity*(radiance-behind))*(-w*(1.f-w)/tau);
        if(hard_hit) derivative=dr::select(is_background,-cfg["background_carve_weight"]*w/tau,derivative);
        derivative += cfg["alpha_geometry_weight"]*dalpha*before*opacity*after*(-w*(1.f-w)/tau);
        Float dg=dr::clip(clean(derivative*distance_scale),-cfg["geometry_grad_clamp"],cfg["geometry_grad_clamp"]);
        Mask contains(true);
        const bool two_sided=cfg["geometry_two_sided_boundary"]!=0.f;
        if(hard_hit && (two_sided || (cfg["geometry_transition_only"]!=0.f && cfg["geometry_hit_expansion_band_tau"]>0.f))) {
            for(uint32_t corner=0;corner<8;++corner) {
                Vector3f corner_offset;
                for(uint32_t j=0;j<3;++j) corner_offset[j]=(float((corner>>j)&1u)-center[j])*m_size[j];
                contains &= dr::squared_norm(rotate(corner_offset,qw,-qv)/scale)<=1.f;
            }
        }
        if(two_sided) {
            float band=tau*cfg["geometry_boundary_band_tau"];
            Float t=band>0.f?dr::clip(1.f-dr::abs(distance)/band,0.f,1.f):Float(0.f);
            Float taper=t*t*(3.f-2.f*t);
            dg*=hard_hit?dr::select(is_background,1.f,taper):taper;
            if(hard_hit) dg=dr::select((dg>0.f)&(is_background|contains),0.f,dg);
        } else if(cfg["geometry_transition_only"]!=0.f) {
            if(hard_hit) {
                float band=tau*cfg["geometry_hit_expansion_band_tau"];
                Float t=band>0.f?dr::clip(1.f+distance/band,0.f,1.f):Float(0.f);
                Float taper=dr::select(!is_background & !contains,t*t*(3.f-2.f*t),0.f);
                dg=dr::select(dg>0.f,dg*taper,dg);
            } else dg=dr::maximum(dg,0.f);
        }
        Vector3f weighted=principal/(scale*scale);
        Vector3f gc=-2.f*rotate(weighted,qw,qv),gs=-2.f*weighted*principal,gr=2.f*dr::cross(weighted,principal);
        // Apply 1/pdf after the nonlinear proxy clamp and transition gates.
        for(uint32_t j=0;j<3;++j) {
            dr::scatter_reduce(ReduceOp::Add,result.gradients["centers"],clean(dg*gc[j]*m_size[j])*weight,id*3+j,valid);
            dr::scatter_reduce(ReduceOp::Add,result.gradients["log_scales"],clean(dg*gs[j])*weight,id*3+j,valid);
            dr::scatter_reduce(ReduceOp::Add,result.gradients["rotation_tangent"],clean(dg*gr[j])*weight,id*3+j,valid);
        }
        dr::scatter_reduce(ReduceOp::Add,result.counters["geometry_valid"],UInt32(1),id,valid);
        if(hard_hit) dr::scatter_reduce(ReduceOp::Add,result.counters["background_carve_valid"],UInt32(1),id,valid&is_background);
    };

    Color3f behind=background;
    Float after(1.f);
    for(int i=int(max_hits)-1;i>=0;--i) {
        Mask valid=UInt32(i)<state.hits;
        UInt32 slot=uint32_t(i)*ray_count+lane,id=dr::gather<UInt32>(records.hit_id,slot,valid);
        Vector3f offset;Color3f radiance;
        for(uint32_t j=0;j<3;++j) {
            offset[j]=dr::gather<Float>(records.hit,j*hit_stride+slot,valid);
            radiance[j]=dr::gather<Float>(records.hit,(4+j)*hit_stride+slot,valid);
        }
        Float gmin=dr::gather<Float>(records.hit,3*hit_stride+slot,valid);
        Float opacity=dr::gather<Float>(records.hit,7*hit_stride+slot,valid);
        Float before=dr::gather<Float>(records.hit,8*hit_stride+slot,valid);
        Color3f grad_rgb=clean(dcolor*before*opacity);
        grad_rgb=dr::select(radiance<=0.f,dr::minimum(grad_rgb,0.f),grad_rgb);
        grad_rgb=dr::select(is_background,Color3f(0.f),grad_rgb)*weight;
        Float grad_opacity=clean(dr::dot(dcolor,before*(radiance-behind))+dalpha*before*after);
        Float grad_logit=grad_opacity*weight*opacity*(1.f-opacity);
        for(uint32_t k=0;k<std::max(m_sh_count,m_opacity_sh_count);++k) {
            Float sh=reconstruction_sh(k,Vector3f(-d));
            if(k<m_sh_count)
                for(uint32_t j=0;j<3;++j)
                    dr::scatter_reduce(ReduceOp::Add,result.gradients["radiance_sh"],grad_rgb[j]*sh,(id*m_sh_count+k)*3+j,valid);
            if(k<m_opacity_sh_count)
                dr::scatter_reduce(ReduceOp::Add,result.gradients["opacity_sh"],grad_logit*sh,id*m_opacity_sh_count+k,valid);
        }
        dr::scatter_reduce(ReduceOp::Add,result.counters["appearance_valid"],UInt32(1),id,valid);
        if(cfg["collect_deletion_evidence"]!=0.f) {
            Color3f dc=before*opacity*(behind-radiance);
            Float da=-before*opacity*after;
            Float delta_loss=clean(dr::dot(deletion_dcolor,dc)+dr::squared_norm(dc)+deletion_dalpha*da+cfg["alpha_loss_weight"]*da*da);
            Mask visible=valid & (before>=cfg["min_evidence_transmittance"]);
            dr::scatter_reduce(ReduceOp::Or,result.counters["foreground_evidence"],UInt32(1),id,
                visible&is_foreground&(delta_loss>cfg["min_removal_loss_delta"]));
            dr::scatter_reduce(ReduceOp::Or,result.counters["background_evidence"],UInt32(1),id,
                visible&is_background&(delta_loss<-cfg["min_removal_loss_delta"]));
        }
        add_geometry(id,offset,gmin,before,opacity,radiance,behind,after,valid,true);
        behind=dr::select(valid,clean(opacity*radiance+(1.f-opacity)*behind),behind);
        after=dr::select(valid,dr::clip(clean(after*(1.f-opacity)),0.f,1.f),after);
        // Reuse consumed record storage for the suffix composite needed by misses.
        // slot = i * ray_count + lane is unique per lane. Its four field
        // planes are disjoint, so these writes can share one kernel. The next
        // iteration's gather still evaluates them before reading this buffer.
        for(uint32_t j=0;j<3;++j)
            dr::scatter(records.hit,behind[j],(4+j)*hit_stride+slot,valid,ReduceMode::NoConflicts);
        dr::scatter(records.hit,after,7*hit_stride+slot,valid,ReduceMode::NoConflicts);
    }
    dr::eval(records.hit);
    for(uint32_t i=0;i<max_candidates;++i) {
        Mask valid=(i<state.candidates)&!is_background;
        const Candidate &candidate=records.candidates[i];
        UInt32 id=candidate.id,front=candidate.front;
        UInt32 suffix=dr::minimum(front,max_hits-1)*ray_count+lane;
        Mask has_suffix=valid&(front<state.hits);
        Color3f suffix_color;
        for(uint32_t j=0;j<3;++j)
            suffix_color[j]=dr::select(has_suffix,dr::gather<Float>(records.hit,(4+j)*hit_stride+suffix,has_suffix),background[j]);
        Float suffix_t=dr::select(has_suffix,dr::gather<Float>(records.hit,7*hit_stride+suffix,has_suffix),1.f);
        Color3f radiance;Float opacity;
        appearance(id,Vector3f(-d),valid,radiance,opacity);
        add_geometry(id,candidate.point,candidate.gmin,candidate.before,opacity,radiance,suffix_color,suffix_t,valid,false);
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
