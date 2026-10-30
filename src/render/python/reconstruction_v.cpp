#include <mitsuba/render/reconstruction.h>
#include <mitsuba/python/python.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <drjit/python.h>

MI_PY_EXPORT(VoxelReconstruction) {
    MI_PY_IMPORT_TYPES()
    using Model = mitsuba::VoxelReconstruction<Float, Spectrum>;
    using PointArray = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
    // Keep device selection and stream ownership inside the Dr.Jit runtime.
    // These narrow helpers let the optional NCCL transport operate on evaluated
    // buffers without changing CUDA_VISIBLE_DEVICES or importing another tensor
    // runtime. The Python caller must retain each array until communication ends.
    m.def("_reconstruction_cuda_context", [](int device) {
        if constexpr (!dr::is_cuda_v<Float>) {
            throw nb::value_error("Reconstruction CUDA communication requires cuda_ad_rgb");
        } else {
            int count = jit_cuda_device_count();
            if (device >= count || device < -1)
                throw nb::value_error("Requested reconstruction CUDA device is unavailable");
            if (device >= 0)
                jit_cuda_set_device(device);
            nb::dict result;
            result["device_count"] = count;
            result["device"] = jit_cuda_device();
            result["raw_device"] = jit_cuda_device_raw();
            result["stream"] = (uintptr_t) jit_cuda_stream();
            result["context"] = (uintptr_t) jit_cuda_context();
            return result;
        }
    }, "device"_a = -1);
    m.def("_reconstruction_cuda_push_context", [](uintptr_t context) {
        if constexpr (!dr::is_cuda_v<Float>)
            throw nb::value_error("Reconstruction CUDA communication requires cuda_ad_rgb");
        else
            jit_cuda_push_context((void *) context);
    }, "context"_a);
    m.def("_reconstruction_cuda_pop_context", []() {
        if constexpr (!dr::is_cuda_v<Float>)
            throw nb::value_error("Reconstruction CUDA communication requires cuda_ad_rgb");
        else
            return (uintptr_t) jit_cuda_pop_context();
    });
    m.def("_reconstruction_float_buffer_info", [](typename Model::FloatBuffer &value) {
        nb::dict result;
        result["pointer"] = (uintptr_t) value.data();
        result["size"] = dr::width(value);
        result["dtype"] = "float32";
        return result;
    }, "value"_a.noconvert());
    m.def("_reconstruction_uint_buffer_info", [](typename Model::UIntBuffer &value) {
        nb::dict result;
        result["pointer"] = (uintptr_t) value.data();
        result["size"] = dr::width(value);
        result["dtype"] = "uint32";
        return result;
    }, "value"_a.noconvert());
    m.def("_reconstruction_optimizer_update_scope_end", [](const std::vector<uint64_t> &handles) {
        for (uint64_t handle : handles)
            jit_var_unstash_ref(handle);
    }, "handles"_a,
       "Release optimizer tracing reference stashes; call from a finally block.");
    auto cls = nb::class_<Model, Object>(m, "VoxelReconstruction", nb::dynamic_attr(),
        "Point-cloud voxel ellipsoids with the Falcor NoLightTransport forward model.")
        .def(nb::init<const std::string &, uint32_t, const std::array<float, 3> &,
                     const std::array<float, 3> &, double, uint32_t, uint32_t>(),
             "filename"_a, "resolution"_a, "grid_min"_a, "grid_max"_a,
             "opacity_threshold"_a = .05, "sh_count"_a = 9, "opacity_sh_count"_a = 9,
             nb::call_guard<nb::gil_scoped_release>())
        .def_static("from_points", [](const PointArray &positions, const PointArray &colors,
                 uint32_t resolution, const std::array<float, 3> &grid_min,
                 const std::array<float, 3> &grid_max, uint32_t sh_count,
                 uint32_t opacity_sh_count, uint32_t core_resolution) {
            if (!positions.size() || positions.size() % 3 || colors.size() != positions.size())
                throw nb::value_error("from_points(): expected nonempty flat XYZ and RGB arrays of equal length");
            nb::gil_scoped_release release;
            return new Model(positions.data(), colors.data(), positions.size() / 3,
                             resolution, grid_min, grid_max, sh_count, opacity_sh_count,
                             core_resolution);
        }, "positions"_a.noconvert(), "colors"_a.noconvert(), "resolution"_a,
           "grid_min"_a, "grid_max"_a, "sh_count"_a = 9, "opacity_sh_count"_a = 9,
           "core_resolution"_a = 0, nb::rv_policy::take_ownership,
           "Initialize from flat CPU float32 XYZ and sRGB arrays in the renderer frame.")
        .def_static("from_state", [](uint32_t resolution,
                 const std::array<float, 3> &grid_min,
                 const std::array<float, 3> &grid_max, const nb::dict &state,
                 uint32_t sh_count, uint32_t opacity_sh_count, uint32_t core_resolution) {
            if (!state.contains("cell_indices"))
                throw nb::value_error("from_state(): missing cell_indices");
            auto cells = nb::cast<typename Model::UIntBuffer>(state["cell_indices"]);
            std::map<std::string, typename Model::FloatBuffer> parameters;
            for (auto item : state) {
                std::string key = nb::cast<std::string>(item.first);
                if (key != "cell_indices")
                    parameters[key] = nb::cast<typename Model::FloatBuffer>(item.second);
            }
            return new Model(resolution, grid_min, grid_max, cells,
                             parameters, sh_count, opacity_sh_count, core_resolution);
        }, "resolution"_a, "grid_min"_a, "grid_max"_a, "state"_a,
           "sh_count"_a = 9, "opacity_sh_count"_a = 9, "core_resolution"_a = 0,
           nb::rv_policy::take_ownership)
        .def("set_parameters", &Model::set_parameters, "parameters"_a,
             "Replace continuous parameter buffers atomically after shape/finite validation.")
        .def("set_parameters_guarded", &Model::set_parameters_guarded,
             "parameters"_a, "allowed"_a,
             "Device-only finite transaction; returns the scalar acceptance mask for optimizer state.")
        .def("set_parameters_sparse", &Model::set_parameters_sparse,
             "appearance_indices"_a, "geometry_indices"_a, "parameters"_a, "allowed"_a,
             "Commit compact sorted appearance/geometry rows with one device acceptance gate.")
        .def("optimizer_step_device", [](Model &model,
                 const std::map<std::string, typename Model::FloatBuffer> &gradients,
                 const typename Model::UIntBuffer &appearance_valid,
                 const typename Model::UIntBuffer &geometry_valid,
                 const typename Model::UIntBuffer &background_carve_valid,
                 const nb::dict &means, const nb::dict &squared_means,
                 const nb::dict &steps, const std::map<std::string, double> &settings,
                 const typename Model::Mask &allowed, bool check_moments) {
            // Borrow the actual bound Dr.Jit objects. Casting arrays by value
            // would retain old full tables and trigger copy-on-write on scatter.
            std::map<std::string, typename Model::FloatBuffer *> mean_ptrs, square_ptrs;
            std::map<std::string, typename Model::UIntBuffer *> step_ptrs;
            for (auto item : means)
                mean_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::FloatBuffer *>(item.second);
            for (auto item : squared_means)
                square_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::FloatBuffer *>(item.second);
            for (auto item : steps)
                step_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::UIntBuffer *>(item.second);
            return model.optimizer_step_device(gradients, appearance_valid, geometry_valid,
                background_carve_valid, mean_ptrs, square_ptrs, step_ptrs, settings,
                allowed, check_moments);
        }, "gradients"_a, "appearance_valid"_a, "geometry_valid"_a,
           "background_carve_valid"_a, "means"_a, "squared_means"_a, "steps"_a,
           "settings"_a, "allowed"_a, "check_moments"_a = true,
           "Device-resident sparse Adam; mutates borrowed moment arrays and returns a shared acceptance mask.")
        .def("optimizer_step_compact_device", [](Model &model,
                 const typename Model::UIntBuffer &indices,
                 const typename Model::UInt32 &row_count,
                 const std::map<std::string, typename Model::FloatBuffer> &gradients,
                 const typename Model::UIntBuffer &appearance_valid,
                 const typename Model::UIntBuffer &geometry_valid,
                 const typename Model::UIntBuffer &background_carve_valid,
                 const nb::dict &means, const nb::dict &squared_means,
                 const nb::dict &steps, const std::map<std::string, double> &settings,
                 const typename Model::Mask &allowed, bool check_moments) {
            std::map<std::string, typename Model::FloatBuffer *> mean_ptrs, square_ptrs;
            std::map<std::string, typename Model::UIntBuffer *> step_ptrs;
            for (auto item : means)
                mean_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::FloatBuffer *>(item.second);
            for (auto item : squared_means)
                square_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::FloatBuffer *>(item.second);
            for (auto item : steps)
                step_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::UIntBuffer *>(item.second);
            return model.optimizer_step_compact_device(indices, row_count, gradients,
                appearance_valid, geometry_valid, background_carve_valid, mean_ptrs,
                square_ptrs, step_ptrs, settings, allowed, check_moments);
        }, "indices"_a, "row_count"_a, "gradients"_a, "appearance_valid"_a,
           "geometry_valid"_a, "background_carve_valid"_a, "means"_a,
           "squared_means"_a, "steps"_a, "settings"_a, "allowed"_a,
           "check_moments"_a = true,
           "Consume compact sorted gradient rows directly, updating full model and Adam storage in place.")
        .def("optimizer_update_scope_begin", [](Model &model, const nb::dict &means,
                 const nb::dict &squared_means, const nb::dict &steps) {
            std::map<std::string, typename Model::FloatBuffer *> mean_ptrs, square_ptrs;
            std::map<std::string, typename Model::UIntBuffer *> step_ptrs;
            for (auto item : means)
                mean_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::FloatBuffer *>(item.second);
            for (auto item : squared_means)
                square_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::FloatBuffer *>(item.second);
            for (auto item : steps)
                step_ptrs[nb::cast<std::string>(item.first)] =
                    nb::cast<typename Model::UIntBuffer *>(item.second);
            auto handles = model.optimizer_update_scope_begin(mean_ptrs, square_ptrs, step_ptrs);
            try {
                nb::list result;
                for (uint64_t handle : handles)
                    result.append(nb::int_(handle));
                return result;
            } catch (...) {
                // Cover allocation/conversion failure while transferring the
                // scope handles to Python, before its finally block can run.
                for (uint64_t handle : handles)
                    jit_var_unstash_ref(handle);
                throw;
            }
        }, "means"_a, "squared_means"_a, "steps"_a,
           "Detach genuine aliases and begin a sparse-optimizer recording/replay scope.")
        .def("set_growth_candidates", [](Model &model, const nb::dict &state) {
            if (!state.contains("cell_indices"))
                throw nb::value_error("set_growth_candidates(): missing cell_indices");
            auto cells = nb::cast<typename Model::UIntBuffer>(state["cell_indices"]);
            std::map<std::string, typename Model::FloatBuffer> parameters;
            for (auto item : state) {
                std::string key = nb::cast<std::string>(item.first);
                if (key != "cell_indices")
                    parameters[key] = nb::cast<typename Model::FloatBuffer>(item.second);
            }
            model.set_growth_candidates(cells, parameters);
        }, "state"_a,
           "Install frozen empty-cell candidates for online RGB evidence; does not grow the model.")
        .def("clear_growth_candidates", &Model::clear_growth_candidates)
        .def("set_scene_mapping", &Model::set_scene_mapping, "mapping"_a)
        .def("set_active_sh_count", &Model::set_active_sh_count, "count"_a,
             "Activate the first count radiance SH coefficients; preserve stored parameters.")
        .def("native_scene_mapping", &Model::native_scene_mapping)
        .def_prop_ro("growth_candidate_count", &Model::growth_candidate_count)
        .def_prop_ro("topology_revision", &Model::topology_revision)
        .def_static("dssim_image_loss_and_grad", [](const Color3f &rgb,
                 const Color3f &reference, uint32_t width, uint32_t height) {
            auto values = Model::dssim_image_loss_and_grad(rgb, reference, width, height);
            nb::dict result;
            result["loss"] = values.first;
            result["dcolor"] = values.second;
            return result;
        }, "rgb"_a, "reference"_a, "width"_a, "height"_a,
           "Per-pixel linear-RGB 1-SSIM and derivative of its pixel sum; "
           "11x11 sigma-1.5 Gaussian, zero padding, no ray traversal.")
        .def("loss_and_grad", [](const Model &model, const Ray3f &ray,
                 const Color3f &reference_rgb, const Float &reference_alpha,
                 const Color3f &background, const Float &inverse_pdf,
                 const std::map<std::string, float> &settings) {
            auto values = model.loss_and_grad(ray, reference_rgb, reference_alpha,
                                              background, inverse_pdf, settings);
            nb::dict result;
            result["rgb"] = values.rgb;
            result["alpha"] = values.alpha;
            result["raw_rgb"] = values.raw_rgb;
            result["raw_alpha"] = values.raw_alpha;
            result["hit_count"] = values.hit_count;
            result["loss"] = values.loss;
            result["rgb_loss"] = values.rgb_loss;
            result["distortion_loss"] = values.distortion_loss;
            if (values.dssim_enabled)
                result["dssim_loss"] = values.dssim_loss;
            result["gradients"] = values.gradients;
            if (values.compact_gradients) {
                result["voxel_ids"] = values.voxel_ids;
                result["voxel_count"] = values.voxel_count;
                result["compact_gradient_overflow"] = values.compact_gradient_overflow;
            }
            for (const auto &[key, value] : values.counters)
                result[nb::str(key.c_str())] = value;
            for (const auto &[key, value] : values.growth_sums)
                result[nb::str(key.c_str())] = value;
            for (const auto &[key, value] : values.growth_counts)
                result[nb::str(key.c_str())] = value;
            for (const auto &[key, value] : values.deletion_sums)
                result[nb::str(key.c_str())] = value;
            for (const auto &[key, value] : values.deletion_counts)
                result[nb::str(key.c_str())] = value;
            if (!values.stage_profile.empty())
                result["stage_profile"] = values.stage_profile;
            return result;
        }, "ray"_a, "reference_rgb"_a, "reference_alpha"_a,
           "background"_a = Color3f(1.f), "inverse_pdf"_a = Float(1.f),
           "settings"_a = std::map<std::string, float>(),
           "Explicit gradients summed over rays. With settings['rgb_only']=1, reference_rgb "
           "is the composited target, reference_alpha is ignored, and alpha supervision, "
           "background carving, legacy deletion evidence, and geometry direction gates are disabled. "
           "collect_rgb_deletion_evidence enables RGB-only removal evidence without another trace.")
        .def("trace", &Model::trace, "ray"_a, "background"_a = Color3f(0.f),
             "transmittance_threshold"_a = .01f, "max_hits"_a = 16,
             "entry_offset_fraction"_a = 0.f, "use_octree"_a = true)
        .def_prop_ro("resolution", &Model::resolution)
        .def_prop_ro("core_resolution", &Model::core_resolution)
        .def_prop_ro("active_voxel_count", &Model::active_voxel_count)
        .def_prop_ro("sh_count", &Model::sh_count)
        .def_prop_ro("active_sh_count", &Model::active_sh_count)
        .def_prop_ro("opacity_sh_count", &Model::opacity_sh_count)
        .def_prop_ro("grid_min", &Model::grid_min)
        .def_prop_ro("grid_max", &Model::grid_max)
        .def_prop_ro("voxel_size", &Model::voxel_size)
        .def("statistics", &Model::statistics)
        .def("snapshot", [](const Model &model) {
            nb::dict result;
            result["cell_indices"] = model.cell_indices();
            result["centers"] = model.centers();
            result["log_scales"] = model.log_scales();
            result["rotations"] = model.rotations();
            result["radiance_sh"] = model.radiance_sh();
            result["opacity_sh"] = model.opacity_sh();
            return result;
        }, "Return flat arrays (voxel-major, quaternion wxyz, SH coefficient then RGB). "
           "Treat these arrays as read-only.");
    cls.attr("rgb_only_version") = nb::int_(Model::rgb_only_version);
    cls.attr("rgb_growth_version") = nb::int_(Model::rgb_growth_version);
    cls.attr("rgb_deletion_version") = nb::int_(Model::rgb_deletion_version);
    cls.attr("background_rgb_deletion_version") = nb::int_(Model::background_rgb_deletion_version);
    cls.attr("distortion_version") = nb::int_(Model::distortion_version);
    cls.attr("dssim_version") = nb::int_(Model::dssim_version);
    cls.attr("sh_schedule_version") = nb::int_(Model::sh_schedule_version);
    cls.attr("projective_version") = nb::int_(Model::projective_version);
    cls.attr("octree_version") = nb::int_(Model::octree_version);
    cls.attr("optimizer_device_version") = nb::int_(Model::optimizer_device_version);
    cls.attr("compact_optimizer_version") = nb::int_(Model::compact_optimizer_version);
    cls.attr("compact_gradient_version") = nb::int_(Model::compact_gradient_version);
    cls.attr("path_record_layout_version") = nb::int_(Model::path_record_layout_version);
    drjit::bind_traverse(cls);
}
