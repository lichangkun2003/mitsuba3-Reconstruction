#include <mitsuba/render/reconstruction.h>
#include <mitsuba/python/python.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <drjit/python.h>

MI_PY_EXPORT(VoxelReconstruction) {
    MI_PY_IMPORT_TYPES()
    using Model = mitsuba::VoxelReconstruction<Float, Spectrum>;
    auto cls = nb::class_<Model, Object>(m, "VoxelReconstruction",
        "Point-cloud voxel ellipsoids with the Falcor NoLightTransport forward model.")
        .def(nb::init<const std::string &, uint32_t, const std::array<float, 3> &,
                     const std::array<float, 3> &, double, uint32_t, uint32_t>(),
             "filename"_a, "resolution"_a, "grid_min"_a, "grid_max"_a,
             "opacity_threshold"_a = .05, "sh_count"_a = 9, "opacity_sh_count"_a = 9,
             nb::call_guard<nb::gil_scoped_release>())
        .def_static("from_state", [](uint32_t resolution,
                 const std::array<float, 3> &grid_min,
                 const std::array<float, 3> &grid_max, const nb::dict &state,
                 uint32_t sh_count, uint32_t opacity_sh_count) {
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
                             parameters, sh_count, opacity_sh_count);
        }, "resolution"_a, "grid_min"_a, "grid_max"_a, "state"_a,
           "sh_count"_a = 9, "opacity_sh_count"_a = 9, nb::rv_policy::take_ownership)
        .def("set_parameters", &Model::set_parameters, "parameters"_a,
             "Replace continuous parameter buffers atomically after shape/finite validation.")
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
            result["gradients"] = values.gradients;
            for (const auto &[key, value] : values.counters)
                result[nb::str(key.c_str())] = value;
            if (!values.stage_profile.empty())
                result["stage_profile"] = values.stage_profile;
            return result;
        }, "ray"_a, "reference_rgb"_a, "reference_alpha"_a,
           "background"_a = Color3f(1.f), "inverse_pdf"_a = Float(1.f),
           "settings"_a = std::map<std::string, float>(),
           "Explicit Falcor gradients summed over rays, including boundary proxy and per-view evidence.")
        .def("trace", &Model::trace, "ray"_a, "background"_a = Color3f(0.f),
             "transmittance_threshold"_a = .01f, "max_hits"_a = 16,
             "entry_offset_fraction"_a = 0.f)
        .def_prop_ro("resolution", &Model::resolution)
        .def_prop_ro("active_voxel_count", &Model::active_voxel_count)
        .def_prop_ro("sh_count", &Model::sh_count)
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
    drjit::bind_traverse(cls);
}
