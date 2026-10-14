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
