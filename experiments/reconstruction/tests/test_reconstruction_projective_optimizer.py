"""CUDA Adam parity and preservation of inherited projective background geometry."""

from pathlib import Path
import copy
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from test_reconstruction_projective_runtime import payload


def check(mi, dr, np):
    state, metadata = payload()
    # These values arise after repeatedly splitting a covering ellipsoid.
    state["centers"][:] = [-.3, 1.4, .5, -.3, 1.4, .5]
    state["log_scales"][:] = np.log([3., 4., 5., 3., 4., 5.])
    snapshot = {"model_state": state, "model_metadata": metadata}
    models = [restore_model(copy.deepcopy(snapshot), mi) for _ in range(4)]
    optimizers = [FalcorOptimizer(model, OptimizerConfig(), sparse_updates=i != 0)
                  for i, model in enumerate(models)]
    for iteration in range(4):
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            out = {"gradients": {name: dr.full(mi.Float, 0. if iteration == 0 else .4,
                                               2*width)
                                 for name, width in optimizer.dimensions.items()},
                   "appearance_valid": mi.UInt32([1, 1]), "geometry_valid": mi.UInt32([1, 1]),
                   "background_carve_valid": mi.UInt32([0, 0])}
            if i == 2:
                out["voxel_ids"] = mi.UInt32([0, 1])
            if i == 3:
                optimizer._step_compact_reference(model, out, sample_count=1, epoch=0,
                                                 coarse_to_fine=True, update_allowed=mi.Bool(True))
            else:
                optimizer.step(model, out, update_allowed=mi.Bool(True))
        dr.sync_thread()
        for other in models[1:]:
            for name, value in models[0].snapshot().items():
                np.testing.assert_allclose(np.asarray(value), np.asarray(other.snapshot()[name]),
                                           atol=3e-7, rtol=3e-5, err_msg=name)
        if iteration == 0:
            actual = models[0].snapshot()
            np.testing.assert_allclose(np.asarray(actual["centers"])[3:], state["centers"][3:])
            np.testing.assert_allclose(np.asarray(actual["log_scales"])[3:], state["log_scales"][3:])
            np.testing.assert_allclose(np.asarray(actual["centers"])[:3], [.01, .99, .5])
            np.testing.assert_allclose(np.exp(np.asarray(actual["log_scales"])[:3]), [2., 2., 2.])
    print("projective inherited geometry + dense/device/compact/reference Adam parity: passed", flush=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="mitsuba-projective-adam-") as output:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(), Path(output), {})
        check(mi, dr, np)
