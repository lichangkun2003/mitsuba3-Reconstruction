"""Path scratch must never expose unwritten slots, including frozen replays."""

import math

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest


_Y00 = 1 / math.sqrt(4 * math.pi)
_BACKGROUND = (.1, .15, .05)
_TARGET = (.2, .3, .25)
_DEPTHS = tuple(z for z in range(16) if z not in (2, 6))
_ALPHA = .06


def _color(z):
    return np.array([.1 + .02 * z, .3, .6 - .01 * z])


def _state(depths):
    scalar = mi.variant() == "scalar_rgb"
    floats = dr.scalar.ArrayXf if scalar else mi.Float
    cells = dr.scalar.ArrayXu64 if scalar else mi.UInt64
    n = len(depths)
    return {
        "cell_indices": cells(np.uint64(depths) * 32**2),
        "centers": floats(np.tile([.5, .5, .5], n)),
        "log_scales": floats(np.full(n * 3, math.log(.4))),
        "rotations": floats(np.tile([1., 0., 0., 0.], n)),
        "radiance_sh": floats(np.array([_color(z) / _Y00 for z in depths]).reshape(-1)),
        "opacity_sh": floats(np.full(n, math.log(_ALPHA / (1 - _ALPHA)) / _Y00)),
    }


def _model():
    model = mi.VoxelReconstruction.from_state(32, [0.] * 3, [32.] * 3,
                                             _state(_DEPTHS), 1, 1)
    model.set_growth_candidates(_state((2, 6)))
    return model


def _options(records, compact=False, distortion=0.):
    return dict(rgb_only=1., rgb_loss_channel_mean=1., max_hits=64.,
                max_candidates=8., transmittance_threshold=1e-7,
                compact_path_records=float(records),
                compact_gradient_capacity=float(len(_DEPTHS) if compact else 0),
                distortion_loss_weight=distortion, collect_deletion_evidence=0.,
                collect_rgb_deletion_evidence=1., collect_growth_evidence=1.,
                collect_adaptive_priority=1., collect_max_contribution=1.)


def _loss(model, ray, options):
    result = model.loss_and_grad(ray, mi.Color3f(*_TARGET), mi.Float(1.),
                                 mi.Color3f(*_BACKGROUND), mi.Float(1.), options)
    dr.eval(*result["gradients"].values(),
            *(value for key, value in result.items() if key != "gradients"))
    return result


def _compare(actual, expected, path="result"):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, dict):
            _compare(value, expected[key], path + "/" + key)
            continue
        a, b = np.asarray(value), np.asarray(expected[key])
        if a.dtype.kind in "biu":
            np.testing.assert_array_equal(a, b, err_msg=path + "/" + key)
        else:
            np.testing.assert_allclose(a, b, rtol=5e-5, atol=2e-6,
                                       err_msg=path + "/" + key)


def _composite(depths):
    rgb, transmission = np.zeros(3), 1.
    for z in sorted(depths):
        rgb += transmission * _ALPHA * _color(z)
        transmission *= 1 - _ALPHA
    return rgb + transmission * np.asarray(_BACKGROUND), 1 - transmission


@pytest.mark.parametrize("records", [False, True])
def test_path_scratch_prefix_and_zero_hits(variant_scalar_rgb, records):
    """Independent color/growth oracles also check buffers without CUDA freeze."""
    model = _model()
    ray = mi.Ray3f(mi.Point3f(.5, .5, -1.), mi.Vector3f(0., 0., 1.))
    ray.maxt = 80.
    result = _loss(model, ray, _options(records, distortion=.05))
    rgb, alpha = _composite(_DEPTHS)
    np.testing.assert_allclose(result["raw_rgb"], rgb, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(result["raw_alpha"], alpha, rtol=2e-6)
    assert int(result["hit_count"]) == len(_DEPTHS)
    loss = np.mean((rgb - _TARGET) ** 2)
    gains = [loss - np.mean((_composite((*_DEPTHS, z))[0] - _TARGET) ** 2)
             for z in (2, 6)]
    np.testing.assert_allclose(result["growth_gain"], gains, rtol=3e-5, atol=2e-7)
    np.testing.assert_array_equal(result["growth_observed"], [1, 1])
    np.testing.assert_array_equal(result["growth_unknown"], [0, 0])

    # No scratch entry is written by this ray; all reads must remain masked.
    ray.o = mi.Point3f(40.5, .5, -1.)
    miss = _loss(model, ray, _options(records, distortion=.05))
    np.testing.assert_allclose(miss["raw_rgb"], _BACKGROUND, atol=1e-7)
    assert int(miss["hit_count"]) == 0
    for gradient in miss["gradients"].values():
        np.testing.assert_array_equal(gradient, np.zeros(len(gradient)))
    np.testing.assert_array_equal(miss["growth_gain"], [0., 0.])


@pytest.mark.parametrize("records", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("distortion", [0., .05])
def test_path_scratch_frozen_hit_distributions(variant_cuda_ad_rgb, records,
                                              compact, distortion):
    """A replay must not consume stale scratch left by a longer previous ray."""
    model = _model()
    options = _options(records, compact, distortion)

    # Pass the model explicitly so freeze traverses all parameter/tree buffers.
    frozen = dr.freeze(lambda m, r: _loss(m, r, options), auto_opaque=False)
    distributions = (
        (np.full(16, .5), -1., len(_DEPTHS)),
        (np.full(16, 40.5), -1., 0),
        (np.tile([.5, .95, 1.5, 40.5], 4), -1., None),
        (np.full(16, .5), 10.25, 6),
        (np.full(16, 40.5), -1., 0),
        (np.full(16, .5), -1., len(_DEPTHS)),
    )
    for x, z, count in distributions:
        ray = mi.Ray3f(mi.Point3f(mi.Float(x), mi.Float(np.full(16, .5)),
                                 mi.Float(np.full(16, z))), mi.Vector3f(0., 0., 1.))
        ray.maxt = dr.opaque(mi.Float, 80.)
        replayed = frozen(model, ray)
        direct = _loss(model, ray, options)
        _compare(replayed, direct)
        hits = np.asarray(replayed["hit_count"])
        if count is not None:
            np.testing.assert_array_equal(hits, np.full(16, count))
        else:
            np.testing.assert_array_equal(hits, np.tile([len(_DEPTHS), 0, 0, 0], 4))
            # x=.95 misses ellipsoids inside occupied cells: geometry candidates
            # must still work when no hit suffix exists for those rays.
            assert np.any(np.asarray(replayed["geometry_valid"]) > 0)
        if count == 0:
            np.testing.assert_allclose(np.asarray(replayed["raw_rgb"]),
                                       np.repeat(np.asarray(_BACKGROUND)[:, None], 16, axis=1), atol=1e-7)
            for gradient in replayed["gradients"].values():
                np.testing.assert_array_equal(np.asarray(gradient), 0.)
        if compact:
            assert not bool(np.asarray(replayed["compact_gradient_overflow"])[0])
    assert frozen.n_recordings == 1
