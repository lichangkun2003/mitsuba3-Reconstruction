"""Octahedral radiance: independent lookup oracle, seams, gradients and lifecycle."""

import math
import sys
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments/reconstruction"))

Y00 = 1 / math.sqrt(4 * math.pi)
BASE = np.array([.3, .4, .2], dtype=np.float32)
ALPHA = .7


def _buffer(value):
    return (dr.scalar.ArrayXf if mi.variant() == "scalar_rgb" else mi.Float)(value)


def _make(table=None, active=26):
    radiance = np.zeros((26, 3), dtype=np.float32)
    radiance[0] = BASE
    if table is not None:
        radiance[1:] = np.asarray(table).reshape(25, 3)
    cells = dr.scalar.ArrayXu64 if mi.variant() == "scalar_rgb" else mi.UInt64
    state = dict(cell_indices=cells([0]), centers=_buffer([.5] * 3),
                 log_scales=_buffer([math.log(.45)] * 3), rotations=_buffer([1., 0., 0., 0.]),
                 radiance_sh=_buffer(radiance.ravel()),
                 opacity_sh=_buffer([math.log(ALPHA / (1 - ALPHA)) / Y00]))
    model = mi.VoxelReconstruction.from_state(1, [-.5] * 3, [.5] * 3, state, 26, 1,
        radiance_representation="octahedral_scale", radiance_resolution=5)
    model.set_active_sh_count(active)
    return model, radiance


def _ray(direction):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    ray = mi.Ray3f(mi.Point3f(*(2 * d).tolist()), mi.Vector3f(*(-d).tolist()))
    ray.maxt = 4.
    return ray


def _lookup(direction, table):
    """CPU reference: interpolate positive scale, not its logarithm."""
    p = np.asarray(direction, dtype=float)
    p /= np.abs(p).sum()
    xy = p[:2] if p[2] >= 0 else (1 - np.abs(p[1::-1])) * np.where(p[:2] >= 0, 1, -1)
    tex = (xy + 1) * 2.5 - .5
    ix, iy = np.floor(tex).astype(int)
    fx, fy = tex - [ix, iy]
    result = np.zeros(3)
    for x, y, w in ((ix, iy, (1-fx)*(1-fy)), (ix+1, iy, fx*(1-fy)),
                    (ix, iy+1, (1-fx)*fy), (ix+1, iy+1, fx*fy)):
        if x < 0:
            x, y = -x-1, 4-y
        elif x >= 5:
            x, y = 9-x, 4-y
        if y < 0:
            x, y = 4-x, -y-1
        elif y >= 5:
            x, y = 4-x, 9-y
        result += w * np.exp(table[y, x])
    return result


def _loss(model, direction, compact=False):
    return model.loss_and_grad(_ray(direction), mi.Color3f(.1, .2, .15), mi.Float(1.),
        mi.Color3f(0.), mi.Float(1.), dict(rgb_only=1., rgb_loss_channel_mean=1.,
        max_hits=4., max_candidates=1., collect_deletion_evidence=0.,
        compact_path_records=float(compact)))


@pytest.mark.parametrize("variant", ["scalar_rgb", "cuda_ad_rgb"])
def test_octahedral_lookup_identity_and_seams(variant):
    mi.set_variant(variant)
    rng = np.random.default_rng(13)
    table = rng.normal(0, .4, (5, 5, 3))
    model, _ = _make(table)
    directions = list(rng.normal(size=(24, 3))) + [
        [1., 0., -2.], [-1., 0., -2.], [0., 1., -2.], [0., -1., -2.],
        [0., 0., -1.], [0., 0., 1.]]
    for direction in directions:
        rgb, alpha, hits = model.trace(_ray(direction))
        np.testing.assert_allclose(np.asarray(rgb).reshape(3),
            ALPHA * BASE * _lookup(direction, table), rtol=4e-6, atol=3e-7)
        np.testing.assert_allclose(np.asarray(alpha), ALPHA, atol=1e-6)
    # Opposite sides of each octahedral seam converge to the same color.
    for d in ([1., 0., -2.], [-1., 0., -2.], [0., 1., -2.], [0., -1., -2.]):
        axis = 0 if d[0] == 0 else 1
        a, b = np.array(d), np.array(d)
        a[axis], b[axis] = 1e-7, -1e-7
        np.testing.assert_allclose(model.trace(_ray(a))[0], model.trace(_ray(b))[0], atol=1e-6)
    corner = [model.trace(_ray([sx*1e-7, sy*1e-7, -1.]))[0]
              for sx in (-1, 1) for sy in (-1, 1)]
    for rgb in corner[1:]:
        np.testing.assert_allclose(rgb, corner[0], atol=1e-6)
    model.set_active_sh_count(1)
    for d in directions[:4]:
        np.testing.assert_allclose(np.asarray(model.trace(_ray(d))[0]).reshape(3), ALPHA*BASE, atol=1e-6)
    identity, _ = _make()
    before = identity.trace(_ray([.4, -.2, -1.]))[0]
    identity.set_active_sh_count(1)
    np.testing.assert_array_equal(before, identity.trace(_ray([.4, -.2, -1.]))[0])


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("direction", [[.4, .3, 1.], [1., 0., -2.], [0., 0., -1.]])
def test_octahedral_gradients(variant_cuda_ad_rgb, compact, direction):
    table = np.random.default_rng(5).normal(0, .25, (5, 5, 3))
    model, values = _make(table)
    result = _loss(model, direction, compact)
    analytic = np.array(result["gradients"]["radiance_sh"])
    # Only base and four gathered RGB texels may receive radiance gradients.
    assert np.count_nonzero(np.any(analytic.reshape(26, 3)[1:] != 0, axis=1)) <= 4
    for index in np.flatnonzero(analytic):
        step = 2e-3
        losses = []
        for sign in (-1, 1):
            perturbed = values.ravel().copy()
            perturbed[index] += sign * step
            model.set_parameters({"radiance_sh": mi.Float(perturbed)})
            losses.append(float(np.asarray(_loss(model, direction, compact)["loss"]).ravel()[0]))
        finite_difference = (losses[1] - losses[0]) / (2 * step)
        np.testing.assert_allclose(analytic[index], finite_difference, rtol=3e-3, atol=3e-6)
    model.set_parameters({"radiance_sh": mi.Float(values.ravel())})
    model.set_active_sh_count(1)
    np.testing.assert_array_equal(np.asarray(_loss(model, direction, compact)["gradients"]["radiance_sh"])[3:], 0.)


def test_octahedral_adam_stage_and_checkpoint(variant_cuda_ad_rgb, tmp_path):
    from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
    from reconstruction_common import model_metadata, restore_model
    from reconstruction_checkpoint import save_checkpoint, load_checkpoint
    models = [_make(active=1)[0], _make(active=1)[0]]
    cfg = OptimizerConfig(lr_center=0., lr_log_scale=0., lr_rotation=0., lr_opacity=0.)
    optimizers = [FalcorOptimizer(m, cfg, sparse_updates=sparse, replay_updates=True,
                                 radiance_degree_steps=True) for m, sparse in zip(models, [False, True])]
    direction = [.4, .1, -1.]
    for active in (1, 26, 26):
        for model, optimizer in zip(models, optimizers):
            model.set_active_sh_count(active)
            optimizer.set_learning_rates([.01, .003 if active > 1 else 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            optimizer.step(model, _loss(model, direction))
            if active == 1:
                np.testing.assert_array_equal(np.asarray(model.snapshot()["radiance_sh"])[3:], 0.)
                np.testing.assert_array_equal(optimizer.state_dict()["groups"]["radiance_sh"]["steps"], [1, 0])
        np.testing.assert_allclose(models[0].snapshot()["radiance_sh"], models[1].snapshot()["radiance_sh"],
                                   rtol=2e-6, atol=2e-7)
    model, opt = models[1], optimizers[1]
    metadata = model_metadata(model)
    state = {key: np.array(value) for key, value in model.snapshot().items()}
    path = tmp_path / "octahedral.npz"
    save_checkpoint(path, state, metadata, optimizer_state=opt.state_dict())
    checkpoint = load_checkpoint(path)
    restored = restore_model(checkpoint, mi)
    assert restored.radiance_representation == "octahedral_scale"
    assert restored.radiance_resolution == 5
    np.testing.assert_array_equal(restored.snapshot()["radiance_sh"], model.snapshot()["radiance_sh"])
    np.testing.assert_array_equal(restored.trace(_ray(direction))[0], model.trace(_ray(direction))[0])
    loaded_optimizer = FalcorOptimizer(restored, cfg)
    loaded_optimizer.load_state_dict(checkpoint["optimizer_state"])
    np.testing.assert_array_equal(loaded_optimizer.state_dict()["groups"]["radiance_sh"]["steps"], [3, 2])


def test_octahedral_replay_activation(variant_cuda_ad_rgb):
    from reconstruction_replay import NativeLossReplay
    model, values = _make(active=1)
    replay = NativeLossReplay(dr)
    options = dict(rgb_only=1., rgb_loss_channel_mean=1., max_hits=4.,
                   max_candidates=1., collect_deletion_evidence=0., compact_path_records=1.,
                   compact_gradient_capacity=1.)
    direction = [.2, -.4, -1.]
    for active, q in ((1, 0.), (1, 0.), (26, 0.), (26, .4), (26, -.2)):
        model.set_active_sh_count(active)
        values[1:] = q
        model.set_parameters({"radiance_sh": mi.Float(values.ravel())})
        inputs = (_ray(direction), mi.Color3f(.1, .2, .15), mi.Float(1.), mi.Color3f(0.), mi.Float(1.))
        actual = replay(model, *inputs, options)
        expected = model.loss_and_grad(*inputs, options)
        np.testing.assert_allclose(actual["raw_rgb"], expected["raw_rgb"], rtol=2e-6, atol=2e-7)
        for key, gradient in expected["gradients"].items():
            np.testing.assert_allclose(actual["gradients"][key], gradient, rtol=2e-5, atol=2e-6)
    assert replay.statistics()["topology_changes"] == 1
    assert replay.statistics()["replays"] >= 2


def test_octahedral_learns_directional_color(variant_cuda_ad_rgb):
    """A fixed primitive must learn different colors in two viewing directions."""
    from reconstruction_replay import NativeLossReplay
    from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
    model, _ = _make(active=1)
    cfg = OptimizerConfig(lr_center=0., lr_log_scale=0., lr_rotation=0., lr_opacity=0.)
    opt = FalcorOptimizer(model, cfg, radiance_degree_steps=True)
    replay = NativeLossReplay(dr)
    ray = mi.Ray3f(mi.Point3f(mi.Float([2., -2.]), 0., 0.),
                  mi.Vector3f(mi.Float([-1., 1.]), 0., 0.))
    ray.maxt = 4.
    target = ALPHA * mi.Color3f(mi.Float([.15, .9]), mi.Float([.25, .95]), mi.Float([.12, .8]))
    options = dict(rgb_only=1., rgb_loss_channel_mean=1., max_hits=4.,
                   max_candidates=1., collect_deletion_evidence=0., compact_path_records=1.)
    losses = []
    for epoch in range(140):
        directional = epoch >= 40
        model.set_active_sh_count(26 if directional else 1)
        opt.set_learning_rates([.01, .04 if directional else 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        result = replay(model, ray, target, mi.Float(1.), mi.Color3f(0.), mi.Float(1.), options)
        if epoch in (39, 139):
            losses.append(float(np.asarray(result["loss"])[0]))
        opt.step(model, result)
    assert losses[1] < losses[0] * .02, losses
