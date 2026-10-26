"""One whole RGB view per GPU, followed by one shared mini-batch Adam step.

Parameters stay fixed while the ranks compute their different views. RGB
topology evidence is transferred and consumed independently per camera; only
gradients, validity counts and loss sums participate in the batch reduction.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import time

from reconstruction_checks import DeferredViewChecks
from reconstruction_view_batch import batch_view_seeds, iter_view_batches, zero_view_partial


def _finite_accumulator(accumulator, loss, finite, dr):
    # A finite sum is not enough: one negative view loss must not be hidden by
    # another view's positive loss. This is evaluated before and after NCCL.
    finite &= dr.all(dr.isfinite(loss)) & dr.all(loss >= 0)
    for value in accumulator["gradients"].values():
        finite &= dr.all(dr.isfinite(value))
    for key in ("rgb_loss_sum", "distortion_loss_sum", "growth_gain", "deletion_gain"):
        if key in accumulator:
            finite &= dr.all(dr.isfinite(accumulator[key]))
    return finite


def run_view_batch_epoch(*, model, config, views, order, epoch, optimizer, topology,
                         sampler, rng, mi, dr, np, reference_cache, gpu_reference_cache,
                         native_replay, collective, distributed, primary, ray_count,
                         alpha_growth_requested, deletion_evidence_collected,
                         rgb_growth_requested, rgb_deletion_enabled,
                         training_resolution, entering_resolution, processed_views,
                         global_step, optimizer_step, profile_path, performance,
                         progress_line):
    """Run the view order once and return epoch totals plus updated counters.

    RGB-only uniform 1 spp is intentional here. The legacy contour sampler has
    persistent per-camera history and continues to use the original driver.
    A reporting record represents one batch mean; its actual view count weights
    the final epoch mean, including the short final batch.
    """
    from train import compute_view_shard, kernel_profile

    if not config.training.rgb_only or getattr(sampler, "enabled", False) or alpha_growth_requested:
        raise ValueError("View-batch training requires RGB-only uniform sampling without alpha contour history")
    if config.render.spp != 1:
        raise ValueError("View-batch training requires exactly 1 spp")
    if ray_count != config.render.width * config.render.height:
        raise ValueError("View-batch ray_count must equal the complete image pixel count")
    if primary != (collective.rank == 0):
        raise ValueError("View-batch primary rank is inconsistent with the collective")
    distortion_enabled = config.gradient.distortion_loss_weight > 0
    checks = DeferredViewChecks(mi, dr, np)
    batch_sizes = []
    stage_totals = dict(reference=0., gradients=0., topology_observation=0., optimizer=0.)
    interval = config.training.profile_interval_epochs
    progress = 0
    for batch_index, indices in enumerate(iter_view_batches(order, collective.world_size)):
        batch_size = len(indices)
        seeds = batch_view_seeds(rng, batch_size)
        local_active = collective.rank < batch_size
        local_index = indices[collective.rank] if local_active else None
        batch_started = time.perf_counter()
        if local_active:
            view = views[local_index]
            rgba = reference_cache.get(view["reference_path"])
            gpu_reference = gpu_reference_cache.get(view["reference_path"], rgba)
        reference_done = time.perf_counter()
        reasons = []
        if processed_views < config.training.profile_views:
            reasons.append("initial_views")
        if interval and batch_index == 0:
            if entering_resolution:
                reasons.append("enter_resolution")
            if (epoch + 1) % interval == 0:
                reasons.append("periodic_epoch")
        profiling = bool(reasons)
        native_stages = {} if profiling and config.training.profile_stages else None
        communication_before = collective.statistics()
        with dr.kernel_history() if profiling else nullcontext() as history:
            if local_active:
                local_rng = np.random.default_rng(seeds[collective.rank])
                offsets, inverse_pdf, token = sampler.sample(view["index"], rgba, local_rng)
                if token is not None:
                    raise ValueError("View-batch sampling cannot consume alpha contour proposals")
                partial = compute_view_shard(model, view, rgba, config, offsets, inverse_pdf,
                    token, mi, dr, need_rendered_alpha=False,
                    collect_deletion_evidence=deletion_evidence_collected,
                    collect_growth_evidence=rgb_growth_requested,
                    stage_profile=native_stages, gpu_reference=gpu_reference,
                    native_replay=native_replay)
            else:
                partial = zero_view_partial(model, mi, dr)
                if distortion_enabled:
                    partial[0].update(rgb_loss_sum=mi.Float(0), distortion_loss_sum=mi.Float(0))
            accumulator, loss_sum, finite, _, _ = partial
            finite = _finite_accumulator(accumulator, loss_sum, finite, dr)
            evidence = {}
            for kind, requested, count in (
                    ("growth", rgb_growth_requested, int(getattr(model, "growth_candidate_count", 0))),
                    ("deletion", rgb_deletion_enabled and deletion_evidence_collected,
                     int(model.active_voxel_count))):
                if not requested:
                    continue
                for suffix in ("gain", "observed", "unknown"):
                    key = f"{kind}_{suffix}"
                    if local_active:
                        if key not in accumulator:
                            raise RuntimeError(f"View-batch native evidence is missing {key}")
                        evidence[key] = accumulator.pop(key)
                    else:
                        evidence[key] = dr.zeros(mi.Float if suffix == "gain" else mi.UInt32, count)
            reduced, loss_sum, finite = collective.reduce_batch(
                accumulator, loss_sum, finite, rgb_only=True,
                mode=getattr(config.training, "batch_reduction", "auto"), compact_output=True)
            finite = _finite_accumulator(reduced, loss_sum, finite, dr)
            divisor = ray_count * batch_size
            loss = loss_sum / divisor
            rgb = reduced.pop("rgb_loss_sum") / divisor if distortion_enabled else loss
            distortion = reduced.pop("distortion_loss_sum") / divisor if distortion_enabled else mi.Float(0)
            finite &= dr.isfinite(loss) & dr.isfinite(rgb) & dr.isfinite(distortion)
            checks.add(indices[0], dr.concat([loss, mi.Float(finite), rgb, distortion]))
            gradients_done = time.perf_counter()
            if evidence:
                uint_keys = tuple(key for key in evidence if not key.endswith("_gain"))
                # Every rank participates in the same broadcasts, even idle
                # tail ranks. Independent gains are never summed across views.
                for source_rank, view_index in enumerate(indices):
                    camera_evidence = collective.broadcast_arrays(evidence, root=source_rank,
                                                                   uint_keys=uint_keys)
                    if primary and rgb_growth_requested:
                        topology.observe_rgb_growth(model, camera_evidence, view_id=view_index,
                                                    epoch=epoch, pixel_count=ray_count)
                    if primary and rgb_deletion_enabled and deletion_evidence_collected:
                        topology.observe_rgb_deletion(model, camera_evidence, view_id=view_index,
                                                      epoch=epoch, pixel_count=ray_count)
            topology_done = time.perf_counter()
            updated = optimizer.step(model, reduced, sample_count=batch_size, epoch=epoch,
                                     coarse_to_fine=config.schedule.enabled, update_allowed=finite)
            checks.updated(updated["accepted"])
            host_measurement = None
            if profiling or config.training.sync_view_checks:
                host_measurement = checks.read(rgb_only=config.training.rgb_only,
                    distortion_weight=config.gradient.distortion_loss_weight)[-1]
            optimizer_done = time.perf_counter()
        batch_sizes.append(batch_size)
        stage = dict(reference=reference_done - batch_started,
                     gradients=gradients_done - reference_done,
                     topology_observation=topology_done - gradients_done,
                     optimizer=optimizer_done - topology_done)
        for name, seconds in stage.items():
            stage_totals[name] += seconds
        processed_views += batch_size
        global_step += batch_size
        optimizer_step += 1
        progress += batch_size
        if distributed is not None:
            distributed.heartbeat(f"epoch {epoch + 1} batch {batch_index + 1} views {progress}")
        loss_text = (f"batch loss={host_measurement['loss']:.6f}" if host_measurement is not None
                     else "loss at epoch end")
        progress_line.update(f"Epoch {epoch + 1}/{config.training.num_epochs} "
            f"| views {progress}/{len(views)} ({100 * progress / len(views):.0f}%) "
            f"| batch={batch_size} | {loss_text} | {optimizer_done - batch_started:.2f}s/batch")
        if profiling and primary:
            communication_after = collective.statistics()
            communication = {name: communication_after.get(name, 0) - communication_before.get(name, 0)
                             for name in ("calls", "payload_bytes", "seconds")}
            profile = dict(epoch=epoch + 1, progress=progress, camera=indices[0] + 1,
                batch_cameras=[index + 1 for index in indices], batch_size=batch_size,
                optimizer_step=optimizer_step, global_step=global_step,
                profile_scope="rank-zero full view, batch collective and one Adam update",
                training_resolution=training_resolution, active_voxels=int(model.active_voxel_count),
                sample_reasons=reasons, growth_evidence_requested=rgb_growth_requested,
                rgb_deletion_evidence_requested=rgb_deletion_enabled and deletion_evidence_collected,
                native_execution=("direct_stage_profile" if native_stages is not None else
                                  "direct_empty_model" if config.training.freeze_native and not model.active_voxel_count else
                                  "frozen" if config.training.freeze_native else "direct"),
                native_replay=native_replay.statistics(), optimizer_execution=optimizer.statistics(),
                wall_seconds=optimizer_done - batch_started, stage_seconds=stage,
                communication=communication, **host_measurement, **kernel_profile(history))
            if native_stages is not None:
                profile["native_stages"] = native_stages
                profile["native_stage_timing"] = "rank-zero view only; synchronized instrumented boundaries"
            with profile_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(profile) + "\n")
            performance.record_profile(profile)
    validation_started = time.perf_counter()
    measurements = checks.read(rgb_only=config.training.rgb_only,
                               distortion_weight=config.gradient.distortion_loss_weight)
    return dict(
        epoch_loss=sum(value["loss"] * count for value, count in zip(measurements, batch_sizes)),
        epoch_rgb_loss=sum(value["rgb_loss"] * count for value, count in zip(measurements, batch_sizes)),
        epoch_distortion_loss=sum(value["distortion_loss"] * count for value, count in zip(measurements, batch_sizes)),
        stage_totals=stage_totals, validation_seconds=time.perf_counter() - validation_started,
        processed_views=processed_views, global_step=global_step, optimizer_step=optimizer_step)
