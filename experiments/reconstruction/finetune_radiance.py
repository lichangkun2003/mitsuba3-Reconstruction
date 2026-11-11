"""Continue a saved reconstruction with radiance as the only trainable parameter.

Defaults: bonsai-11.04, 1000 additional epochs over every training camera,
one full-image 1-spp update per camera. Geometry, opacity, cameras and topology
stay fixed. Original RGB losses, Adam settings and clipping are retained.
After completion, render every train/test camera and train/0091 SH components
at 64 spp. This driver never invokes the normal training loop or topology code.
Relative paths use the workspace (parent of the Mitsuba source checkout).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preview
import reconstruction_common as common
from reconstruction_checkpoint import load_checkpoint, save_checkpoint
from reconstruction_checks import DeferredViewChecks
from reconstruction_export import checkpoint_array_sources, device_array_source
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_memory import probe_cuda_allocator
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from reconstruction_sampling import PrefetchedUniformSampler
from train import TrainConfig, update, train_view, configure_training_memory


FROZEN_FIELDS = ('cell_indices', 'centers', 'log_scales', 'rotations', 'opacity_sh')
MODE = 'radiance_only_v1'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/bonsai-11.04-E0-307min/model.npz')
    parser.add_argument('--resume', help='Resume model.npz written by this driver')
    parser.add_argument('--epochs', type=int, default=1000, help='Total additional fine-tuning epochs')
    parser.add_argument('--output-dir')
    parser.add_argument('--lr', type=float, help='Default: saved radiance learning rate')
    parser.add_argument('--reset-adam', action='store_true', help='Start fresh moments instead of continuing saved Adam')
    parser.add_argument('--save-every', type=int, default=100, help='Overwrite the new run model.npz every N epochs; 0: final only')
    parser.add_argument('--preview-every', type=int, default=100, help='Save full-SH train/0091 preview every N epochs; 0: initial/final only')
    parser.add_argument('--preview-view', type=int, default=91, help='Saved training camera index')
    parser.add_argument('--stop-after-epochs', type=int, help='Stop after N epochs this invocation, save resumable state; defer final evaluation')
    parser.add_argument('--reference-cache-mib', type=int, default=8192)
    parser.add_argument('--gpu-reference-cache-mib', type=int, default=4096)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--build-dir', default=preview.RuntimeConfig().build_dir)
    args = parser.parse_args()
    if args.epochs < 1 or (args.stop_after_epochs is not None and args.stop_after_epochs < 1):
        parser.error('epochs and stop-after-epochs must be positive')
    if min(args.save_every, args.preview_every, args.preview_view,
           args.reference_cache_mib, args.gpu_reference_cache_mib) < 0:
        parser.error('intervals, preview-view and cache budgets must be nonnegative')
    if args.lr is not None and not 0 < args.lr < float('inf'):
        parser.error('lr must be finite and positive')
    if args.resume and (args.reset_adam or args.lr is not None):
        parser.error('resume preserves the saved Adam and learning rate')
    return args


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def array_digest(source):
    digest = hashlib.sha256()
    for begin in range(0, source.size, source.max_chunk_elements):
        count = min(source.max_chunk_elements, source.size - begin)
        digest.update(memoryview(source.read_chunk(begin, count)).cast('B'))
    return digest.hexdigest()


def frozen_digests(model, dr):
    state = model.snapshot()
    return {name: array_digest(device_array_source(state[name], dr=dr)) for name in FROZEN_FIELDS}


def ensure_frozen(model, expected, expected_metadata, dr):
    if common.model_metadata(model) != expected_metadata:
        raise RuntimeError('Model topology or active SH settings changed')
    actual = frozen_digests(model, dr)
    if actual != expected:
        changed = [name for name in FROZEN_FIELDS if actual[name] != expected[name]]
        raise RuntimeError(f'Frozen parameters changed: {changed}')
    return actual


def run_final_outputs(args, output, report):
    scripts = Path(__file__).resolve().parent
    runtime = ['--build-dir', args.build_dir, '--gpu', str(args.gpu)]
    model_path = str(output / 'model.npz')
    jobs = [
        ('evaluation', [str(scripts / 'evaluate.py'), '--checkpoint', model_path,
                        '--split', 'both', '--train-max-views', '0', '--test-max-views', '0',
                        '--spp', '64', '--output-dir', str(output), *runtime]),
        ('sh_components', [str(scripts / 'render_sh_components.py'), '--checkpoint', model_path,
                           '--split', 'train', '--view', str(args.preview_view),
                           '--output-dir', str(output / 'sh-components' / f'train-{args.preview_view:05d}'),
                           *runtime]),
    ]
    for name, command in jobs:
        report[name] = {'status': 'running', 'spp': 64}
        write_json(output / 'radiance-report.json', report)
        print(f'Starting {name}: 64 spp', flush=True)
        subprocess.run([sys.executable, '-B', '-I', *command], check=True)
        report[name]['status'] = 'passed'
        write_json(output / 'radiance-report.json', report)


def run(args):
    source_path = preview.workspace_path(args.resume or args.checkpoint, preview.WORKSPACE)
    output = (preview.workspace_path(args.output_dir, preview.WORKSPACE) if args.output_dir else
              source_path.parent if args.resume else
              source_path.parent.with_name(source_path.parent.name + f'-radiance-only-{args.epochs}ep'))
    if not args.resume and output == source_path.parent:
        raise ValueError('Use a new output directory; the source experiment must be preserved')
    if args.resume:
        if output != source_path.parent:
            raise ValueError('Resume into the same fine-tuning output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'mode': MODE, 'source_checkpoint': str(source_path),
              'planned_epochs': args.epochs, 'completed_epochs': 0,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'resolved': {}}
    report_path = output / 'radiance-report.json'
    sampler = None
    try:
        print(f'Radiance-only fine-tuning: {source_path}\nOutput: {output}', flush=True)
        runtime = preview.RuntimeConfig(build_dir=args.build_dir, gpu=args.gpu)
        mi, dr, np, dll_handles = common.initialize_runtime(runtime, output, report)
        report['cuda_allocator'] = probe_cuda_allocator(dr)
        print('Loading and validating checkpoint...', flush=True)
        checkpoint = load_checkpoint(source_path)
        state = checkpoint['training_state']
        if args.resume and state.get('mode') != MODE:
            raise ValueError('--resume requires a checkpoint produced by finetune_radiance.py')
        config = TrainConfig()
        update(config, checkpoint['config'])
        original_distortion = config.gradient.distortion_loss_weight
        config.runtime = runtime
        config.render.spp = 1
        config.training.num_epochs = args.epochs
        config.training.output = str(output)
        config.training.rgb_only = True
        config.training.chunk_size = 0
        config.training.sh_degree_progression = False
        config.training.opacity_sh_degree_progression = False
        config.training.profile_views = 0
        config.training.profile_stages = False
        config.training.reference_cache_mib = args.reference_cache_mib
        config.training.gpu_reference_cache_mib = args.gpu_reference_cache_mib
        config.gradient.alpha_loss_weight = 0.
        # With fixed geometry and opacity, distortion is constant in radiance.
        config.gradient.distortion_loss_weight = 0.
        config.topology.enable_growth = config.topology.enable_deletion = False
        config.topology.use_rgb_growth = config.topology.use_growth_evidence = False
        config.topology.use_rgb_deletion = config.topology.use_low_opacity_deletion = False
        config.schedule.enabled = False
        config.optimizer = replace(config.optimizer,
            lr_radiance=config.optimizer.lr_radiance if args.lr is None else args.lr,
            lr_opacity=0., lr_center=0., lr_log_scale=0., lr_rotation=0.,
            lr_schedule_enabled=False, lr_radiance_sh=None, lr_opacity_sh=None)
        config.optimizer.validate()
        completed = int(state.get('completed_epochs', 0)) if args.resume else 0
        if args.epochs < completed:
            raise ValueError('epochs must not be smaller than completed fine-tuning epochs')
        if args.resume and (output / 'training.jsonl').exists():
            # An interrupted run can have completed/logged epochs newer than
            # its last saved weights. Discard those rows when replaying them.
            log_path = output / 'training.jsonl'
            lines = log_path.read_text(encoding='utf-8').splitlines()
            kept = [line for line in lines if int(json.loads(line)['epoch']) <= completed]
            report['discarded_unsaved_log_epochs'] = len(lines) - len(kept)
            log_path.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
        model = common.restore_model(checkpoint, mi)
        # Train every stored RGB SH coefficient from the start. Opacity stays at
        # exactly the saved activation, including its high-order coefficients.
        model.set_active_sh_count(int(model.sh_count))
        expected_metadata = common.model_metadata(model)
        print('Hashing fixed geometry, opacity and cell indices...', flush=True)
        expected = frozen_digests(model, dr)
        if args.resume and expected != state['frozen_sha256']:
            raise ValueError('Resume checkpoint violates its frozen-parameter checksums')
        initial_radiance = array_digest(device_array_source(model.snapshot()['radiance_sh'], dr=dr))
        optimizer = FalcorOptimizer(model, config.optimizer, radiance_degree_steps=True,
                                   opacity_degree_steps=True)
        if checkpoint['optimizer_state'] and not args.reset_adam:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            # load_state_dict restores the original config too; immediately
            # reinstate frozen-group rates BEFORE any step can be submitted.
            optimizer.config = config.optimizer
            report['adam_initialization'] = 'continue saved moments and clocks'
        else:
            report['adam_initialization'] = 'fresh moments and clocks'
        optimizer.set_learning_rates(None)
        assert all(getattr(optimizer.config, name) == 0 for name in
                   ('lr_opacity', 'lr_center', 'lr_log_scale', 'lr_rotation'))
        rng = np.random.default_rng(config.training.seed)
        if args.resume:
            rng.bit_generator.state = state['rng_state']
        base_checkpoint = state.get('base_checkpoint', str(source_path)) if args.resume else str(source_path)
        base_epochs = int(state.get('base_completed_epochs', 0)) if args.resume else int(state.get('completed_epochs', 0))
        global_step = int(state.get('global_step', 0))
        del checkpoint, state
        gc.collect()
        dr.flush_malloc_cache()
        dataset = preview.workspace_path(config.data.dataset, preview.WORKSPACE)
        views = common.load_views(dataset, config.data.cameras, config.render, np,
                                  data=config.data, split='train')
        if not views:
            raise ValueError('No training cameras')
        preview_view = next((v for v in views if int(v['index']) == args.preview_view), None)
        if preview_view is None:
            raise ValueError(f'No saved training camera index {args.preview_view}')
        reference_cache = common.ReferenceImageCache(config.render.width, config.render.height, mi, np,
            args.reference_cache_mib, rgb_only=True, background=config.render.background,
            reference_compositing=config.data.reference_compositing)
        gpu_cache = GPUReferenceCache(config.render.width, config.render.height, mi, dr, np,
                                      args.gpu_reference_cache_mib, rgb_only=True)
        # Reuse normal training's memory preflight/storage choice. This only
        # selects dense versus compact gradient buffers, never topology work.
        memory_config = deepcopy(config)
        memory_config.training.sh_degree_progression = optimizer.radiance_degree_steps
        memory_config.training.opacity_sh_degree_progression = optimizer.opacity_degree_steps
        configure_training_memory(memory_config, model, gpu_cache, report, len(views))
        native_replay = NativeLossReplay(dr, enabled=config.training.freeze_native)

        def guard_gradient_capacity(current_model, capacity):
            configure_training_memory(memory_config, current_model, gpu_cache, report,
                                      len(views), compact_capacity_override=capacity)

        native_replay.gradient_capacity_guard = guard_gradient_capacity
        sampler = PrefetchedUniformSampler(np, workers=config.training.sampling_workers)
        report.update(completed_epochs=completed, base_checkpoint=base_checkpoint,
            base_completed_epochs=base_epochs, train_views=len(views),
            updates_per_epoch=len(views), training_spp=1, evaluation_spp=64,
            frozen_sha256=expected, model_metadata=expected_metadata,
            initial_radiance_sha256=initial_radiance, optimizer=asdict(config.optimizer),
            distortion_note=f'Original weight {original_distortion}; omitted because it is constant in radiance.',
            preview_reference=preview_view['reference_path'])
        write_json(output / 'radiance-config.json', asdict(config))
        write_json(report_path, report)
        preview_dir = output / 'progress'
        preview_dir.mkdir(exist_ok=True)

        def render_preview(label):
            rgb, _ = common.render_view(model, config.render, preview_view['camera'], 64,
                                        42, 65536, mi, dr, np)
            common.write_png(preview_dir / f'train-{args.preview_view:04d}-{label}.png', rgb, mi, np)

        def save(completed_epochs):
            print(f'Checking frozen fields and saving epoch {completed_epochs}...', flush=True)
            ensure_frozen(model, expected, expected_metadata, dr)
            arrays = checkpoint_array_sources(model.snapshot(), dr=dr)
            moments = checkpoint_array_sources(optimizer.device_state_dict(), dr=dr)
            save_checkpoint(output / 'model.npz', arrays, expected_metadata,
                training_state={'mode': MODE, 'completed_epochs': completed_epochs,
                    'global_step': global_step, 'rng_state': rng.bit_generator.state,
                    'base_checkpoint': base_checkpoint, 'base_completed_epochs': base_epochs,
                    'frozen_sha256': expected}, optimizer_state=moments,
                config=asdict(config), provenance=report['provenance'])
            report.update(checkpoint=str(output / 'model.npz'), saved_epochs=completed_epochs,
                          frozen_parameters_verified=True)
            write_json(report_path, report)

        if not args.resume:
            render_preview('initial')
        end_epoch = min(args.epochs, completed + args.stop_after_epochs) if args.stop_after_epochs else args.epochs
        for epoch in range(completed, end_epoch):
            started = time.perf_counter()
            order = rng.permutation(len(views)) if config.training.shuffle else np.arange(len(views))
            checks = DeferredViewChecks(mi, dr, np)
            print(f'Epoch {epoch+1}/{args.epochs}: 0/{len(views)} views', flush=True)
            last_progress = time.monotonic()
            for progress, index in enumerate(order, 1):
                view = views[int(index)]
                rgba = reference_cache.get(view['reference_path'])
                gpu_reference = gpu_cache.get(view['reference_path'], rgba)
                gradients, loss, alpha = train_view(model, view, rgba, config, sampler, rng, mi, dr, np,
                    collect_deletion_evidence=False, collect_growth_evidence=False,
                    gpu_reference=gpu_reference, native_replay=native_replay, deferred_checks=checks)
                result = optimizer.step(model, gradients, sample_count=1, epoch=epoch,
                                        coarse_to_fine=False, update_allowed=gradients['_update_allowed'])
                checks.updated(result['accepted'])
                del gradients, loss, alpha, result
                global_step += 1
                if time.monotonic() - last_progress >= 30:
                    print(f'Epoch {epoch+1}/{args.epochs}: {progress}/{len(views)} views', flush=True)
                    last_progress = time.monotonic()
            measurements = checks.read(rgb_only=True, distortion_weight=0.,
                                       dssim_weight=config.gradient.dssim_loss_weight)
            record = {'epoch': epoch+1, 'seconds': time.perf_counter()-started,
                      'mean_loss': sum(x['loss'] for x in measurements)/len(views),
                      'mean_rgb_loss': sum(x['rgb_loss'] for x in measurements)/len(views),
                      'mean_dssim_loss': sum(x['dssim_loss'] for x in measurements)/len(views),
                      'global_step': global_step}
            with (output / 'training.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            report.update(completed_epochs=epoch+1, last_epoch=record)
            write_json(report_path, report)
            print(f'Epoch {epoch+1}/{args.epochs} done | RGB loss={record["mean_rgb_loss"]:.7f} '
                  f'| total={record["mean_loss"]:.7f} | {record["seconds"]:.2f}s', flush=True)
            if args.preview_every and (epoch+1) % args.preview_every == 0 and epoch+1 < end_epoch:
                render_preview(f'epoch-{epoch+1:05d}')
            if args.save_every and (epoch+1) % args.save_every == 0 and epoch+1 < end_epoch:
                save(epoch+1)
        render_preview(f'epoch-{end_epoch:05d}')
        final_radiance = array_digest(device_array_source(model.snapshot()['radiance_sh'], dr=dr))
        report.update(final_radiance_sha256=final_radiance,
                      radiance_changed=final_radiance != initial_radiance)
        if end_epoch > completed and not report['radiance_changed']:
            raise RuntimeError('Radiance did not change during fine-tuning')
        save(end_epoch)
        report['status'] = 'stopped' if end_epoch < args.epochs else 'rendering'
        write_json(report_path, report)
        sampler.close()
        sampler = None
        reference_cache.clear()
        gpu_cache.clear()
        native_replay.clear()
        optimizer.clear_replay()
        # Release every model/Adam/reference alias before launching the existing
        # evaluation tools in separate processes. They get a clean CUDA runtime.
        render_preview = save = guard_gradient_capacity = model = optimizer = native_replay = None
        rgba = gpu_reference = checks = None
        gc.collect()
        dr.sync_thread()
        dr.flush_malloc_cache()
        if end_epoch < args.epochs:
            print(f'Stopped at epoch {end_epoch}; resume with --resume "{output / "model.npz"}"', flush=True)
            return 0
        run_final_outputs(args, output, report)
        report['status'] = 'passed'
        write_json(report_path, report)
        print(f'Completed {args.epochs} radiance-only epochs and all 64-spp renders: {output}', flush=True)
        return 0
    except BaseException:
        report.update(status='interrupted' if isinstance(sys.exc_info()[1], KeyboardInterrupt) else 'failed',
                      error=traceback.format_exc())
        write_json(report_path, report)
        raise
    finally:
        if sampler is not None:
            sampler.close()


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
