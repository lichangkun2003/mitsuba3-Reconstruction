"""Render DC, higher-order RGB SH, and full RGB SH at one saved camera (64 spp).

Geometry and opacity SH stay unchanged. Higher-order output zeros RGB DC;
it is rendered through the normal renderer, including its color activation.
Relative paths are resolved from the workspace, as in evaluate.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preview
import reconstruction_common as common
from reconstruction_checkpoint import load_checkpoint
from evaluate import checkpoint_data_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='outputs/bonsai-11.04-E0-307min/model.npz')
    parser.add_argument('--dataset', help='Override the dataset saved in the checkpoint')
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--cameras', help='Override camera JSON filename')
    parser.add_argument('--view', type=int, default=0,
                        help='Saved camera index (e.g. 91 for train/0091.png), not split list position')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk-size', type=int, default=65536)
    parser.add_argument('--output-dir')
    parser.add_argument('--build-dir', default=preview.RuntimeConfig().build_dir)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.view < 0 or args.seed < 0 or args.chunk_size < 1:
        parser.error('view/seed must be nonnegative and chunk-size positive')
    checkpoint_path = preview.workspace_path(args.checkpoint, preview.WORKSPACE)
    output = (preview.workspace_path(args.output_dir, preview.WORKSPACE) if args.output_dir else
              checkpoint_path.parent / 'sh-components' / f'{args.split}-{args.view:05d}')
    output.mkdir(parents=True, exist_ok=True)
    report = {'spp': 64, 'seed': args.seed, 'split': args.split, 'view': args.view,
              'checkpoint': str(checkpoint_path), 'resolved': {}, 'components': {}}
    runtime = preview.RuntimeConfig(build_dir=args.build_dir, gpu=args.gpu)
    mi, dr, np, dll_handles = common.initialize_runtime(runtime, output, report)
    checkpoint = load_checkpoint(checkpoint_path)
    saved = checkpoint['config']
    render = preview.RenderConfig(**saved.get('render', {}))
    render.spp = 64
    dataset = preview.workspace_path(args.dataset or saved['data']['dataset'], preview.WORKSPACE)
    cameras = args.cameras or (saved.get('data', {}).get('cameras', 'transforms_train.json')
                              if args.split == 'train' else 'transforms_test.json')
    views = common.load_views(dataset, cameras, render, np,
                              data=checkpoint_data_config(saved), split=args.split)
    matching = [view for view in views if int(view['index']) == args.view]
    if len(matching) != 1:
        raise ValueError(f'Expected one camera with saved index {args.view} in {args.split}; '
                         f'found {len(matching)}. Available indices: '
                         f'{[int(view["index"]) for view in views]}')
    view = matching[0]
    model = common.restore_model(checkpoint, mi)
    sh_count = int(model.sh_count)
    original = checkpoint['model_state']['radiance_sh'].reshape(-1, sh_count, 3)
    report.update(render=asdict(render), camera=view['camera'], dataset=str(dataset),
                  source_view_index=int(view['index']), reference_path=view['reference_path'],
                  sh_count=sh_count, active_sh_count=int(model.active_sh_count),
                  semantics='Only RGB SH changes; geometry and all opacity SH are preserved. '
                            'Higher-order zeros RGB DC; normal renderer activation still applies. '
                            'All modes use identical rays. Linear images are retained in renders.npz.')
    images = {}
    for name in ('dc', 'high_order', 'full'):
        coefficients = original.copy()
        if name == 'dc':
            coefficients[:, 1:, :] = 0
        elif name == 'high_order':
            coefficients[:, 0, :] = 0
        model.set_parameters({'radiance_sh': mi.Float(coefficients.reshape(-1))})
        started = time.perf_counter()
        print(f'Rendering {name}: {args.split} view {args.view}, 64 spp', flush=True)
        rgb, alpha = common.render_view(model, render, view['camera'], 64, args.seed,
                                        args.chunk_size, mi, dr, np)
        images[name] = rgb
        if 'alpha' in images and not np.array_equal(images['alpha'], alpha):
            raise RuntimeError('Opacity changed between RGB SH components')
        images['alpha'] = alpha
        common.write_png(output / f'{name}.png', rgb, mi, np)
        report['components'][name] = {'seconds': time.perf_counter() - started,
                                      'min': float(rgb.min()), 'max': float(rgb.max())}
    np.savez_compressed(output / 'renders.npz', **images)
    common.write_png(output / 'comparison.png',
                     np.concatenate([images[k] for k in ('dc', 'high_order', 'full')], axis=1), mi, np)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Saved DC | higher-order | full comparison: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
