"""Bounded FP64 projection of adaptive leaf boxes over all training cameras.

Only detached projection scores are computed here. The CPU camera preparation,
eight-corner bounds, visibility tests, near clipping and final float32 scores
are shared with the original selector; no camera or corner is subsampled.
"""

from __future__ import annotations


_PAIR_SCRATCH_BYTES = 2048  # Conservative allowance for FP64 intermediates/reduction.
_FIXED_SCRATCH_BYTES = 1024 * 1024
_MAX_WORKSPACE_BYTES = 512 * 1024 * 1024


def projection_batch_layout(leaf_count, camera_count, *, max_pairs=65536,
                            camera_batch_size=16, workspace_limit_bytes=256 * 1024 * 1024):
    """Plan fixed-width batches without materializing leaf_count * camera_count."""
    values = (leaf_count, camera_count, max_pairs, camera_batch_size, workspace_limit_bytes)
    if any(type(value) is not int for value in values):
        raise ValueError("Projection batch dimensions must be integers")
    if leaf_count < 0 or camera_count < 0 or max_pairs < 1 or camera_batch_size < 1:
        raise ValueError("Invalid projection batch dimensions")
    if not _FIXED_SCRATCH_BYTES < workspace_limit_bytes < _MAX_WORKSPACE_BYTES:
        raise ValueError("Projection workspace must be positive and below 512 MiB")
    # All cameras are small persistent input arrays; leaves are uploaded by batch.
    fixed_bytes = _FIXED_SCRATCH_BYTES + camera_count * 19 * 8
    capacity = min(max_pairs, (workspace_limit_bytes-fixed_bytes)//_PAIR_SCRATCH_BYTES)
    if capacity < 1:
        raise ValueError("Projection camera inputs exceed the workspace budget")
    cameras = min(max(camera_count, 1), camera_batch_size, capacity)
    leaves = min(max(leaf_count, 1), capacity//cameras)
    pairs = leaves*cameras
    return dict(leaf_batch_size=leaves, camera_batch_size=cameras, pair_capacity=pairs,
                estimated_workspace_bytes=fixed_bytes+pairs*_PAIR_SCRATCH_BYTES,
                workspace_limit_bytes=workspace_limit_bytes)


class GPUBoxProjector:
    """Callable GPU equivalent of projected_boxes_pixels, with bounded scratch."""

    def __init__(self, dr, np, *, max_pairs=65536, camera_batch_size=16,
                 workspace_limit_bytes=256 * 1024 * 1024):
        self.dr, self.np = dr, np
        self.options = dict(max_pairs=max_pairs, camera_batch_size=camera_batch_size,
                            workspace_limit_bytes=workspace_limit_bytes)
        projection_batch_layout(0, 0, **self.options)
        self._camera_host = self._camera_device = None
        self.camera_uploads = 0
        self.last_plan = None
        self.last_batch_count = 0

    def _cameras(self, cameras):
        np, dr = self.np, self.dr
        table = np.asarray([
            [float(value) for key in ("position", "right", "up", "forward")
             for value in camera[key]] +
            [float(camera[key]) for key in ("fx", "fy", "cx", "cy", "width", "height", "near_clip")]
            for camera in cameras], dtype=np.float64).reshape(-1, 19)
        if self._camera_host is None or not np.array_equal(table, self._camera_host):
            self._camera_device = tuple(dr.cuda.Float64(np.ascontiguousarray(table[:, axis]))
                                        for axis in range(19))
            dr.eval(*self._camera_device)
            self._camera_host = table.copy()
            self.camera_uploads += 1
        return self._camera_device

    def __call__(self, lower, upper, cameras):
        np, dr = self.np, self.dr
        low, high = np.asarray(lower), np.asarray(upper)
        if low.ndim != 2 or low.shape[1] != 3 or high.shape != low.shape:
            raise ValueError("Projection bounds must have equal (leaf_count, 3) shapes")
        count, camera_count = len(low), len(cameras)
        scores = np.zeros(count, np.float32)
        self.last_plan = projection_batch_layout(count, camera_count, **self.options)
        self.last_batch_count = 0
        if not count or not camera_count:
            return scores
        camera_arrays = self._cameras(cameras)
        # Disable approximate fast-math transformations in the FP64 projection.
        # The final cast happens only after all camera maxima.
        with dr.scoped_set_flag(dr.JitFlag.FastMath, False):
            return self._project(low, high, camera_arrays, camera_count, scores)

    def _project(self, low, high, camera_arrays, camera_count, scores):
        np, dr = self.np, self.dr
        U, D, F = dr.cuda.UInt32, dr.cuda.Float64, dr.cuda.Float32
        plan = self.last_plan
        leaf_width, camera_width = plan["leaf_batch_size"], plan["camera_batch_size"]
        lanes = dr.arange(U, plan["pair_capacity"])
        # Offsets and widths are runtime scalars, so batches/tails reuse kernels.
        camera_stride = dr.opaque(U, camera_width)
        leaf_ids, camera_lanes = lanes//camera_stride, lanes % camera_stride
        camera_total = dr.opaque(U, camera_count)
        for start in range(0, len(low), leaf_width):
            size = min(leaf_width, len(low)-start)
            center = (low[start:start+size]+high[start:start+size])*.5
            half = (high[start:start+size]-low[start:start+size])*.5
            # Fixed-size upload buffers also keep the final partial batch from
            # introducing a new gather/input width into the JIT cache key.
            host = np.zeros((6, leaf_width), np.float64)
            host[:3, :size], host[3:, :size] = center.T, half.T
            bounds = tuple(D(np.ascontiguousarray(row)) for row in host)
            dr.eval(*bounds)
            leaf_valid = leaf_ids < dr.opaque(U, size)
            center = tuple(dr.gather(D, bounds[axis], leaf_ids) for axis in range(3))
            half = tuple(dr.gather(D, bounds[axis+3], leaf_ids) for axis in range(3))
            current = dr.zeros(D, leaf_width)
            for camera_start in range(0, camera_count, camera_width):
                camera_ids = camera_lanes+dr.opaque(U, camera_start)
                valid = leaf_valid & (camera_ids < camera_total)
                camera = tuple(dr.gather(D, value, camera_ids, valid) for value in camera_arrays)
                relative = tuple(center[axis]-camera[axis] for axis in range(3))

                def dot(vector, offset):
                    return ((vector[0]*camera[offset]+vector[1]*camera[offset+1])
                            + vector[2]*camera[offset+2])

                x, y, z = dot(relative, 3), dot(relative, 6), dot(relative, 9)
                min_x = min_y = D(float("inf"))
                max_x = max_y = max_z = D(-float("inf"))
                for octant in range(8):
                    offset = tuple(half[axis] if octant & (1 << axis) else -half[axis]
                                   for axis in range(3))
                    depth = z+dot(offset, 9)
                    # Invalid padded camera lanes use denominator 1, avoiding
                    # unnecessary NaNs before their visibility mask is applied.
                    divisor = dr.select(valid, dr.maximum(depth, camera[18]), 1.)
                    px = camera[14]+camera[12]*(x+dot(offset, 3))/divisor
                    py = camera[15]-camera[13]*(y+dot(offset, 6))/divisor
                    min_x, max_x = dr.minimum(min_x, px), dr.maximum(max_x, px)
                    min_y, max_y = dr.minimum(min_y, py), dr.maximum(max_y, py)
                    max_z = dr.maximum(max_z, depth)
                visible = (valid & (max_z > camera[18]) & (max_x >= 0.) & (min_x < camera[16])
                           & (max_y >= 0.) & (min_y < camera[17]))
                span = dr.minimum(dr.maximum(max_x-min_x, max_y-min_y),
                                  4.*dr.maximum(camera[16], camera[17]))
                per_leaf = dr.block_reduce(dr.ReduceOp.Max, dr.select(visible, span, 0.), camera_width)
                current = dr.maximum(current, per_leaf)
                # Materialize each bounded camera batch before building another;
                # otherwise lazy expressions could retain every camera at once.
                dr.eval(current)
                self.last_batch_count += 1
            output = F(current)
            dr.eval(output)
            scores[start:start+size] = np.asarray(output)[:size]
        return scores


def make_adaptive_projector(dr, np, **options):
    return GPUBoxProjector(dr, np, **options)
