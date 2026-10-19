"""Device-resident evidence for the frozen seven-point growth proposals.

Projection uses float64, like the NumPy geometry oracle. Alpha subtraction
retains the input float32 precision. Only a topology boundary or checkpoint
needs to materialize the accumulated counters; observing a view never reads
an image or a proposal counter back to the host.
"""

from __future__ import annotations

import math


class GrowthEvidenceDevice:
    """Cache one pending proposal batch and accumulate unique-view votes.

    ``pending`` remains the owner of the CPU checkpoint representation. Its
    counters are refreshed by :meth:`materialize`, while ``seen_views`` is
    updated immediately. The caller must discard this object when replacing
    the proposal batch (including checkpoint loads and resolution changes).
    """

    def __init__(self, world_points, pending, mi, dr, np):
        self.mi, self.dr, self.np = mi, dr, np
        self.pending = pending
        self.Float = dr.detached_t(mi.Float)
        self.Float64 = dr.float64_array_t(self.Float)
        self.UInt32 = mi.UInt32
        self.count = len(pending["foreground"])
        if len(world_points) != 7 or len(pending["background"]) != self.count:
            raise ValueError("Growth evidence requires seven samples and matching counters")
        self.points = []
        for world in world_points:
            world = np.asarray(world, dtype=np.float64)
            if world.shape != (self.count, 3):
                raise ValueError("Growth world samples must have shape (proposal_count, 3)")
            self.points.append(tuple(self.Float64(np.ascontiguousarray(world[:, axis]))
                                     for axis in range(3)))
        self.foreground = self.UInt32(np.asarray(pending["foreground"], dtype=np.uint32))
        self.background = self.UInt32(np.asarray(pending["background"], dtype=np.uint32))
        dr.eval(*self.points, self.foreground, self.background)
        self.materializations = 0

    def _reference_plane(self, value, width, height):
        dr, np = self.dr, self.np
        if isinstance(value, np.ndarray):
            if value.shape != (height, width) or value.dtype != np.dtype(np.float32):
                raise ValueError("Growth reference alpha must be a float32 HxW image")
            value = self.Float(np.ascontiguousarray(value).reshape(-1))
        elif not dr.is_array_v(type(value)):
            raise ValueError("Growth reference alpha must be a GPU plane or float32 HxW image")
        if dr.width(value) != width * height:
            raise ValueError("Growth reference alpha size differs from the image dimensions")
        return self.Float(dr.detach(value))

    def observe(self, camera, reference_alpha, rendered_alpha, width, height,
                view_id, settings):
        """Queue one view's projection and votes on the current CUDA stream."""
        if view_id in self.pending["seen_views"]:
            return
        if (isinstance(width, bool) or isinstance(height, bool)
                or not isinstance(width, int) or not isinstance(height, int)
                or width < 1 or height < 1 or width * height > 2**32 - 1):
            raise ValueError("Growth evidence needs positive UInt32-indexable image dimensions")
        mi, dr, np = self.mi, self.dr, self.np
        reference = self._reference_plane(reference_alpha, width, height)
        if not dr.is_array_v(type(rendered_alpha)) or dr.width(rendered_alpha) != width * height:
            raise ValueError("Growth rendered alpha must be a GPU plane matching the image dimensions")
        rendered = self.Float(dr.detach(rendered_alpha))
        if self.count == 0:
            self.pending["seen_views"].add(int(view_id))
            return

        # Opaque camera values keep subsequent camera views on the same kernel.
        scalar = lambda value: dr.opaque(self.Float64, float(value))
        position, right, up, forward = [tuple(scalar(value) for value in camera[key])
                                       for key in ("position", "right", "up", "forward")]
        tan_x = math.tan(camera["camera_angle_x"] * .5)
        tan_y = tan_x * height / width
        tan_x, tan_y = scalar(tan_x), scalar(tan_y)
        near = scalar(camera.get("near_clip", .1))
        far = scalar(camera.get("far_clip", 1000.))
        fg_threshold = dr.opaque(self.Float, settings.foreground_alpha_min)
        bg_threshold = dr.opaque(self.Float, settings.background_alpha_max)
        deficit = np.clip(settings.growth_min_alpha_deficit, .001, 1.)
        # NumPy 2 promotes the float32 difference against this np.float64
        # scalar. NumPy 1 can retain float32. Preserve either installed oracle.
        deficit_type = (self.Float64 if np.result_type(np.empty(0, dtype=np.float32), deficit)
                        == np.dtype(np.float64) else self.Float)
        deficit = dr.opaque(deficit_type, float(deficit))
        foreground = dr.zeros(mi.Bool, self.count)
        background = dr.zeros(mi.Bool, self.count)
        dot = lambda a, b: (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
        for world in self.points:
            relative = tuple(world[axis] - position[axis] for axis in range(3))
            depth = dot(relative, forward)
            u = .5 + .5 * dot(relative, right) / dr.maximum(depth * tan_x, 1e-30)
            v = .5 - .5 * dot(relative, up) / dr.maximum(depth * tan_y, 1e-30)
            visible = ((depth >= near) & (depth <= far)
                       & (u >= 0) & (u < 1) & (v >= 0) & (v < 1))
            # Inactive projections must not create invalid float-to-int casts
            # or out-of-range gathers, including points behind the camera.
            x = self.UInt32(dr.floor(dr.select(visible, u, 0.) * width))
            y = self.UInt32(dr.floor(dr.select(visible, v, 0.) * height))
            index = y * width + x
            ref = dr.gather(self.Float, reference, index, visible)
            pred = dr.gather(self.Float, rendered, index, visible)
            background |= visible & (ref <= bg_threshold)
            foreground |= (visible & (ref >= fg_threshold)
                           & (deficit_type(ref - pred) >= deficit))
        # A single background sample vetoes foreground for this entire view,
        # matching Falcor and the CPU oracle. A view contributes at most once.
        self.foreground = self.foreground + self.UInt32(foreground & ~background)
        self.background = self.background + self.UInt32(background)
        dr.eval(self.foreground, self.background)
        self.pending["seen_views"].add(int(view_id))

    def materialize(self):
        """Refresh serializable CPU counters at a topology/checkpoint boundary."""
        dr, np = self.dr, self.np
        dr.eval(self.foreground, self.background)
        dr.sync_thread()
        result = {"foreground": np.array(self.foreground, dtype=np.uint32, copy=True),
                  "background": np.array(self.background, dtype=np.uint32, copy=True)}
        self.pending.update(result)
        self.materializations += 1
        return result
