"""Bounded device cache of the static, linear RGB or RGBA training references.

Legacy RGBA targets remain available for contour and topology observations.
RGB-only mode stores three composited color planes and has no reference-alpha
plane. Clear the cache before reusing a path for a different reference image.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


class GPUReference:
    """Immutable-by-convention device planes for one validated reference."""

    def __init__(self, rgba, width, height, mi, dr, np, *, rgb_only=False):
        self.width, self.height = width, height
        self.mi, self.dr = mi, dr
        self.rgb_only = rgb_only
        channels = 3 if rgb_only else 4
        flat = rgba.reshape(-1, channels)
        planes = [mi.Float(np.ascontiguousarray(flat[:, channel])) for channel in range(channels)]
        self.rgb = mi.Color3f(*planes[:3])
        # A scalar compatibility argument is not an image plane or alpha target.
        self.alpha = mi.Float(1) if rgb_only else planes[3]
        # Materialize the upload once; leave synchronization to the consumer's
        # stream, without an extra host wait or device-to-host pixel transfer.
        dr.eval(self.rgb, self.alpha)

    def batch(self, begin, end):
        """Return the unchanged full image or gather a contiguous pixel batch."""
        if (isinstance(begin, bool) or not isinstance(begin, int)
                or isinstance(end, bool) or not isinstance(end, int)
                or not 0 <= begin < end <= self.width * self.height):
            raise ValueError("Reference batch must satisfy 0 <= begin < end <= pixel count")
        if begin == 0 and end == self.width * self.height:
            return self.rgb, self.alpha
        mi, dr = self.mi, self.dr
        # The offset is runtime data so equally sized chunks reuse kernels.
        index = dr.arange(mi.UInt32, end - begin) + dr.opaque(mi.UInt32, begin)
        alpha = self.alpha if self.rgb_only else dr.gather(mi.Float, self.alpha, index)
        return dr.gather(mi.Color3f, self.rgb, index), alpha


class GPUReferenceCache:
    """LRU cache bounded by the logical size of resident float32 image planes.

    Keys are canonical image paths and dimensions. The source image for a key
    must remain unchanged during a run, matching ReferenceImageCache's static
    dataset contract. ``None`` means the caller should use its previous upload
    path (disabled cache or one image larger than the entire capacity).

    Byte accounting excludes Dr.Jit's allocator reserve and any arrays still
    referenced by an in-flight caller after eviction or ``clear()``.
    """

    def __init__(self, width, height, mi, dr, np, capacity_mib=1024, *, rgb_only=False):
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Reference {name} must be a positive integer")
        if width * height > 2**32 - 1:
            raise ValueError("Reference pixel count exceeds the UInt32 index range")
        if isinstance(capacity_mib, bool) or not isinstance(capacity_mib, int) or capacity_mib < 0:
            raise ValueError("gpu_reference_cache_mib must be a nonnegative integer")
        self.width, self.height = width, height
        self.mi, self.dr, self.np = mi, dr, np
        self.rgb_only = rgb_only
        self.channels = 3 if rgb_only else 4
        self.capacity_bytes = capacity_mib * 1024 * 1024
        self.image_bytes = width * height * self.channels * 4
        self.resident_bytes = 0
        self.hits = self.misses = self.uploads = self.uploaded_bytes = self.evictions = 0
        self._images = OrderedDict()

    def get(self, path, rgba):
        """Upload three RGB or four legacy RGBA planes only on a cache miss."""
        np = self.np
        if (not isinstance(rgba, np.ndarray) or rgba.dtype != np.dtype(np.float32)
                or rgba.shape != (self.height, self.width, self.channels)):
            raise ValueError(f"GPU reference must be a float32 image of shape (height, width, {self.channels})")
        if self.capacity_bytes < self.image_bytes:
            self.misses += 1
            return None
        key = (str(Path(path).resolve()), self.width, self.height)
        if key in self._images:
            self.hits += 1
            self._images.move_to_end(key)
            return self._images[key]
        self.misses += 1
        if (not np.isfinite(rgba).all() or (not self.rgb_only
                and (np.any(rgba[..., 3] < 0) or np.any(rgba[..., 3] > 1)))):
            raise ValueError(f"Reference contains nonfinite values or invalid alpha: {path}")
        while self.resident_bytes + self.image_bytes > self.capacity_bytes:
            self._images.popitem(last=False)
            self.resident_bytes -= self.image_bytes
            self.evictions += 1
        reference = GPUReference(rgba, self.width, self.height, self.mi, self.dr, np,
                                 rgb_only=self.rgb_only)
        self._images[key] = reference
        self.resident_bytes += self.image_bytes
        self.uploads += 1
        self.uploaded_bytes += self.image_bytes
        return reference

    def clear(self):
        """Drop resident references; preserve cumulative transfer statistics."""
        self._images.clear()
        self.resident_bytes = 0

    def resize(self, capacity_mib):
        """Adjust the LRU budget at a model/topology boundary."""
        if type(capacity_mib) is not int or capacity_mib < 0:
            raise ValueError("Reference cache capacity must be a nonnegative integer")
        self.capacity_bytes = capacity_mib * 1024 * 1024
        while self.resident_bytes > self.capacity_bytes:
            self._images.popitem(last=False)
            self.resident_bytes -= self.image_bytes
            self.evictions += 1

    def statistics(self):
        return {"capacity_bytes": self.capacity_bytes, "resident_bytes": self.resident_bytes,
                "images": len(self._images), "hits": self.hits, "misses": self.misses,
                "uploads": self.uploads, "uploaded_bytes": self.uploaded_bytes,
                "evictions": self.evictions}
