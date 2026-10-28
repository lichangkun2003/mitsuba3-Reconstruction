"""CPU regression for GPU-reference LRU resizing and cumulative accounting."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_reference import GPUReferenceCache


class _HostTypes:
    @staticmethod
    def Float(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def Color3f(*planes):
        return planes


class _HostEvaluation:
    @staticmethod
    def eval(*values):
        pass


class ReferenceCacheResizeTests(unittest.TestCase):
    def test_shrinking_evicts_least_recent_images_and_preserves_transfer_counters(self):
        # Four RGB images occupy exactly 3 MiB; a 1 MiB cap holds only one.
        cache = GPUReferenceCache(256, 256, _HostTypes, _HostEvaluation, np,
                                  capacity_mib=3, rgb_only=True)
        pixels = np.full((256, 256, 3), .25, dtype=np.float32)
        references = {name: cache.get(f"resize-{name}.png", pixels) for name in "abcd"}
        self.assertIs(cache.get("resize-b.png", pixels), references["b"])
        before = cache.statistics()
        self.assertEqual(before["images"], 4)
        self.assertEqual(before["resident_bytes"], 3 * 1024**2)

        cache.resize(1)
        after = cache.statistics()
        self.assertEqual(after["capacity_bytes"], 1024**2)
        self.assertEqual(after["resident_bytes"], 256 * 256 * 3 * 4)
        self.assertEqual(after["images"], 1)
        self.assertEqual(after["evictions"], 3)
        self.assertEqual([Path(key[0]).name for key in cache._images], ["resize-b.png"])
        for name in ("hits", "misses", "uploads", "uploaded_bytes"):
            self.assertEqual(after[name], before[name], name)
        self.assertIs(cache.get("resize-b.png", pixels), references["b"])

        # Resizing never creates an upload or alters a retained caller's image.
        stable = cache.statistics()
        cache.resize(1)
        self.assertEqual(cache.statistics(), stable)
        cache.resize(3)
        self.assertEqual(cache.statistics(), {**stable, "capacity_bytes": 3 * 1024**2})
        np.testing.assert_array_equal(references["a"].rgb[0], pixels[..., 0].ravel())
        self.assertIsNot(cache.get("resize-d.png", pixels), references["d"])
        self.assertEqual(cache.statistics()["uploads"], 5)
        self.assertEqual(cache.statistics()["misses"], 5)
        self.assertEqual(cache.statistics()["evictions"], 3)

        cache.resize(0)
        disabled = cache.statistics()
        self.assertEqual(disabled["capacity_bytes"], 0)
        self.assertEqual(disabled["resident_bytes"], 0)
        self.assertEqual(disabled["images"], 0)
        self.assertEqual(disabled["evictions"], 5)
        self.assertIsNone(cache.get("resize-b.png", pixels))
        self.assertEqual(cache.statistics()["uploads"], disabled["uploads"])
        self.assertEqual(cache.statistics()["misses"], disabled["misses"] + 1)
        for value in (-1, True, 1.5, "1"):
            with self.subTest(invalid_capacity=value):
                snapshot = cache.statistics()
                with self.assertRaises(ValueError):
                    cache.resize(value)
                self.assertEqual(cache.statistics(), snapshot)


if __name__ == "__main__":
    unittest.main()
