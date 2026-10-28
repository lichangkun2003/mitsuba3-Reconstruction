"""CPU contract checks and optional CUDA bit-exact bounded checkpoint readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_export import (ArraySource, CHUNK_BYTES, checkpoint_array_sources,
                                   device_array_source, zero_array_source)


class FakeArray:
    numpy_dtype = np.dtype('f4')
    vartype = 'Float32'

    def __init__(self, values, *, gathered=False):
        self.values = np.asarray(values, dtype=self.numpy_dtype)
        self.gathered = gathered

    def __len__(self):
        return self.values.size

    def __array__(self, dtype=None, copy=None):
        if not self.gathered:
            raise AssertionError('Attempted full device-source readback')
        return self.values.astype(dtype or self.values.dtype, copy=bool(copy))

    def __add__(self, other):
        return type(self)(self.values + other.values)


class FakeUInt32(FakeArray):
    numpy_dtype = np.dtype('u4')
    vartype = 'UInt32'


class FakeUInt64(FakeArray):
    numpy_dtype = np.dtype('u8')
    vartype = 'UInt64'


class FakeUInt8(FakeArray):
    numpy_dtype = np.dtype('u1')
    vartype = 'UInt8'


class FakeDr:
    VarType = SimpleNamespace(Float32='Float32', UInt32='UInt32', UInt64='UInt64', UInt8='UInt8')

    def __init__(self):
        self.requests = []
        self.offsets = []

    @staticmethod
    def is_array_v(dtype):
        return isinstance(dtype, type) and issubclass(dtype, FakeArray)

    is_tensor_v = staticmethod(lambda dtype: False)
    is_dynamic_v = staticmethod(lambda dtype: True)
    depth_v = staticmethod(lambda dtype: 1)
    detach = staticmethod(lambda value, preserve_type: value)
    type_v = staticmethod(lambda dtype: dtype.vartype)
    uint32_array_t = staticmethod(lambda dtype: FakeUInt32)
    arange = staticmethod(lambda dtype, count: dtype(np.arange(count, dtype=np.uint32)))
    eval = staticmethod(lambda value: None)

    def opaque(self, dtype, value):
        self.offsets.append(value)
        return dtype([value])

    def gather(self, dtype, source, index):
        self.requests.append((int(index.values[0]), len(index), index.values.nbytes))
        return dtype(source.values[index.values], gathered=True)


class ExportSourceTests(unittest.TestCase):
    def test_source_validates_shape_range_and_callback_contract(self):
        calls = []
        source = ArraySource((2, 3), 'u4', lambda start, count:
                             calls.append((start, count)) or np.arange(start, start+count, dtype=np.uint32))
        self.assertEqual(source.shape, (2, 3))
        self.assertEqual(source.size, 6)
        self.assertFalse(hasattr(source, '__array__'))
        np.testing.assert_array_equal(source.read_chunk(2, 3), [2, 3, 4])
        self.assertEqual(calls, [(2, 3)])
        self.assertEqual(source.read_chunk(6, 0).shape, (0,))
        self.assertEqual(len(calls), 1)
        for start, count in ((-1, 1), (0, -1), (6, 1), (7, 0), (True, 1), (0, 1.5)):
            with self.assertRaises(ValueError):
                source.read_chunk(start, count)
        for result in (np.zeros(2, np.float32), np.zeros((1, 2), np.uint32),
                       np.zeros(4, np.uint32)[::2], [0, 0]):
            with self.assertRaisesRegex(ValueError, 'exact 1D'):
                ArraySource((2,), 'u4', lambda start, count: result).read_chunk(0, 2)

    def test_large_virtual_zero_source_does_not_allocate_its_shape(self):
        source = zero_array_source((50_000_000, 6), 'u4')
        self.assertEqual(source.size, 300_000_000)
        np.testing.assert_array_equal(source.read_chunk(source.size-7, 7), np.zeros(7, np.uint32))
        with self.assertRaisesRegex(ValueError, 'max_chunk_elements'):
            source.read_chunk(0, source.max_chunk_elements+1)
        self.assertEqual(zero_array_source((0, 3), 'f4').read_chunk(0, 0).size, 0)
        self.assertEqual(zero_array_source((), 'u8').read_chunk(0, 1).size, 1)

    def test_numpy_sources_are_views_and_schema_casts_are_bounded(self):
        original = np.arange(12, dtype=np.int32).reshape(3, 4)
        source = device_array_source(original)
        self.assertEqual(source.shape, original.shape)
        self.assertTrue(np.shares_memory(source.read_chunk(2, 5), original))
        widened = device_array_source(original, dtype=np.int64, shape=(12,))
        chunk = widened.read_chunk(9, 3)
        self.assertEqual(chunk.dtype, np.dtype('i8'))
        np.testing.assert_array_equal(chunk, [9, 10, 11])
        with self.assertRaisesRegex(ValueError, 'C-contiguous'):
            device_array_source(original.T)
        with self.assertRaisesRegex(ValueError, 'element count'):
            device_array_source(original, shape=(13,))
        for data, target in ((np.array([2**53+1], np.uint64), 'f8'),
                             (np.array([2**53+1], np.int64), 'f8'),
                             (original, 'u4'), (original, 'i2')):
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                device_array_source(data, dtype=target)

    def test_device_gather_never_converts_the_full_source_and_preserves_bits(self):
        runtime = FakeDr()
        words = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                          0x3f800000, 0xff7fffff], np.uint32)
        original = FakeArray(words.view(np.float32))
        source = device_array_source(original, dr=runtime)
        first = source.read_chunk(0, 3)
        second = source.read_chunk(3, 5)
        np.testing.assert_array_equal(np.concatenate([first, second]).view(np.uint32), words)
        np.testing.assert_array_equal(original.values.view(np.uint32), words)
        self.assertEqual(runtime.offsets, [0, 3])
        self.assertEqual(runtime.requests, [(0, 3, 12), (3, 5, 20)])

    def test_uint64_does_not_pass_through_floating_point(self):
        values = np.array([0, 2**32+1, 2**53+1, 2**63+1, 2**64-1], np.uint64)
        source = device_array_source(FakeUInt64(values), dr=FakeDr())
        self.assertEqual(source.max_chunk_elements, CHUNK_BYTES//8)
        np.testing.assert_array_equal(source.read_chunk(1, 4), values[1:])
        byte_source = device_array_source(FakeUInt8([0, 255]), dtype='u4', dr=FakeDr())
        self.assertEqual(byte_source.max_chunk_elements, CHUNK_BYTES//4)
        np.testing.assert_array_equal(byte_source.read_chunk(0, 2), np.array([0, 255], np.uint32))

    def test_nested_wrapper_preserves_numpy_arrays_and_container_kinds(self):
        cpu = np.arange(3, dtype=np.uint32)
        gpu = FakeUInt32([4, 5, 6])
        old_source = zero_array_source((2,), 'u4')
        original = {'array':cpu, 'items':[gpu, (old_source, 'label', 17)]}
        result = checkpoint_array_sources(original, dr=FakeDr())
        self.assertIs(result['array'], cpu)
        self.assertIsInstance(result['items'], list)
        self.assertIsInstance(result['items'][1], tuple)
        self.assertIs(result['items'][1][0], old_source)
        self.assertIs(original['items'][0], gpu)
        np.testing.assert_array_equal(result['items'][0].read_chunk(0, 3), [4, 5, 6])


class CudaExportTests(unittest.TestCase):
    runtime = None

    def test_bit_exact_multichunk_export_without_mutating_sources(self):
        dr = self.runtime
        if dr is None:
            self.skipTest('Use --cuda for GPU checks')
        cases = [
            (dr.cuda.Float32, np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                                       0x00800000, 0x3f800000, 0xff7fffff], np.uint32).view(np.float32)),
            (dr.cuda.UInt32, np.array([0, 1, 2**31, 2**32-1], np.uint32)),
            (dr.cuda.UInt64, np.array([0, 2**32+1, 2**53+1, 2**63+1, 2**64-1], np.uint64)),
        ]
        for dtype, pattern in cases:
            count = CHUNK_BYTES // max(pattern.dtype.itemsize, 4) + 19
            expected = np.resize(pattern, count)
            device = dtype(expected)
            dr.eval(device)
            source = device_array_source(device, dr=dr)
            chunks = []
            for start in range(0, count, source.max_chunk_elements):
                chunk = source.read_chunk(start, min(source.max_chunk_elements, count-start))
                self.assertLessEqual(chunk.nbytes, CHUNK_BYTES)
                # Test-only host copies permit comparison after each pinned
                # view is released; the actual writer consumes memoryviews.
                chunks.append(chunk.copy())
                del chunk
            actual = np.concatenate(chunks)
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
            np.testing.assert_array_equal(source.read_chunk(0, len(pattern)).view(np.uint8),
                                           pattern.view(np.uint8))
            self.assertEqual(len(device), count)
            del source, device, chunks, actual, expected
        widened = device_array_source(dr.cuda.Int32([-2**31, 0, 2**31-1]), dtype='i8', dr=dr)
        np.testing.assert_array_equal(widened.read_chunk(0, 3), np.array([-2**31, 0, 2**31-1], np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    handles = None
    if args.cuda:
        from preview import RuntimeConfig, WORKSPACE
        from reconstruction_common import initialize_runtime
        output = WORKSPACE / '.cache/export-regression'
        output.mkdir(parents=True, exist_ok=True)
        mi, dr, unused_numpy, handles = initialize_runtime(RuntimeConfig(gpu=args.gpu), output, {})
        CudaExportTests.runtime = dr
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ExportSourceTests)])
    if args.cuda:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CudaExportTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.cuda:
        dr.sync_thread()
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
