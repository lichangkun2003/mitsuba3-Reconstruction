"""Stable proposal merge oracle and optional CUDA parity, without training."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import reconstruction_adaptive_growth as adaptive
import reconstruction_growth_candidates_gpu as core
from reconstruction_world_shell import adaptive_encode_numpy


class ArrayDevice:
    cuda=SimpleNamespace(UInt64=lambda value:np.asarray(value,np.uint64),
                         UInt32=lambda value:np.asarray(value,np.uint32),
                         Int32=lambda value:np.asarray(value,np.int32),
                         Int64=lambda value:np.asarray(value,np.int64),
                         Float64=lambda value:np.asarray(value,np.float64),
                         Float32=lambda value:np.asarray(value,np.float32),
                         Array3f64=lambda *values:np.asarray(np.broadcast_arrays(*values),np.float64),
                         Array4f64=lambda *values:np.asarray(np.broadcast_arrays(*values),np.float64),
                         Bool=lambda value:np.asarray(value,bool))
    ReduceOp=SimpleNamespace(Add='add')
    ReduceMode=SimpleNamespace(Permute='permute')
    maximum=staticmethod(np.maximum)

    @staticmethod
    def argsort(value): return np.argsort(value,kind='stable').astype(np.uint32)
    @staticmethod
    def width(value): return len(np.atleast_1d(value))
    @staticmethod
    def arange(dtype,count): return dtype(np.arange(count))
    @staticmethod
    def empty(dtype,count): return np.empty(count,dtype=dtype(0).dtype)
    @staticmethod
    def zeros(dtype,count): return dtype(np.zeros(count))
    @staticmethod
    def opaque(dtype,value): return dtype(value)
    @staticmethod
    def compress(mask): return np.flatnonzero(mask).astype(np.uint32)
    @staticmethod
    def cumsum(value): return np.cumsum(value,dtype=np.uint32)
    @staticmethod
    def gather(dtype,value,indices): return (np.asarray(value[indices]) if dtype is np.ndarray else dtype(value[indices]))
    @staticmethod
    def scatter(target,value,indices,**kwargs): target[indices]=value
    @staticmethod
    def scatter_reduce(operation,target,value,indices):
        assert operation=='add'
        np.add.at(target,indices,value)
    @staticmethod
    def eval(*values): pass
    @staticmethod
    def ravel(value,order): return value.ravel(order=order)


METADATA=dict(resolution=8,background_shell_levels=3,background_refinement='adaptive')


def proposal_fixture(count=170):
    rng=np.random.default_rng(39182+count)
    # Several ancestors/descendants and disjoint targets, plus foreground.
    targets=np.r_[np.array([1,2,31,147],np.uint64),
        adaptive_encode_numpy([0,0,0,0,1,56,56,112],[0,1,2,2,0,0,1,4],
                              [0,0,0,1,0,0,7,512],np)]
    cells=targets[rng.integers(0,len(targets),count)]
    parents=np.array([0,17,2**32+1,2**63,2**63+7,2**64-1],np.uint64)[rng.integers(0,6,count)]
    return cells,parents


def oracle(cells,parents,minimum_support):
    # Python's stable tuple sort deliberately differs from the GPU's two radix
    # passes; duplicate pair geometry must come from the first original row.
    order=sorted(range(len(cells)),key=lambda row:(int(cells[row]),int(parents[row])))
    seen=set()
    unique=[]
    for row in order:
        pair=int(cells[row]),int(parents[row])
        if pair not in seen:
            unique.append(row)
            seen.add(pair)
    support=Counter(int(cells[row]) for row in unique)
    kept=[row for row in unique if support[int(cells[row])]>=minimum_support]
    targets=np.array(sorted({int(cells[row]) for row in kept}),np.uint64)
    targets=targets[adaptive.filter_nonoverlapping_growth_numpy(targets,METADATA,np)]
    allowed=set(map(int,targets))
    rows=np.array([row for row in kept if int(cells[row]) in allowed],np.uint32)
    return rows,np.array([support[int(cells[row])] for row in rows],np.uint32)


def check_merge(dr,cells,parents,minimum_support):
    expected_rows,expected_support=oracle(cells,parents,minimum_support)
    rows,support=adaptive._merge_proposal_rows_gpu(dr.cuda.UInt64(cells),dr.cuda.UInt64(parents),
                                                 METADATA,minimum_support,dr)
    dr.eval(rows,support)
    np.testing.assert_array_equal(np.asarray(rows),expected_rows)
    np.testing.assert_array_equal(np.asarray(support),expected_support)
    # Different values on duplicate input rows make stable tie preservation
    # observable in every geometry field, rather than only in cell addresses.
    for width in (1,3,4):
        values=np.arange(len(cells)*width,dtype=np.float64).reshape(-1,width)/17
        np.testing.assert_array_equal(values[np.asarray(rows)],values[expected_rows])


def check_append(dr,capacity,batch_sizes):
    buffer=adaptive._BackgroundProposalBuffer(capacity,dr)
    assert buffer.fields is None
    expected={name:[] for name in adaptive._WIDTHS}
    begin=0
    for size in batch_sizes:
        piece={}
        for name,width in adaptive._WIDTHS.items():
            if name in ('cells','parent_cells'):
                values=np.arange(begin,begin+size,dtype=np.uint64)+np.uint64(2**63+17)
                dtype=dr.cuda.UInt64
            elif name=='parent_ids':
                values=np.arange(begin,begin+size,dtype=np.uint32)
                dtype=dr.cuda.UInt32
            else:
                values=np.arange(begin*width,(begin+size)*width,dtype=np.float64)/17+1e-11
                dtype=dr.cuda.Float64
            expected[name].append(values.copy())
            piece[name]=dtype(values)
        buffer.append(piece)
        begin+=size
        assert buffer.count==begin
        if not begin:
            assert buffer.fields is None
    actual=buffer.finish()
    assert buffer.fields is None
    if not begin:
        assert actual is None
        return
    dr.eval(*actual.values())
    for name in adaptive._WIDTHS:
        np.testing.assert_array_equal(np.asarray(actual[name]),np.concatenate(expected[name]),err_msg=name)


def check_pack(dr,count,width,dtype):
    if dtype is dr.cuda.Float64:
        values=np.array([0.,-0.,1.+2.**-40,1e-280,1e280,np.inf,-np.inf,np.nan],np.float64)
        bits=np.uint64
    else:
        values=np.array([0.,-0.,1.+2.**-20,1e-30,1e30,np.inf,-np.inf,np.nan],np.float32)
        bits=np.uint32
    rows=np.resize(values,(count,width))
    fields=[dtype(rows[:,j]) for j in range(width)]
    actual=core._pack(fields,dtype,dr)
    dr.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual).view(bits),rows.ravel().view(bits))


class GrowthMergeTests(unittest.TestCase):
    def test_static_vector_pack_preserves_fp64_bits_and_aos_layout(self):
        dr=ArrayDevice()
        for count in (0,1,19,2048):
            for width in (3,4):
                with self.subTest(count=count,width=width):
                    check_pack(dr,count,width,dr.cuda.Float64)

    def test_pack_fallback_preserves_other_widths_and_float32(self):
        dr=ArrayDevice()
        with patch.object(dr,'ravel',side_effect=AssertionError('unexpected static-vector path')):
            for dtype,width in ((dr.cuda.Float64,2),(dr.cuda.Float32,3),(dr.cuda.Float32,4)):
                check_pack(dr,19,width,dtype)

    def test_bounded_background_append_preserves_batches_precision_and_tail(self):
        for capacity,sizes in ((0,[]),(0,[0]),(12,[0,3,1,5]),(9,[3,1,5]),(8,[8])):
            with self.subTest(capacity=capacity,sizes=sizes):
                check_append(ArrayDevice(),capacity,sizes)
        with self.assertRaisesRegex(RuntimeError,'candidate cap'):
            check_append(ArrayDevice(),1,[2])

    def test_stable_pair_dedup_support_and_nested_leaf_rejection(self):
        dr=ArrayDevice()
        with patch.object(adaptive,'filter_nonoverlapping_growth_gpu',
                side_effect=lambda cells,metadata,device:adaptive.filter_nonoverlapping_growth_numpy(cells,metadata,np)):
            for count in (0,1,3,64,170,511):
                for support in (1,2,3,26):
                    with self.subTest(count=count,support=support):
                        check_merge(dr,*proposal_fixture(count),support)

    def test_raw_core_stream_defers_its_redundant_sort_and_support_pass(self):
        dr=ArrayDevice()
        state=dict(cell_indices=np.array([1,3],np.uint64),centers=np.zeros(6,np.float32),
                   log_scales=np.zeros(6,np.float32),rotations=np.tile([1,0,0,0],2).astype(np.float32))
        payload=dict(model_metadata=dict(resolution=8),model_state=state)
        cfg=SimpleNamespace(use_growth_evidence=False,use_rgb_growth=True,growth_evidence_penetration=.05,
                            growth_use_26_neighbors=True,growth_min_supporting_parents=2)
        manager=SimpleNamespace(settings=cfg,schedule=SimpleNamespace(enabled=False))
        raw=dict(cells=np.array([4,2],np.uint32),parent_cells=np.array([3,1],np.uint32),
                 parent_ids=np.array([1,0],np.uint32),centers=np.arange(6,dtype=np.float64),
                 log_scales=np.arange(6,dtype=np.float64)+10,rotations=np.arange(8,dtype=np.float64))
        with patch.object(core,'_make_bitmap',return_value=np.array([0],np.uint32)), \
             patch.object(core,'_exposed_parent_ids',return_value=np.array([0,1],np.uint32)), \
             patch.object(core,'_parent_batch',return_value=dict(raw)), \
             patch.object(core,'_concatenate_fields',return_value=dict(raw)), \
             patch.object(dr,'argsort',side_effect=AssertionError('redundant sort')):
            actual=core._generate(manager,payload,30,None,np.ones(2,bool),np.array([],np.uint64),65536,dr,
                                  minimum_support=1,geometry_only=True)
        for name,value in raw.items():
            np.testing.assert_array_equal(actual[name],value)
        self.assertEqual(actual['boundary'],30)


def check_cuda(dr):
    for count in (0,1,19,2048):
        for width in (3,4):
            check_pack(dr,count,width,dr.cuda.Float64)
    for dtype,width in ((dr.cuda.Float64,2),(dr.cuda.Float32,3),(dr.cuda.Float32,4)):
        check_pack(dr,19,width,dtype)
    for count in (0,1,3,64,170,511,4099):
        for support in (1,2,3,26):
            check_merge(dr,*proposal_fixture(count),support)
    for capacity,sizes in ((0,[]),(0,[0]),(12,[0,3,1,5]),(9,[3,1,5]),(8192,[2048,3,2048,17])):
        check_append(dr,capacity,sizes)
    dr.sync_thread()
    print('PASS CUDA FP64 AoS packing, stable merge, duplicate ties, UInt64 keys and mixed-depth targets',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    tests=unittest.defaultTestLoader.loadTestsFromTestCase(GrowthMergeTests)
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful(): raise SystemExit(1)
    if args.gpu:
        from preview import RuntimeConfig
        from reconstruction_common import initialize_runtime,WORKSPACE
        _,dr,_,_=initialize_runtime(RuntimeConfig(),WORKSPACE/'.cache'/'growth-merge',{})
        check_cuda(dr)
