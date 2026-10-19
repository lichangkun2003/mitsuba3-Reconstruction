"""Test preview runtime setup without importing GPU libraries or rendering.

Run with the project's interpreter, including Python isolated mode (-I).
Every environment, subprocess, Dr.Jit object and device array below is mocked.
"""

from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GPU_UUID = "GPU-12345678-1234-abcd-5678-123456789abc"
MIG_UUID = "MIG-12345678-1234-abcd-5678-123456789abc"
LEGACY_MIG_UUID = f"MIG-{GPU_UUID}/1/0"
SETUP_ERRORS = (RuntimeError, ValueError)


def load_preview():
    path = Path(__file__).resolve().parents[1] / "preview.py"
    name = "_reconstruction_preview_runtime_tests"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load preview module: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    sys.modules[name] = module
    original_import = builtins.__import__

    def stdlib_only(name, *args, **kwargs):
        if name.partition(".")[0] in {"numpy", "mitsuba", "drjit"}:
            raise AssertionError(f"Runtime setup tests must not import {name}")
        return original_import(name, *args, **kwargs)

    with mock.patch.object(sys, "dont_write_bytecode", True), \
            mock.patch.object(builtins, "__import__", side_effect=stdlib_only):
        spec.loader.exec_module(module)
    return module


class FakeDrJit:
    def __init__(self, actual=24, forced_after_set=None):
        self.actual = actual
        self.forced_after_set = forced_after_set
        self.requests = []

    def set_thread_count(self, requested):
        self.requests.append(requested)
        self.actual = requested if self.forced_after_set is None else self.forced_after_set

    def thread_count(self):
        return self.actual


class FakeDeviceArray:
    def __init__(self, device):
        self.device = device
        self.queries = 0

    def __dlpack_device__(self):
        self.queries += 1
        return self.device


class RuntimeSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preview = load_preview()

    def slurm_environment(self, visible="0", **updates):
        result = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_GPUS": "5,7",
            "SLURM_STEP_GPUS": "5,7",
            "CUDA_VISIBLE_DEVICES": visible,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": "scheduler-owned",
            "ROCR_VISIBLE_DEVICES": "scheduler-owned",
            "OMP_NUM_THREADS": "37",
            "MKL_NUM_THREADS": "31",
        }
        result.update(updates)
        return result

    def prepare_slurm(self, environment, gpu=7):
        report = {}
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(self.preview.subprocess, "run") as run:
            before = dict(os.environ)
            self.preview.prepare_device_environment(gpu, report)
            run.assert_not_called()
            self.assertEqual(dict(os.environ), before, "Slurm-owned environment was changed")
        return report

    def test_import_uses_only_standard_library(self):
        self.assertTrue(callable(self.preview.prepare_device_environment))
        self.assertTrue(callable(self.preview.configure_drjit_threads))
        self.assertTrue(callable(self.preview.verify_cuda_device))

    def test_slurm_accepts_device_forms_without_querying_or_rewriting(self):
        values = ("0", "2,5", GPU_UUID, "GPU-12345678", MIG_UUID, LEGACY_MIG_UUID,
                  f"{GPU_UUID},{MIG_UUID}", " 2 , 5 ")
        for value in values:
            with self.subTest(visible=value):
                report = self.prepare_slurm(self.slurm_environment(value))
                gpu = report["gpu"]
                self.assertEqual(gpu["selection"], "slurm")
                self.assertEqual(gpu["cuda_visible_devices"], value)
                self.assertEqual(gpu["visible_device_tokens"], [part.strip() for part in value.split(",")])
                self.assertEqual(gpu["local_gpu_index_ignored"], 7)
                self.assertEqual(gpu["slurm"]["SLURM_JOB_ID"], "12345")

    def test_slurm_detection_uses_key_presence_for_both_names(self):
        for key in ("SLURM_JOB_ID", "SLURM_JOBID"):
            for value in ("12345", ""):
                with self.subTest(key=key, value=value):
                    report = self.prepare_slurm({key: value, "CUDA_VISIBLE_DEVICES": "3"})
                    self.assertEqual(report["gpu"]["selection"], "slurm")
                    self.assertEqual(report["gpu"]["slurm"][key], value)

    def test_slurm_rejects_missing_or_invalid_devices_without_fallback(self):
        invalid = (None, "", " ", "NoDevFiles", "none", "-1", "0,-1", "-2,0", "1.0",
                   "0,,1", ",0", "0,", "GPU-", "GPU-not-hex", "MIG-", "MIG-not-a-uuid",
                   f"MIG-{GPU_UUID}/-1/0", f"MIG-{GPU_UUID}/1/x", f"{GPU_UUID}/1/0")
        for value in invalid:
            with self.subTest(visible=value):
                environment = self.slurm_environment(value or "")
                if value is None:
                    del environment["CUDA_VISIBLE_DEVICES"]
                with mock.patch.dict(os.environ, environment, clear=True), \
                        mock.patch.object(self.preview.subprocess, "run") as run:
                    before = dict(os.environ)
                    with self.assertRaises(SETUP_ERRORS):
                        self.preview.prepare_device_environment(0, {})
                    run.assert_not_called()
                    self.assertEqual(dict(os.environ), before)

    def test_slurm_cpu_quota_defaults_to_one(self):
        report = self.prepare_slurm(self.slurm_environment())
        self.assertEqual(report["cpu_threads"]["requested"], 1)
        self.assertIn("source", report["cpu_threads"])

    def test_slurm_cpu_quota_accepts_positive_integer(self):
        report = self.prepare_slurm(self.slurm_environment(SLURM_CPUS_PER_TASK="8"))
        self.assertEqual(report["cpu_threads"]["requested"], 8)
        self.assertIn("source", report["cpu_threads"])

    def test_slurm_cpu_quota_rejects_invalid_values_without_fallback(self):
        for value in ("", "0", "-1", "8.5", "NaN", "abc", " 8", "8 ", "+8", "８", "٨"):
            with self.subTest(cpus=value):
                environment = self.slurm_environment(SLURM_CPUS_PER_TASK=value)
                with mock.patch.dict(os.environ, environment, clear=True), \
                        mock.patch.object(self.preview.subprocess, "run") as run:
                    before = dict(os.environ)
                    report = {}
                    dr = FakeDrJit()
                    with self.assertRaises(SETUP_ERRORS):
                        self.preview.prepare_device_environment(0, report)
                        self.preview.configure_drjit_threads(dr, report)
                    run.assert_not_called()
                    self.assertEqual(dr.requests, [], "Invalid quotas must fail before configuring Dr.Jit")
                    self.assertEqual(dict(os.environ), before)

    def test_local_queries_requested_physical_index_and_sets_its_uuid(self):
        response = SimpleNamespace(stdout=f"7, NVIDIA Test GPU, {GPU_UUID}, 610.62, 8192\n")
        environment = {"CUDA_VISIBLE_DEVICES": "old-mask", "OMP_NUM_THREADS": "37",
                       "NVIDIA_VISIBLE_DEVICES": "local-container"}
        report = {}
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(self.preview.subprocess, "run", return_value=response) as run:
            self.preview.prepare_device_environment(7, report)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[0], "nvidia-smi")
            self.assertIn("--id=7", command)
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], GPU_UUID)
            for key, value in environment.items():
                if key != "CUDA_VISIBLE_DEVICES":
                    self.assertEqual(os.environ[key], value)
        self.assertEqual(report["gpu"]["selection"], "local")
        self.assertEqual(report["gpu"]["uuid"], GPU_UUID)

    def test_local_query_failure_is_not_a_cpu_fallback(self):
        report = {}
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(self.preview.subprocess, "run",
                                  side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                self.preview.prepare_device_environment(0, report)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_slurm_drjit_thread_count_is_configured_and_recorded(self):
        for cpus in (None, "8"):
            with self.subTest(cpus=cpus):
                environment = self.slurm_environment()
                if cpus is not None:
                    environment["SLURM_CPUS_PER_TASK"] = cpus
                report = self.prepare_slurm(environment)
                dr = FakeDrJit()
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.preview.configure_drjit_threads(dr, report)
                expected = 1 if cpus is None else 8
                self.assertEqual(dr.requests, [expected])
                self.assertEqual(report["cpu_threads"]["actual"], expected)

    def test_slurm_drjit_may_use_fewer_threads_than_quota(self):
        environment = self.slurm_environment(SLURM_CPUS_PER_TASK="8")
        report = self.prepare_slurm(environment)
        dr = FakeDrJit(forced_after_set=2)
        with mock.patch.dict(os.environ, environment, clear=True):
            self.preview.configure_drjit_threads(dr, report)
        self.assertEqual(dr.requests, [8])
        self.assertEqual(report["cpu_threads"]["actual"], 2)

    def test_slurm_drjit_must_not_exceed_quota(self):
        environment = self.slurm_environment(SLURM_CPUS_PER_TASK="8")
        report = self.prepare_slurm(environment)
        dr = FakeDrJit(forced_after_set=9)
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(SETUP_ERRORS):
                self.preview.configure_drjit_threads(dr, report)
        self.assertEqual(dr.requests, [8])

    def test_local_drjit_threads_are_read_without_changes(self):
        report = {"gpu": {"selection": "local"}}
        dr = FakeDrJit(actual=24)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.preview.configure_drjit_threads(dr, report)
        self.assertEqual(dr.requests, [])
        self.assertEqual(report["cpu_threads"]["actual"], 24)

    def test_slurm_cuda_device_accepts_nonzero_allocated_logical_index(self):
        for device in ((2, 0), (2, 1)):
            with self.subTest(device=device):
                environment = self.slurm_environment(f"{GPU_UUID},{MIG_UUID}")
                report = self.prepare_slurm(environment)
                array = FakeDeviceArray(device)
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.preview.verify_cuda_device(array, report)
                self.assertEqual(array.queries, 1)
                self.assertEqual(report["dlpack_device"], list(device))
                self.assertEqual(report["gpu"]["process_visible_device_index"], device[1])

    def test_slurm_cuda_device_rejects_cpu_negative_and_unallocated_index(self):
        for device in ((1, 0), (2, -1), (2, 2)):
            with self.subTest(device=device):
                environment = self.slurm_environment("2,5")
                report = self.prepare_slurm(environment)
                with mock.patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(SETUP_ERRORS):
                        self.preview.verify_cuda_device(FakeDeviceArray(device), report)

    def test_local_cuda_device_requires_logical_zero(self):
        report = {"gpu": {"selection": "local", "uuid": GPU_UUID}}
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU_UUID}, clear=True):
            self.preview.verify_cuda_device(FakeDeviceArray((2, 0)), report)
            for device in ((1, 0), (2, -1), (2, 1)):
                with self.subTest(device=device):
                    with self.assertRaises(SETUP_ERRORS):
                        self.preview.verify_cuda_device(FakeDeviceArray(device), report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
