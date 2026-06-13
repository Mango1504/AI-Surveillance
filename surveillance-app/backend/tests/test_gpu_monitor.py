"""Tests for GPUMonitor class in the benchmarking suite.

Tests verify the GPUMonitor's background sampling, aggregation logic,
and graceful fallback behavior when GPU tools are unavailable.
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.benchmark import GPUMonitor


class TestGPUMonitorStartStop:
    """Test GPUMonitor lifecycle (start/stop) behavior."""

    def test_stop_returns_zeros_when_no_gpu_available(self):
        """When neither pynvml nor nvidia-smi is available, returns (0.0, 0.0)."""
        monitor = GPUMonitor(interval_secs=0.05)

        with patch.object(monitor, "_init_pynvml", return_value=False), \
             patch.object(monitor, "_read_nvidia_smi", return_value=(None, 0.0)):
            monitor.start()
            time.sleep(0.15)
            avg_util, peak_mem = monitor.stop()

        assert avg_util == 0.0
        assert peak_mem == 0.0

    def test_stop_returns_average_utilization_and_peak_memory(self):
        """Verify average utilization and peak memory are computed correctly."""
        monitor = GPUMonitor(interval_secs=0.05)

        # Mock samples: utilization [40, 60, 80], memory [100, 200, 150]
        sample_data = iter([(40.0, 100.0), (60.0, 200.0), (80.0, 150.0)])

        def mock_read():
            try:
                return next(sample_data)
            except StopIteration:
                return (None, 0.0)

        with patch.object(monitor, "_init_pynvml", return_value=False), \
             patch.object(monitor, "_read_gpu_sample", side_effect=mock_read):
            monitor._stop_event.clear()
            monitor._utilization_samples = []
            monitor._memory_samples = []
            monitor._use_pynvml = False

            # Manually run the sample loop a few times
            for _ in range(3):
                utilization, memory_mb = monitor._read_gpu_sample()
                if utilization is not None:
                    monitor._utilization_samples.append(utilization)
                    monitor._memory_samples.append(memory_mb)

        # Compute expected values
        avg_util = sum(monitor._utilization_samples) / len(monitor._utilization_samples)
        peak_mem = max(monitor._memory_samples)

        assert avg_util == pytest.approx(60.0)
        assert peak_mem == pytest.approx(200.0)

    def test_start_creates_daemon_thread(self):
        """Verify start() creates a daemon thread that can be stopped."""
        monitor = GPUMonitor(interval_secs=0.05)

        with patch.object(monitor, "_init_pynvml", return_value=False), \
             patch.object(monitor, "_read_nvidia_smi", return_value=(None, 0.0)):
            monitor.start()
            assert monitor._thread is not None
            assert monitor._thread.is_alive()
            assert monitor._thread.daemon is True
            monitor.stop()
            assert monitor._thread is None

    def test_stop_without_start_returns_zeros(self):
        """Calling stop() without start() should return (0.0, 0.0) safely."""
        monitor = GPUMonitor()
        avg_util, peak_mem = monitor.stop()
        assert avg_util == 0.0
        assert peak_mem == 0.0


class TestGPUMonitorPynvml:
    """Test GPUMonitor with mocked pynvml backend."""

    def test_uses_pynvml_when_available(self):
        """When pynvml initializes successfully, it should be used for sampling."""
        monitor = GPUMonitor(interval_secs=0.05)

        mock_util = MagicMock()
        mock_util.gpu = 75
        mock_mem = MagicMock()
        mock_mem.used = 2048 * 1024 * 1024  # 2048 MB

        # Create a mock pynvml module and inject it into sys.modules
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex = MagicMock(return_value="handle")
        mock_pynvml.nvmlDeviceGetUtilizationRates = MagicMock(return_value=mock_util)
        mock_pynvml.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mock_mem)
        mock_pynvml.nvmlShutdown = MagicMock()

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            monitor.start()
            time.sleep(0.2)
            avg_util, peak_mem = monitor.stop()

        assert avg_util == pytest.approx(75.0)
        assert peak_mem == pytest.approx(2048.0)


class TestGPUMonitorNvidiaSmi:
    """Test GPUMonitor with nvidia-smi subprocess fallback."""

    def test_parses_nvidia_smi_output(self):
        """Verify nvidia-smi CSV output is parsed correctly."""
        monitor = GPUMonitor(interval_secs=0.05)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "65, 4096\n"

        with patch.object(monitor, "_init_pynvml", return_value=False), \
             patch("subprocess.run", return_value=mock_result):
            monitor.start()
            time.sleep(0.2)
            avg_util, peak_mem = monitor.stop()

        assert avg_util == pytest.approx(65.0)
        assert peak_mem == pytest.approx(4096.0)

    def test_handles_nvidia_smi_not_found(self):
        """When nvidia-smi is not found, returns (None, 0.0) per sample."""
        monitor = GPUMonitor(interval_secs=0.05)

        with patch.object(monitor, "_init_pynvml", return_value=False), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            monitor.start()
            time.sleep(0.15)
            avg_util, peak_mem = monitor.stop()

        assert avg_util == 0.0
        assert peak_mem == 0.0

    def test_handles_nvidia_smi_timeout(self):
        """When nvidia-smi times out, returns (None, 0.0) per sample."""
        import subprocess as sp
        monitor = GPUMonitor(interval_secs=0.05)

        with patch.object(monitor, "_init_pynvml", return_value=False), \
             patch("subprocess.run", side_effect=sp.TimeoutExpired("nvidia-smi", 2.0)):
            monitor.start()
            time.sleep(0.15)
            avg_util, peak_mem = monitor.stop()

        assert avg_util == 0.0
        assert peak_mem == 0.0
