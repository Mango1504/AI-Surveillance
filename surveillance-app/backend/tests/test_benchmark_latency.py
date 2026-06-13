"""Tests for latency percentile calculation in the benchmarking suite.

Tests verify that calculate_latency_percentiles() correctly computes
P50, P95, and P99 values from timing data, and that the full benchmark
run produces valid percentile ordering.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.benchmark import BenchmarkRunner, calculate_latency_percentiles


class TestCalculateLatencyPercentiles:
    """Tests for the standalone calculate_latency_percentiles helper."""

    def test_p50_is_median_for_known_distribution(self):
        """P50 should equal the median for a simple ordered list."""
        # Odd-length list: median is the middle value
        timings = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        p50, _, _ = calculate_latency_percentiles(timings)
        assert p50 == pytest.approx(6.0)

        # Even-length list: median is average of two middle values
        timings_even = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p50_even, _, _ = calculate_latency_percentiles(timings_even)
        assert p50_even == pytest.approx(5.5)

    def test_p95_and_p99_correct_for_known_distribution(self):
        """P95 and P99 should be correct for a known 100-element distribution."""
        # Use 1..100 so percentiles are straightforward
        timings = [float(i) for i in range(1, 101)]
        p50, p95, p99 = calculate_latency_percentiles(timings)

        # numpy percentile uses linear interpolation by default
        expected_p50 = float(np.percentile(timings, 50))
        expected_p95 = float(np.percentile(timings, 95))
        expected_p99 = float(np.percentile(timings, 99))

        assert p50 == pytest.approx(expected_p50)
        assert p95 == pytest.approx(expected_p95)
        assert p99 == pytest.approx(expected_p99)

        # P95 should be near 95.05 and P99 near 99.01 for 1..100
        assert p95 == pytest.approx(95.05, abs=0.5)
        assert p99 == pytest.approx(99.01, abs=0.5)

    def test_single_timing_value(self):
        """With a single timing value, all percentiles should equal that value."""
        timings = [42.5]
        p50, p95, p99 = calculate_latency_percentiles(timings)
        assert p50 == pytest.approx(42.5)
        assert p95 == pytest.approx(42.5)
        assert p99 == pytest.approx(42.5)

    def test_uniform_distribution_p50_near_middle(self):
        """For a uniform distribution, P50 should be near the midpoint."""
        # Generate uniform values between 10 and 20
        np.random.seed(42)
        timings = np.random.uniform(10.0, 20.0, size=1000).tolist()
        p50, p95, p99 = calculate_latency_percentiles(timings)

        # P50 should be near 15.0 (midpoint of [10, 20])
        assert p50 == pytest.approx(15.0, abs=0.5)
        # P95 should be near 19.5
        assert p95 == pytest.approx(19.5, abs=0.5)
        # P99 should be near 19.9
        assert p99 == pytest.approx(19.9, abs=0.5)

    def test_empty_timings_raises_value_error(self):
        """An empty timings list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one value"):
            calculate_latency_percentiles([])

    def test_percentile_ordering(self):
        """P50 <= P95 <= P99 must always hold."""
        timings = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]
        p50, p95, p99 = calculate_latency_percentiles(timings)
        assert p50 <= p95 <= p99


class TestBenchmarkRunPercentiles:
    """Test that the full benchmark run produces valid percentile values."""

    def test_benchmark_run_produces_valid_percentile_ordering(self):
        """P50 <= P95 <= P99 in the BenchmarkResult from run_inference_benchmark."""
        runner = BenchmarkRunner(output_dir="benchmarks/results")

        # Use a minimal run with a simple inference function
        def fast_inference(img):
            return img.astype(np.float32) / 255.0

        result = runner.run_inference_benchmark(
            pipeline="legacy",
            dataset="nonexistent_dataset",  # Will use synthetic images
            num_iterations=100,
            warmup_iterations=5,
            inference_fn=fast_inference,
        )

        # Verify percentile ordering: P50 <= P95 <= P99
        assert result.latency_p50_ms <= result.latency_p95_ms
        assert result.latency_p95_ms <= result.latency_p99_ms

        # All latencies should be positive
        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms > 0
        assert result.latency_p99_ms > 0

        # FPS should be positive
        assert result.fps > 0
