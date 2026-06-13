"""Tests for regression detection in the BenchmarkRunner.

Tests verify that save_baseline, load_baseline, and check_regression
correctly persist baselines and detect performance regressions exceeding
the configured threshold (default 10%).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.benchmark import BenchmarkResult, BenchmarkRunner


def _make_result(
    pipeline_name: str = "tensorrt",
    fps: float = 300.0,
    latency_p50_ms: float = 3.3,
    latency_p95_ms: float = 5.0,
    latency_p99_ms: float = 7.0,
    gpu_utilization_pct: float = 75.0,
    gpu_memory_mb: float = 4096.0,
    num_iterations: int = 1000,
    warmup_iterations: int = 50,
) -> BenchmarkResult:
    """Helper to create a BenchmarkResult with sensible defaults."""
    return BenchmarkResult(
        pipeline_name=pipeline_name,
        fps=fps,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        latency_p99_ms=latency_p99_ms,
        gpu_utilization_pct=gpu_utilization_pct,
        gpu_memory_mb=gpu_memory_mb,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
        timestamp="2024-01-15T12:00:00+00:00",
    )


class TestSaveBaseline:
    """Tests for BenchmarkRunner.save_baseline()."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.runner = BenchmarkRunner(output_dir=self.tmp_dir)

    def test_save_baseline_creates_json_file(self):
        """save_baseline should create a JSON file in the output directory."""
        result = _make_result("tensorrt")
        filepath = self.runner.save_baseline(result)

        assert os.path.exists(filepath)
        assert filepath.endswith("tensorrt_baseline.json")

    def test_save_baseline_default_filename(self):
        """Default filename should be {pipeline_name}_baseline.json."""
        result = _make_result("triton")
        filepath = self.runner.save_baseline(result)

        expected_path = os.path.join(self.tmp_dir, "triton_baseline.json")
        assert filepath == expected_path

    def test_save_baseline_custom_filename(self):
        """Custom filename should be used when provided."""
        result = _make_result("tensorrt")
        filepath = self.runner.save_baseline(result, filename="custom_baseline.json")

        expected_path = os.path.join(self.tmp_dir, "custom_baseline.json")
        assert filepath == expected_path

    def test_save_baseline_content_is_valid_json(self):
        """Saved file should contain valid JSON matching the result."""
        result = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        filepath = self.runner.save_baseline(result)

        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["pipeline_name"] == "tensorrt"
        assert data["fps"] == 300.0
        assert data["latency_p95_ms"] == 5.0
        assert data["num_iterations"] == 1000

    def test_save_baseline_overwrites_existing(self):
        """Saving a baseline should overwrite any existing baseline file."""
        result1 = _make_result("tensorrt", fps=300.0)
        result2 = _make_result("tensorrt", fps=350.0)

        self.runner.save_baseline(result1)
        filepath = self.runner.save_baseline(result2)

        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["fps"] == 350.0


class TestLoadBaseline:
    """Tests for BenchmarkRunner.load_baseline()."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.runner = BenchmarkRunner(output_dir=self.tmp_dir)

    def test_load_baseline_returns_none_when_no_file(self):
        """load_baseline should return None if no baseline file exists."""
        result = self.runner.load_baseline("nonexistent_pipeline")
        assert result is None

    def test_load_baseline_returns_benchmark_result(self):
        """load_baseline should return a BenchmarkResult matching saved data."""
        original = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(original)

        loaded = self.runner.load_baseline("tensorrt")

        assert loaded is not None
        assert loaded.pipeline_name == "tensorrt"
        assert loaded.fps == 300.0
        assert loaded.latency_p95_ms == 5.0
        assert loaded.num_iterations == 1000
        assert loaded.warmup_iterations == 50

    def test_load_baseline_roundtrip_all_fields(self):
        """All fields should survive a save/load roundtrip."""
        original = _make_result(
            pipeline_name="deepstream",
            fps=450.0,
            latency_p50_ms=2.2,
            latency_p95_ms=3.5,
            latency_p99_ms=4.8,
            gpu_utilization_pct=85.0,
            gpu_memory_mb=6000.0,
            num_iterations=2000,
            warmup_iterations=100,
        )
        self.runner.save_baseline(original)
        loaded = self.runner.load_baseline("deepstream")

        assert loaded.pipeline_name == original.pipeline_name
        assert loaded.fps == original.fps
        assert loaded.latency_p50_ms == original.latency_p50_ms
        assert loaded.latency_p95_ms == original.latency_p95_ms
        assert loaded.latency_p99_ms == original.latency_p99_ms
        assert loaded.gpu_utilization_pct == original.gpu_utilization_pct
        assert loaded.gpu_memory_mb == original.gpu_memory_mb
        assert loaded.num_iterations == original.num_iterations
        assert loaded.warmup_iterations == original.warmup_iterations


class TestCheckRegression:
    """Tests for BenchmarkRunner.check_regression()."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.runner = BenchmarkRunner(output_dir=self.tmp_dir)

    def test_no_baseline_returns_no_regression(self):
        """When no baseline exists, check_regression should report no regression."""
        result = _make_result("tensorrt", fps=300.0)
        check = self.runner.check_regression(result)

        assert check["has_regression"] is False
        assert check["regressions"] == []
        assert check["baseline"] == {}

    def test_no_regression_when_within_threshold(self):
        """No regression when current is within 10% of baseline."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # FPS dropped 5% (within 10% threshold), latency increased 5%
        current = _make_result("tensorrt", fps=285.0, latency_p95_ms=5.25)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is False
        assert check["regressions"] == []

    def test_fps_regression_detected(self):
        """FPS regression should be detected when drop exceeds threshold."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # FPS dropped 15% (exceeds 10% threshold)
        current = _make_result("tensorrt", fps=255.0, latency_p95_ms=5.0)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is True
        assert len(check["regressions"]) == 1
        assert "FPS regression" in check["regressions"][0]

    def test_latency_regression_detected(self):
        """Latency regression should be detected when P95 increase exceeds threshold."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # Latency increased 15% (exceeds 10% threshold)
        current = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.75)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is True
        assert len(check["regressions"]) == 1
        assert "Latency P95 regression" in check["regressions"][0]

    def test_both_regressions_detected(self):
        """Both FPS and latency regressions should be reported together."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # Both FPS dropped 20% and latency increased 25%
        current = _make_result("tensorrt", fps=240.0, latency_p95_ms=6.25)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is True
        assert len(check["regressions"]) == 2

    def test_exact_threshold_boundary_no_regression(self):
        """At exactly the threshold boundary, no regression should be flagged."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # FPS at exactly 10% below: 300 * 0.9 = 270 (not below, so no regression)
        # Latency at exactly 10% above: 5.0 * 1.1 = 5.5 (not above, so no regression)
        current = _make_result("tensorrt", fps=270.0, latency_p95_ms=5.5)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is False

    def test_just_below_threshold_triggers_regression(self):
        """Just below the FPS threshold should trigger a regression."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # FPS just below threshold: 269.9 < 270.0
        current = _make_result("tensorrt", fps=269.9, latency_p95_ms=5.0)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is True
        assert "FPS regression" in check["regressions"][0]

    def test_just_above_latency_threshold_triggers_regression(self):
        """Just above the latency threshold should trigger a regression."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # Latency just above threshold: 5.501 > 5.5
        current = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.501)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is True
        assert "Latency P95 regression" in check["regressions"][0]

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # 8% FPS drop - within default 10% but outside 5% threshold
        current = _make_result("tensorrt", fps=276.0, latency_p95_ms=5.0)

        # With default 10% threshold: no regression
        check_default = self.runner.check_regression(current, threshold=0.10)
        assert check_default["has_regression"] is False

        # With stricter 5% threshold: regression detected
        check_strict = self.runner.check_regression(current, threshold=0.05)
        assert check_strict["has_regression"] is True

    def test_result_dict_contains_current_and_baseline(self):
        """Return dict should contain both current and baseline values."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        current = _make_result("tensorrt", fps=290.0, latency_p95_ms=5.2)
        check = self.runner.check_regression(current)

        assert check["current"]["fps"] == 290.0
        assert check["current"]["latency_p95_ms"] == 5.2
        assert check["baseline"]["fps"] == 300.0
        assert check["baseline"]["latency_p95_ms"] == 5.0

    def test_improved_performance_no_regression(self):
        """Better performance than baseline should not flag regression."""
        baseline = _make_result("tensorrt", fps=300.0, latency_p95_ms=5.0)
        self.runner.save_baseline(baseline)

        # Performance improved: higher FPS, lower latency
        current = _make_result("tensorrt", fps=350.0, latency_p95_ms=4.0)
        check = self.runner.check_regression(current)

        assert check["has_regression"] is False
        assert check["regressions"] == []
