"""Tests for the BenchmarkRunner.compare() method.

Tests verify that compare() correctly identifies best pipelines,
generates valid JSON and markdown reports, and handles edge cases.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.benchmark import BenchmarkResult, BenchmarkRunner, ComparisonReport


def _make_result(
    pipeline_name: str,
    fps: float = 100.0,
    latency_p50_ms: float = 10.0,
    latency_p95_ms: float = 12.0,
    latency_p99_ms: float = 15.0,
    gpu_utilization_pct: float = 50.0,
    gpu_memory_mb: float = 2048.0,
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
        num_iterations=1000,
        warmup_iterations=50,
        timestamp="2024-01-01T00:00:00+00:00",
    )


class TestCompareMethod:
    """Tests for BenchmarkRunner.compare()."""

    def setup_method(self):
        self.runner = BenchmarkRunner(output_dir="benchmarks/results")

    def test_empty_results_raises_value_error(self):
        """compare() with empty list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            self.runner.compare([])

    def test_single_result_is_valid(self):
        """compare() with a single result should work and identify it as best."""
        result = _make_result("tensorrt", fps=300.0, latency_p50_ms=3.3)
        report = self.runner.compare([result])

        assert isinstance(report, ComparisonReport)
        assert report.best_fps_pipeline == "tensorrt"
        assert report.best_latency_pipeline == "tensorrt"
        assert len(report.results) == 1

    def test_best_fps_pipeline_identified(self):
        """compare() should identify the pipeline with highest FPS."""
        results = [
            _make_result("legacy", fps=100.0),
            _make_result("tensorrt", fps=300.0),
            _make_result("triton", fps=250.0),
        ]
        report = self.runner.compare(results)
        assert report.best_fps_pipeline == "tensorrt"

    def test_best_latency_pipeline_identified(self):
        """compare() should identify the pipeline with lowest P50 latency."""
        results = [
            _make_result("legacy", latency_p50_ms=10.0),
            _make_result("tensorrt", latency_p50_ms=3.3),
            _make_result("triton", latency_p50_ms=4.0),
        ]
        report = self.runner.compare(results)
        assert report.best_latency_pipeline == "tensorrt"

    def test_json_report_is_valid_json(self):
        """report_json should be valid JSON."""
        results = [
            _make_result("legacy", fps=100.0, latency_p50_ms=10.0),
            _make_result("tensorrt", fps=300.0, latency_p50_ms=3.3),
        ]
        report = self.runner.compare(results)
        data = json.loads(report.report_json)

        assert "comparison" in data
        assert "results" in data
        assert data["comparison"]["best_fps_pipeline"] == "tensorrt"
        assert data["comparison"]["best_latency_pipeline"] == "tensorrt"
        assert data["comparison"]["num_pipelines"] == 2

    def test_json_report_contains_speedup_ratios(self):
        """JSON report should include speedup ratios relative to slowest pipeline."""
        results = [
            _make_result("legacy", fps=100.0),
            _make_result("tensorrt", fps=300.0),
        ]
        report = self.runner.compare(results)
        data = json.loads(report.report_json)

        speedup = data["comparison"]["speedup_ratios"]
        assert speedup["legacy"] == 1.0
        assert speedup["tensorrt"] == 3.0

    def test_json_report_contains_all_result_dicts(self):
        """JSON report results array should contain to_dict() for each result."""
        results = [
            _make_result("legacy"),
            _make_result("tensorrt"),
        ]
        report = self.runner.compare(results)
        data = json.loads(report.report_json)

        assert len(data["results"]) == 2
        assert data["results"][0]["pipeline_name"] == "legacy"
        assert data["results"][1]["pipeline_name"] == "tensorrt"

    def test_markdown_report_contains_header(self):
        """Markdown report should contain the table header row."""
        results = [_make_result("legacy")]
        report = self.runner.compare(results)

        assert "| Pipeline | FPS | P50 (ms) | P95 (ms) | P99 (ms) | GPU Util (%) | GPU Mem (MB) |" in report.report_markdown
        assert "|----------|-----|----------|----------|----------|--------------|--------------|" in report.report_markdown

    def test_markdown_report_contains_pipeline_rows(self):
        """Markdown report should contain a row for each pipeline."""
        results = [
            _make_result("legacy", fps=100.0, latency_p50_ms=10.0),
            _make_result("tensorrt", fps=300.0, latency_p50_ms=3.3),
        ]
        report = self.runner.compare(results)

        assert "| legacy |" in report.report_markdown
        assert "| tensorrt |" in report.report_markdown

    def test_comparison_report_dataclass_fields(self):
        """ComparisonReport should have all expected fields populated."""
        results = [
            _make_result("legacy", fps=100.0, latency_p50_ms=10.0),
            _make_result("tensorrt", fps=300.0, latency_p50_ms=3.3),
        ]
        report = self.runner.compare(results)

        assert report.results == results
        assert isinstance(report.best_fps_pipeline, str)
        assert isinstance(report.best_latency_pipeline, str)
        assert isinstance(report.report_json, str)
        assert isinstance(report.report_markdown, str)
