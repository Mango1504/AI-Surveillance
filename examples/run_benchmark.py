"""Example: Run performance benchmarks across pipeline configurations.

Demonstrates:
  - Running inference benchmarks for different pipelines
  - Comparing results across configurations
  - Saving baselines and checking for regressions
  - Generating markdown comparison tables

Prerequisites:
  - pip install numpy
  - For GPU benchmarks: NVIDIA GPU + TensorRT
  - For Triton benchmarks: Running Triton server

Usage:
  python examples/run_benchmark.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "surveillance-app" / "backend"))

from metropolis.benchmark import BenchmarkRunner, calculate_latency_percentiles


def main():
    print("=== Performance Benchmark Suite ===")
    print()

    # Initialize runner
    output_dir = "benchmarks/results"
    runner = BenchmarkRunner(output_dir=output_dir)
    print(f"Output directory: {output_dir}")
    print()

    # Define pipelines to benchmark
    # In practice, only pipelines with available hardware will succeed
    pipelines = ["legacy", "tensorrt", "triton", "deepstream"]
    dataset = "test_images/"
    num_iterations = 100  # Use 1000+ for production benchmarks

    print(f"Dataset: {dataset}")
    print(f"Iterations: {num_iterations}")
    print(f"Pipelines: {pipelines}")
    print()

    # Run benchmarks
    results = []
    for pipeline in pipelines:
        print(f"--- Benchmarking: {pipeline} ---")
        try:
            result = runner.run_inference_benchmark(
                pipeline=pipeline,
                dataset=dataset,
                num_iterations=num_iterations,
            )
            results.append(result)
            print(f"  FPS: {result.fps:.1f}")
            print(f"  P50 latency: {result.p50_ms:.2f} ms")
            print(f"  P95 latency: {result.p95_ms:.2f} ms")
            print(f"  P99 latency: {result.p99_ms:.2f} ms")
            print(f"  GPU utilization: {result.gpu_utilization:.1f}%")
            print(f"  GPU memory peak: {result.gpu_memory_peak_mb:.0f} MB")
        except Exception as e:
            print(f"  Skipped ({e})")
        print()

    # Generate comparison report
    if len(results) >= 2:
        print("--- Comparison Report ---")
        report = runner.compare(results)
        print(report.markdown_table)
        print()

    # Save baseline (using first successful result)
    if results:
        best = max(results, key=lambda r: r.fps)
        print(f"--- Saving Baseline ---")
        print(f"Best pipeline: {best.pipeline} ({best.fps:.1f} FPS)")
        runner.save_baseline(best)
        print(f"Baseline saved to {output_dir}/")
        print()

        # Check regression against baseline
        print("--- Regression Check ---")
        regression = runner.check_regression(best)
        if regression["is_regression"]:
            print(f"  REGRESSION: FPS dropped {regression['fps_drop_pct']:.1f}%")
        else:
            print(f"  No regression detected")
        print()

    # Demonstrate latency percentile calculation
    print("--- Latency Percentile Utility ---")
    sample_timings = [5.2, 6.1, 5.8, 7.3, 5.5, 6.0, 5.9, 8.1, 5.7, 6.2]
    p50, p95, p99 = calculate_latency_percentiles(sample_timings)
    print(f"  Sample timings: {sample_timings}")
    print(f"  P50: {p50:.2f} ms")
    print(f"  P95: {p95:.2f} ms")
    print(f"  P99: {p99:.2f} ms")

    print()
    print("=== Benchmark Complete ===")


if __name__ == "__main__":
    main()
