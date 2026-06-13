#!/usr/bin/env python
"""CLI entry point for running the full benchmark suite.

Runs inference benchmarks across pipeline configurations, generates
comparison reports, checks for regressions against stored baselines,
and optionally saves new baselines.

Usage:
    python benchmarks/run_benchmarks.py --dataset data/test_images
    python benchmarks/run_benchmarks.py --dataset data/test_images --pipelines tensorrt,triton
    python benchmarks/run_benchmarks.py --dataset data/test_images --compare --check-regression
    python benchmarks/run_benchmarks.py --dataset data/test_images --save-baseline
"""

import argparse
import sys
import os

# Add project root to path so metropolis package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "surveillance-app", "backend"))

from metropolis.benchmark import BenchmarkRunner


ALL_PIPELINES = ["legacy", "tensorrt", "triton", "deepstream"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full benchmark suite across pipeline configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --dataset data/test_images
  %(prog)s --dataset data/test_images --pipelines tensorrt,triton
  %(prog)s --dataset data/test_images --compare --check-regression
  %(prog)s --dataset data/test_images --save-baseline
""",
    )

    parser.add_argument(
        "--pipelines",
        type=str,
        default=None,
        help=(
            "Comma-separated list of pipelines to benchmark. "
            f"Options: {', '.join(ALL_PIPELINES)}. "
            "Default: all pipelines."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to test images directory.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of measurement iterations (default: 1000).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results (default: benchmarks/results).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison report after all benchmarks complete.",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Check for regressions against stored baselines.",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baselines.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the benchmark suite and return exit code.

    Returns:
        0 on success, 1 if regressions are detected.
    """
    args = parse_args()

    # Determine which pipelines to benchmark
    if args.pipelines is not None:
        pipelines = [p.strip() for p in args.pipelines.split(",")]
        invalid = [p for p in pipelines if p not in ALL_PIPELINES]
        if invalid:
            print(f"Error: Unknown pipeline(s): {', '.join(invalid)}", file=sys.stderr)
            print(f"Valid pipelines: {', '.join(ALL_PIPELINES)}", file=sys.stderr)
            return 1
    else:
        pipelines = list(ALL_PIPELINES)

    # Validate dataset path
    if not os.path.isdir(args.dataset):
        print(f"Error: Dataset directory not found: {args.dataset}", file=sys.stderr)
        return 1

    print(f"Benchmark Suite")
    print(f"{'=' * 60}")
    print(f"Pipelines:  {', '.join(pipelines)}")
    print(f"Dataset:    {args.dataset}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup:     {args.warmup}")
    print(f"Output:     {args.output_dir}")
    print(f"{'=' * 60}\n")

    # Initialize the benchmark runner
    runner = BenchmarkRunner(output_dir=args.output_dir)

    # Run benchmarks for each pipeline
    results = []
    regression_detected = False

    for pipeline in pipelines:
        print(f"\n{'─' * 60}")
        print(f"Running benchmark: {pipeline}")
        print(f"{'─' * 60}")

        try:
            result = runner.run_inference_benchmark(
                pipeline=pipeline,
                dataset=args.dataset,
                num_iterations=args.iterations,
                warmup_iterations=args.warmup,
            )
            results.append(result)

            # Print result summary
            print(f"\n  Results for '{pipeline}':")
            print(f"    FPS:          {result.fps:.1f}")
            print(f"    Latency P50:  {result.latency_p50_ms:.2f} ms")
            print(f"    Latency P95:  {result.latency_p95_ms:.2f} ms")
            print(f"    Latency P99:  {result.latency_p99_ms:.2f} ms")
            print(f"    GPU Util:     {result.gpu_utilization_pct:.1f}%")
            print(f"    GPU Memory:   {result.gpu_memory_mb:.0f} MB")

            # Save baseline if requested
            if args.save_baseline:
                baseline_path = runner.save_baseline(result)
                print(f"    Baseline saved: {baseline_path}")

            # Check regression if requested
            if args.check_regression:
                regression_info = runner.check_regression(result)
                if regression_info["has_regression"]:
                    regression_detected = True
                    print(f"\n  ⚠ REGRESSION DETECTED for '{pipeline}':")
                    for reg in regression_info["regressions"]:
                        print(f"    - {reg}")
                else:
                    if regression_info["baseline"]:
                        print(f"    ✓ No regression (baseline exists)")
                    else:
                        print(f"    ⊘ No baseline found, skipping regression check")

        except Exception as e:
            print(f"\n  ✗ Error benchmarking '{pipeline}': {e}", file=sys.stderr)
            continue

    # Generate comparison report if requested
    if args.compare and len(results) > 1:
        print(f"\n\n{'=' * 60}")
        print("Comparison Report")
        print(f"{'=' * 60}\n")

        report = runner.compare(results)
        print(report.report_markdown)
        print(f"\n  Best FPS:     {report.best_fps_pipeline}")
        print(f"  Best Latency: {report.best_latency_pipeline}")

        # Save comparison report to output directory
        report_path = os.path.join(args.output_dir, "comparison_report.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report.report_json)
        print(f"\n  Report saved: {report_path}")

    elif args.compare and len(results) <= 1:
        print("\n  Comparison requires at least 2 pipeline results. Skipping.")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Benchmark suite complete. {len(results)}/{len(pipelines)} pipelines ran successfully.")

    if regression_detected:
        print("\n⚠ REGRESSIONS DETECTED — exiting with code 1.")
        return 1

    print("✓ All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
