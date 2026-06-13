# Benchmarks

Performance comparison methodology, execution instructions, and result interpretation for the AI Surveillance pipeline variants.

## Overview

The benchmarking suite measures and compares performance across four pipeline configurations:

| Pipeline | Description |
|----------|-------------|
| `legacy` | Python/PyTorch/Ultralytics (CPU or GPU via PyTorch) |
| `tensorrt` | Local TensorRT engine inference (FP16/INT8) |
| `triton` | Triton Inference Server with dynamic batching |
| `deepstream` | Full DeepStream/GStreamer pipeline |

## Metrics Collected

| Metric | Unit | Description |
|--------|------|-------------|
| FPS | frames/sec | Inference throughput (excluding warmup) |
| P50 Latency | ms | Median per-frame inference time |
| P95 Latency | ms | 95th percentile latency |
| P99 Latency | ms | 99th percentile latency |
| GPU Utilization | % | Average GPU compute utilization during run |
| GPU Memory Peak | MB | Maximum GPU memory allocated |

## How to Run Benchmarks

### Quick Run

```bash
# Run all pipeline benchmarks with defaults
python benchmarks/run_benchmarks.py

# Run a specific pipeline
python benchmarks/run_benchmarks.py --pipeline tensorrt

# Custom iteration count
python benchmarks/run_benchmarks.py --pipeline tensorrt --iterations 2000 --warmup 100
```

### Programmatic Usage

```python
from metropolis.benchmark import BenchmarkRunner

runner = BenchmarkRunner(output_dir="benchmarks/results")

# Run inference benchmark for a single pipeline
result = runner.run_inference_benchmark(
    pipeline="tensorrt",
    dataset="test_images/",
    num_iterations=1000,
)

print(f"FPS: {result.fps:.1f}")
print(f"P50: {result.p50_ms:.2f} ms")
print(f"P95: {result.p95_ms:.2f} ms")
print(f"P99: {result.p99_ms:.2f} ms")
print(f"GPU Util: {result.gpu_utilization:.1f}%")
print(f"GPU Memory: {result.gpu_memory_peak_mb:.0f} MB")
```

### End-to-End Benchmark

```python
# Benchmark full pipeline with real video input
e2e_result = runner.run_e2e_benchmark(
    video_source="rtsp://camera:554/stream",
    duration_secs=60.0,
)
```

### Compare Pipelines

```python
results = []
for pipeline in ["legacy", "tensorrt", "triton", "deepstream"]:
    result = runner.run_inference_benchmark(pipeline=pipeline, dataset="test_images/")
    results.append(result)

report = runner.compare(results)
print(report.markdown_table)
```

## Expected Results Format

### JSON Output

Results are saved to `benchmarks/results/<pipeline>_<timestamp>.json`:

```json
{
    "pipeline": "tensorrt",
    "precision": "fp16",
    "batch_size": 8,
    "num_iterations": 1000,
    "warmup_iterations": 50,
    "fps": 142.5,
    "p50_ms": 6.8,
    "p95_ms": 7.9,
    "p99_ms": 9.2,
    "gpu_utilization": 78.3,
    "gpu_memory_peak_mb": 2048,
    "timestamp": "2024-01-15T10:30:00Z",
    "hardware": {
        "gpu": "NVIDIA RTX 4090",
        "cuda_version": "12.2",
        "tensorrt_version": "8.6.1"
    }
}
```

### Markdown Comparison Table

The `compare()` method generates a table like:

```markdown
| Pipeline   | FPS    | P50 (ms) | P95 (ms) | P99 (ms) | GPU Util | Memory (MB) |
|------------|--------|----------|----------|----------|----------|-------------|
| legacy     | 28.3   | 35.2     | 42.1     | 48.7     | 45%      | 3200        |
| tensorrt   | 142.5  | 6.8      | 7.9      | 9.2      | 78%      | 2048        |
| triton     | 135.0  | 7.2      | 8.5      | 10.1     | 75%      | 2100        |
| deepstream | 165.0  | 5.9      | 6.8      | 7.5      | 82%      | 2500        |
```

## Regression Detection

The benchmark suite compares results against stored baselines to detect performance regressions.

### How It Works

1. A baseline result is stored as JSON in `benchmarks/results/<pipeline>_baseline.json`
2. When a new benchmark runs, it compares FPS against the baseline
3. If FPS drops more than **10%** from baseline, the benchmark flags a regression

### Save a Baseline

```python
runner = BenchmarkRunner()
result = runner.run_inference_benchmark(pipeline="tensorrt", dataset="test_images/")

# Save as the baseline for future comparisons
runner.save_baseline(result)
```

### Check for Regression

```python
# Run benchmark and check against baseline
result = runner.run_inference_benchmark(pipeline="tensorrt", dataset="test_images/")
regression = runner.check_regression(result)

if regression["is_regression"]:
    print(f"REGRESSION DETECTED: FPS dropped {regression['fps_drop_pct']:.1f}%")
    print(f"  Baseline: {regression['baseline_fps']:.1f} FPS")
    print(f"  Current:  {regression['current_fps']:.1f} FPS")
else:
    print("No regression detected")
```

### CI Integration

The CI pipeline runs regression checks automatically:

```yaml
# In .github/workflows/metropolis-ci.yml
- name: Benchmark Regression Check
  run: python benchmarks/run_benchmarks.py --check-regression --pipeline tensorrt
```

A regression failure exits with code 1, failing the CI job.

## How to Update Baselines

Baselines should be updated when:
- Hardware changes (new GPU, driver update)
- Intentional model changes (different architecture, precision)
- TensorRT version upgrades

```bash
# Re-run benchmark and save as new baseline
python benchmarks/run_benchmarks.py --pipeline tensorrt --save-baseline

# Or programmatically
python -c "
from metropolis.benchmark import BenchmarkRunner
runner = BenchmarkRunner()
result = runner.run_inference_benchmark(pipeline='tensorrt', dataset='test_images/', num_iterations=2000)
runner.save_baseline(result)
print(f'Baseline saved: {result.fps:.1f} FPS')
"
```

## Reproducibility

To ensure reproducible results:

1. **Warmup**: The first N iterations (default 50) are excluded from measurements to allow GPU frequency scaling and cache warming
2. **Iteration count**: Use at least 1000 iterations for stable statistics
3. **Isolation**: Close other GPU-intensive applications during benchmarking
4. **Consistency**: Running the same benchmark twice on identical hardware should produce FPS within 5% of each other
5. **Fixed seed**: Use `--hypothesis-seed=0` for property-based tests in CI

## Methodology Notes

- **Timing**: Per-frame latency is measured using `time.perf_counter()` around the inference call
- **GPU monitoring**: Utilization is sampled via `pynvml` at 100ms intervals during the benchmark
- **Percentiles**: Computed from the full timing array after excluding warmup iterations
- **FPS calculation**: `num_iterations / total_elapsed_time` (wall clock)
- **Batch effects**: Throughput is reported per-frame even when batching is used
