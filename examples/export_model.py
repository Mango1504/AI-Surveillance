"""Example: Export a YOLOv8 model to TensorRT engine.

Demonstrates the full export pipeline:
  1. PyTorch .pt → ONNX (opset 17, dynamic batch)
  2. ONNX → TensorRT engine (FP16 precision)
  3. Accuracy validation against PyTorch baseline

Prerequisites:
  - NVIDIA GPU with TensorRT installed
  - YOLOv8 model file (e.g., yolov8m.pt)
  - pip install ultralytics tensorrt pycuda

Usage:
  python examples/export_model.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "surveillance-app" / "backend"))

from metropolis.export_tensorrt import TensorRTExporter


def main():
    # Configuration
    model_path = "models/yolov8m.pt"
    output_dir = "models/exported"
    precision = "fp16"  # Options: "fp32", "fp16", "int8"
    max_batch_size = 8

    print(f"=== TensorRT Export Pipeline ===")
    print(f"Model: {model_path}")
    print(f"Precision: {precision}")
    print(f"Max batch size: {max_batch_size}")
    print()

    # Initialize exporter
    exporter = TensorRTExporter(model_path, output_dir)

    # Step 1: Export to ONNX
    print("[1/3] Exporting to ONNX format...")
    onnx_path = exporter.export_onnx(opset=17, dynamic_batch=True)
    print(f"  ONNX saved: {onnx_path}")
    print()

    # Step 2: Build TensorRT engine
    print(f"[2/3] Building TensorRT engine ({precision})...")
    engine_path = exporter.build_engine(
        precision=precision,
        max_batch_size=max_batch_size,
        workspace_mb=4096,
    )
    print(f"  Engine saved: {engine_path}")
    print()

    # Step 3: Validate accuracy (optional, requires test images)
    test_images_dir = Path("test_images")
    if test_images_dir.exists():
        test_images = [str(p) for p in test_images_dir.glob("*.jpg")][:10]
        if test_images:
            print("[3/3] Validating engine accuracy...")
            metrics = exporter.validate(test_images=test_images)
            print(f"  Engine mAP: {metrics['mAP']:.4f}")
            print(f"  Baseline mAP: {metrics['baseline_mAP']:.4f}")
            print(f"  mAP drop: {metrics['mAP_drop']:.4f}")
            print(f"  Status: {'PASS' if metrics['engine_matches_baseline'] else 'FAIL'}")
        else:
            print("[3/3] Skipped validation (no test images found)")
    else:
        print("[3/3] Skipped validation (test_images/ directory not found)")

    print()
    print("=== Export Complete ===")
    print(f"Engine ready at: {engine_path}")


if __name__ == "__main__":
    main()
