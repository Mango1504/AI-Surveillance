"""TensorRT model export pipeline for YOLOv8 models.

Converts YOLOv8 .pt models through ONNX to optimized TensorRT engines with
FP16/INT8 quantization. Supports the full export pipeline including ONNX
conversion, engine building with precision selection, and accuracy validation
against the PyTorch baseline.

Typical usage:
    exporter = TensorRTExporter("models/yolov8m.pt", "models/output")
    onnx_path = exporter.export_onnx(opset=17, dynamic_batch=True)
    engine_path = exporter.build_engine(precision="fp16", max_batch_size=8)
    metrics = exporter.validate(test_images=["img1.jpg", "img2.jpg"])
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Create TRT_LOGGER lazily since tensorrt may not be installed
try:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_LOGGER = None


class TensorRTExporter:
    """Export YOLOv8 PyTorch models to optimized TensorRT engines.

    Handles the full .pt → ONNX → TensorRT conversion pipeline with support
    for FP16 and INT8 quantization. Provides validation against the original
    PyTorch model to ensure accuracy is preserved within acceptable thresholds.

    Attributes:
        model_path: Path to the source YOLOv8 .pt model file.
        output_dir: Directory where exported ONNX and engine files are saved.
        onnx_path: Path to the exported ONNX file, or None if not yet exported.
        engine_path: Path to the built TensorRT engine, or None if not yet built.
    """

    def __init__(self, model_path: str, output_dir: str) -> None:
        """Initialize the exporter with source model and output directory.

        Args:
            model_path: Path to the YOLOv8 .pt model file to export.
            output_dir: Directory where ONNX and TensorRT engine files will
                be written. Created if it does not exist.

        Raises:
            FileNotFoundError: If model_path does not exist.
            ValueError: If model_path does not end with .pt extension.
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )
        if model_file.suffix != ".pt":
            raise ValueError(
                f"Model file must have .pt extension, got: '{model_file.suffix}'"
            )

        self.model_path = model_path
        self.output_dir = output_dir
        self.onnx_path: Optional[str] = None
        self.engine_path: Optional[str] = None

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            "Initialized TensorRTExporter with model=%s, output_dir=%s",
            model_path,
            output_dir,
        )

    def export_onnx(self, opset: int = 17, dynamic_batch: bool = True) -> str:
        """Export the PyTorch model to ONNX format.

        Uses Ultralytics export with the specified ONNX opset version and
        optional dynamic batch axis for variable batch size inference.

        Args:
            opset: ONNX opset version to use. Default is 17 for broad
                TensorRT compatibility.
            dynamic_batch: Whether to enable dynamic batch dimension in the
                exported ONNX model. Enables variable batch sizes at inference.

        Returns:
            Path to the exported .onnx file.

        Raises:
            FileNotFoundError: If the source .pt model cannot be loaded.
            RuntimeError: If ONNX export fails due to unsupported operations.
        """
        from ultralytics import YOLO

        logger.info(
            "Starting ONNX export: model=%s, opset=%d, dynamic_batch=%s",
            self.model_path,
            opset,
            dynamic_batch,
        )

        # Load the YOLOv8 model
        model = YOLO(self.model_path)

        # Export to ONNX format using Ultralytics built-in export
        logger.info("Exporting model to ONNX format...")
        exported_path = model.export(
            format="onnx",
            opset=opset,
            dynamic=dynamic_batch,
            simplify=True,
        )

        # Move the exported ONNX file to the output directory
        exported_file = Path(exported_path)
        destination = Path(self.output_dir) / exported_file.name

        if exported_file.exists() and exported_file != destination:
            import shutil
            shutil.move(str(exported_file), str(destination))
            logger.info(
                "Moved ONNX file from %s to %s", exported_file, destination
            )

        # Validate the ONNX file exists
        if not destination.exists():
            raise RuntimeError(
                f"ONNX export failed: expected file not found at {destination}"
            )

        self.onnx_path = str(destination)
        logger.info("ONNX export complete: %s", self.onnx_path)

        return self.onnx_path

    def build_engine(
        self,
        precision: str = "fp16",
        max_batch_size: int = 8,
        workspace_mb: int = 4096,
        calibration_data: Optional[str] = None,
    ) -> str:
        """Build a TensorRT engine from the exported ONNX model.

        Constructs an optimized TensorRT engine with the specified precision
        mode and batch size configuration. For INT8 precision, a calibration
        dataset is required to compute quantization scales.

        Args:
            precision: Inference precision mode. One of "fp16", "int8", or
                "fp32". FP16 provides ~2x speedup with minimal accuracy loss.
                INT8 provides ~4x speedup but requires calibration data.
            max_batch_size: Maximum batch size the engine will support.
                Optimization profiles are created for batch sizes 1 (min),
                max_batch_size//2 (optimal), and max_batch_size (max).
            workspace_mb: Maximum GPU memory in MB available to the TensorRT
                builder for tactic selection. Higher values may find faster
                kernels but use more memory during build.
            calibration_data: Path to a directory of calibration images.
                Required when precision is "int8". Images are used to compute
                activation ranges for quantization.

        Returns:
            Path to the built .engine file.

        Raises:
            ValueError: If precision is "int8" and calibration_data is None.
            ValueError: If precision is not one of "fp16", "int8", "fp32".
            RuntimeError: If ONNX model has not been exported yet (call
                export_onnx first).
            RuntimeError: If TensorRT engine build fails.
        """
        # Validate precision
        valid_precisions = ("fp16", "int8", "fp32")
        if precision not in valid_precisions:
            raise ValueError(
                f"Precision must be one of {valid_precisions}, got: '{precision}'"
            )

        # Validate ONNX model has been exported
        if self.onnx_path is None:
            raise RuntimeError(
                "ONNX model has not been exported yet. Call export_onnx() first."
            )

        # Validate calibration data for INT8
        if precision == "int8" and calibration_data is None:
            raise ValueError(
                "calibration_data is required when precision is 'int8'. "
                "Provide a path to a directory of calibration images."
            )

        # Lazy import tensorrt
        import tensorrt as trt

        global TRT_LOGGER
        if TRT_LOGGER is None:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        logger.info(
            "Building TensorRT engine: precision=%s, max_batch_size=%d, "
            "workspace_mb=%d",
            precision,
            max_batch_size,
            workspace_mb,
        )

        # Create builder and network
        logger.info("Creating TensorRT builder and network...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        # Parse ONNX model
        logger.info("Parsing ONNX model: %s", self.onnx_path)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(str(parser.get_error(i)))
                raise RuntimeError(
                    f"Failed to parse ONNX model. Errors:\n"
                    + "\n".join(errors)
                )
        logger.info("ONNX model parsed successfully.")

        # Create builder config
        logger.info("Configuring builder (workspace=%d MB)...", workspace_mb)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
        )

        # Set precision flags
        if precision == "fp16":
            logger.info("Enabling FP16 precision mode.")
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            logger.info("Enabling INT8 precision mode with calibration.")
            config.set_flag(trt.BuilderFlag.INT8)
            # Import and set up the INT8 calibrator
            from metropolis.calibrator import EntropyCalibrator

            calibrator = EntropyCalibrator(calibration_data, batch_size=max_batch_size)
            config.int8_calibrator = calibrator

        # Create optimization profile with dynamic batch shapes
        logger.info(
            "Creating optimization profile: min=1, opt=%d, max=%d",
            max(max_batch_size // 2, 1),
            max_batch_size,
        )
        profile = builder.create_optimization_profile()

        # Get the input tensor name from the network
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        opt_batch = max(max_batch_size // 2, 1) if max_batch_size > 1 else 1
        profile.set_shape(
            input_name,
            min=(1, 3, 640, 640),
            opt=(opt_batch, 3, 640, 640),
            max=(max_batch_size, 3, 640, 640),
        )
        config.add_optimization_profile(profile)

        # Build serialized network
        logger.info("Building serialized TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(
                "TensorRT engine build failed. Check GPU memory and model compatibility."
            )

        # Write engine to file
        model_stem = Path(self.model_path).stem
        engine_filename = f"{model_stem}_{precision}.engine"
        engine_path = str(Path(self.output_dir) / engine_filename)

        logger.info("Writing engine to: %s", engine_path)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        self.engine_path = engine_path
        logger.info(
            "TensorRT engine build complete: %s (precision=%s, max_batch=%d)",
            self.engine_path,
            precision,
            max_batch_size,
        )

        return self.engine_path

    def validate(
        self, test_images: list[str], iou_threshold: float = 0.5
    ) -> dict:
        """Validate TensorRT engine accuracy against PyTorch baseline.

        Runs inference on test images using both the original PyTorch model
        and the TensorRT engine, then compares detection outputs to compute
        accuracy metrics. This ensures the optimization process has not
        degraded model quality beyond acceptable thresholds.

        Args:
            test_images: List of paths to test image files for validation.
            iou_threshold: IoU threshold for matching predicted boxes to
                baseline boxes when computing precision/recall. Default 0.5
                follows COCO evaluation convention.

        Returns:
            Dictionary containing accuracy metrics:
                - "mAP": Mean Average Precision across all classes.
                - "precision": Overall precision at the given IoU threshold.
                - "num_images": Number of images evaluated.
                - "engine_matches_baseline": Boolean indicating if accuracy
                  is within acceptable tolerance (1% for FP16, 2% for INT8).
                - "baseline_mAP": mAP of the PyTorch baseline model.
                - "mAP_drop": Absolute mAP drop from baseline.
                - "recall": Overall recall at the given IoU threshold.

        Raises:
            RuntimeError: If TensorRT engine has not been built yet (call
                build_engine first).
            FileNotFoundError: If any test image path does not exist.
        """
        # Validate engine has been built
        if self.engine_path is None:
            raise RuntimeError(
                "TensorRT engine has not been built yet. Call build_engine() first."
            )

        # Validate all test images exist
        for img_path in test_images:
            if not Path(img_path).exists():
                raise FileNotFoundError(
                    f"Test image not found: {img_path}"
                )

        from ultralytics import YOLO
        import numpy as np

        logger.info(
            "Starting validation: %d images, iou_threshold=%.2f",
            len(test_images),
            iou_threshold,
        )

        # Step 1: Run PyTorch baseline inference
        logger.info("Running PyTorch baseline inference...")
        model = YOLO(self.model_path)
        baseline_predictions = []
        for img_path in test_images:
            results = model.predict(img_path, verbose=False)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        det = {
                            "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                            "confidence": float(boxes.conf[i].cpu().numpy()),
                            "class_id": int(boxes.cls[i].cpu().numpy()),
                        }
                        detections.append(det)
            baseline_predictions.append(detections)

        # Step 2: Run TensorRT engine inference
        logger.info("Running TensorRT engine inference...")
        engine_predictions = []
        for img_path in test_images:
            detections = self._infer_tensorrt(img_path)
            engine_predictions.append(detections)

        # Step 3: Compute baseline mAP (baseline vs itself = 1.0 by definition,
        # but we use baseline as ground truth for the engine comparison)
        # Use baseline predictions as ground truth reference
        baseline_map = self._compute_map(
            baseline_predictions, baseline_predictions, iou_threshold
        )

        # Step 4: Compute engine mAP against baseline (baseline = ground truth)
        engine_metrics = self._compute_map(
            engine_predictions, baseline_predictions, iou_threshold
        )

        # Step 5: Determine tolerance based on precision mode
        engine_filename = Path(self.engine_path).stem
        if "int8" in engine_filename.lower():
            tolerance = 0.02  # 2% for INT8
        else:
            tolerance = 0.01  # 1% for FP16 (and FP32)

        mAP_drop = max(0.0, baseline_map["mAP"] - engine_metrics["mAP"])
        engine_matches = mAP_drop < tolerance

        result = {
            "mAP": engine_metrics["mAP"],
            "precision": engine_metrics["precision"],
            "recall": engine_metrics["recall"],
            "num_images": len(test_images),
            "engine_matches_baseline": engine_matches,
            "baseline_mAP": baseline_map["mAP"],
            "mAP_drop": mAP_drop,
        }

        logger.info(
            "Validation complete: engine_mAP=%.4f, baseline_mAP=%.4f, "
            "mAP_drop=%.4f, tolerance=%.4f, matches=%s",
            result["mAP"],
            result["baseline_mAP"],
            result["mAP_drop"],
            tolerance,
            result["engine_matches_baseline"],
        )

        return result

    def _infer_tensorrt(self, image_path: str) -> list[dict]:
        """Run inference on a single image using the TensorRT engine.

        Loads and preprocesses the image, runs it through the TensorRT
        engine, and parses the raw output into detection format.

        Args:
            image_path: Path to the image file to run inference on.

        Returns:
            List of detection dicts with keys: "bbox" (xyxy), "confidence",
            "class_id".
        """
        import numpy as np
        import tensorrt as trt

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "pycuda is required for TensorRT inference. "
                "Install with: pip install pycuda"
            )

        # Load and preprocess image
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize to 640x640
        img_resized = img.resize((640, 640), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32)

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # HWC -> CHW
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension: (1, 3, 640, 640)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.ascontiguousarray(img_array)

        # Load TensorRT engine
        global TRT_LOGGER
        if TRT_LOGGER is None:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Set input shape for dynamic batch
        input_name = engine.get_tensor_name(0)
        context.set_input_shape(input_name, (1, 3, 640, 640))

        # Allocate buffers
        input_size = int(np.prod((1, 3, 640, 640))) * 4  # float32
        d_input = cuda.mem_alloc(input_size)

        # Determine output shape
        output_name = engine.get_tensor_name(1)
        output_shape = context.get_tensor_shape(output_name)
        output_size = int(np.prod(output_shape)) * 4  # float32
        d_output = cuda.mem_alloc(output_size)

        # Set tensor addresses
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))

        # Copy input to device
        cuda.memcpy_htod(d_input, img_array)

        # Run inference
        stream = cuda.Stream()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

        # Copy output from device
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)

        # Parse YOLOv8 output
        # YOLOv8 output shape is typically (1, 84, 8400) for 80 classes
        # where 84 = 4 (bbox) + 80 (class scores)
        # Transpose to (8400, 84) for easier processing
        detections = self._parse_yolov8_output(
            output, orig_w, orig_h, conf_threshold=0.25
        )

        return detections

    def _parse_yolov8_output(
        self,
        raw_output: "np.ndarray",
        orig_w: int,
        orig_h: int,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[dict]:
        """Parse raw YOLOv8 TensorRT output into detection dicts.

        Args:
            raw_output: Raw output tensor from TensorRT engine.
            orig_w: Original image width for coordinate scaling.
            orig_h: Original image height for coordinate scaling.
            conf_threshold: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.

        Returns:
            List of detection dicts with "bbox", "confidence", "class_id".
        """
        import numpy as np

        # Handle batch dimension: (1, num_features, num_predictions)
        if raw_output.ndim == 3:
            output = raw_output[0]  # Remove batch dim -> (84, 8400)
        else:
            output = raw_output

        # Transpose to (num_predictions, num_features) -> (8400, 84)
        if output.shape[0] < output.shape[1]:
            output = output.T

        num_predictions = output.shape[0]
        num_classes = output.shape[1] - 4  # First 4 are bbox (cx, cy, w, h)

        # Extract boxes (center format) and class scores
        boxes_cxcywh = output[:, :4]
        class_scores = output[:, 4:]

        # Get max class score and class id for each prediction
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = max_scores > conf_threshold
        boxes_cxcywh = boxes_cxcywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # Convert from center format to xyxy
        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

        # Scale boxes from 640x640 to original image size
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0
        boxes_xyxy[:, 0] *= scale_x
        boxes_xyxy[:, 2] *= scale_x
        boxes_xyxy[:, 1] *= scale_y
        boxes_xyxy[:, 3] *= scale_y

        # Apply NMS per class
        keep_indices = self._nms(boxes_xyxy, max_scores, iou_threshold)

        detections = []
        for idx in keep_indices:
            det = {
                "bbox": boxes_xyxy[idx].tolist(),
                "confidence": float(max_scores[idx]),
                "class_id": int(class_ids[idx]),
            }
            detections.append(det)

        return detections

    @staticmethod
    def _nms(
        boxes: "np.ndarray",
        scores: "np.ndarray",
        iou_threshold: float,
    ) -> list[int]:
        """Apply Non-Maximum Suppression to filter overlapping boxes.

        Args:
            boxes: Array of shape (N, 4) with xyxy coordinates.
            scores: Array of shape (N,) with confidence scores.
            iou_threshold: IoU threshold above which boxes are suppressed.

        Returns:
            List of indices of boxes to keep.
        """
        import numpy as np

        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))

            if len(order) == 1:
                break

            # Compute IoU of the picked box with the rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _compute_map(
        self,
        predictions: list[list[dict]],
        ground_truth: list[list[dict]],
        iou_threshold: float,
    ) -> dict:
        """Compute mAP, precision, and recall comparing predictions to ground truth.

        For each image, matches predicted boxes to ground truth boxes using IoU.
        Computes precision-recall curve and calculates average precision per class.

        Args:
            predictions: List of per-image prediction lists. Each prediction
                is a dict with "bbox", "confidence", "class_id".
            ground_truth: List of per-image ground truth lists with same format.
            iou_threshold: IoU threshold for considering a match.

        Returns:
            Dict with "mAP", "precision", "recall" keys.
        """
        import numpy as np

        # Collect all class IDs present in ground truth
        all_classes = set()
        for gt_dets in ground_truth:
            for det in gt_dets:
                all_classes.add(det["class_id"])

        if not all_classes:
            # No ground truth detections at all
            return {"mAP": 1.0, "precision": 1.0, "recall": 1.0}

        # Compute AP per class
        ap_per_class = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for class_id in sorted(all_classes):
            # Gather all predictions and ground truths for this class
            all_preds = []  # (confidence, image_idx, pred_idx)
            gt_per_image = {}  # image_idx -> list of gt boxes

            for img_idx in range(len(ground_truth)):
                # Ground truth for this class in this image
                gt_boxes = [
                    det["bbox"]
                    for det in ground_truth[img_idx]
                    if det["class_id"] == class_id
                ]
                gt_per_image[img_idx] = {
                    "boxes": gt_boxes,
                    "matched": [False] * len(gt_boxes),
                }

                # Predictions for this class in this image
                for pred_idx, pred in enumerate(predictions[img_idx]):
                    if pred["class_id"] == class_id:
                        all_preds.append(
                            (pred["confidence"], img_idx, pred["bbox"])
                        )

            # Sort predictions by confidence (descending)
            all_preds.sort(key=lambda x: x[0], reverse=True)

            num_gt = sum(len(v["boxes"]) for v in gt_per_image.values())

            if num_gt == 0:
                continue

            tp_list = []
            fp_list = []

            for conf, img_idx, pred_box in all_preds:
                gt_data = gt_per_image[img_idx]
                gt_boxes = gt_data["boxes"]

                if len(gt_boxes) == 0:
                    tp_list.append(0)
                    fp_list.append(1)
                    continue

                # Compute IoU with all ground truth boxes for this image/class
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    iou = self._compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and not gt_data["matched"][best_gt_idx]:
                    tp_list.append(1)
                    fp_list.append(0)
                    gt_data["matched"][best_gt_idx] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

            # Compute precision-recall curve
            tp_cumsum = np.cumsum(tp_list)
            fp_cumsum = np.cumsum(fp_list)

            recalls = tp_cumsum / num_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            # Compute AP using 11-point interpolation
            ap = self._compute_ap(precisions, recalls)
            ap_per_class.append(ap)

            # Accumulate totals
            total_tp += int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
            total_fp += int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
            total_fn += num_gt - (int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0)

        # Compute mean AP
        mAP = float(np.mean(ap_per_class)) if ap_per_class else 0.0

        # Compute overall precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        return {
            "mAP": mAP,
            "precision": precision,
            "recall": recall,
        }

    @staticmethod
    def _compute_ap(precisions: "np.ndarray", recalls: "np.ndarray") -> float:
        """Compute Average Precision using 11-point interpolation.

        Args:
            precisions: Array of precision values at each detection.
            recalls: Array of recall values at each detection.

        Returns:
            Average precision value.
        """
        import numpy as np

        if len(precisions) == 0:
            return 0.0

        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            # Find precision at recall >= t
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask])

        ap /= 11.0
        return float(ap)

    @staticmethod
    def _compute_iou(box1: list[float], box2: list[float]) -> float:
        """Compute Intersection over Union between two boxes in xyxy format.

        Args:
            box1: First box as [x1, y1, x2, y2].
            box2: Second box as [x1, y1, x2, y2].

        Returns:
            IoU value between 0 and 1.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union


def main() -> None:
    """CLI entry point for the TensorRT export pipeline.

    Provides a command-line interface for exporting YOLOv8 models to
    optimized TensorRT engines with configurable precision and validation.

    Usage:
        python -m metropolis.export_tensorrt --model models/yolov8m.pt --precision fp16
        python -m metropolis.export_tensorrt --model models/yolov8m.pt --precision int8 --calibration-data data/calib/
        python -m metropolis.export_tensorrt --model models/yolov8m.pt --validate --test-images data/test/
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Export YOLOv8 models to optimized TensorRT engines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m metropolis.export_tensorrt --model yolov8m.pt --precision fp16\n"
            "  python -m metropolis.export_tensorrt --model yolov8m.pt --precision int8 "
            "--calibration-data ./calib_images/\n"
            "  python -m metropolis.export_tensorrt --model yolov8m.pt --validate "
            "--test-images ./test_images/\n"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the YOLOv8 .pt model file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/output",
        help="Output directory for exported files (default: models/output)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "int8", "fp32"],
        help="Precision mode for TensorRT engine (default: fp16)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for the engine (default: 8)",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=4096,
        help="TensorRT workspace memory in MB (default: 4096)",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration images directory (required for int8 precision)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after export to compare against PyTorch baseline",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        default=None,
        help="Directory of test images for validation (used with --validate)",
    )

    args = parser.parse_args()

    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate INT8 requires calibration data
    if args.precision == "int8" and args.calibration_data is None:
        parser.error("--calibration-data is required when --precision is int8")

    # Validate --validate requires --test-images
    if args.validate and args.test_images is None:
        parser.error("--test-images is required when --validate is set")

    print(f"{'='*60}")
    print("TensorRT Export Pipeline")
    print(f"{'='*60}")
    print(f"  Model:          {args.model}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Precision:      {args.precision}")
    print(f"  Max batch size: {args.max_batch_size}")
    print(f"  Workspace:      {args.workspace_mb} MB")
    print(f"  ONNX opset:     {args.opset}")
    if args.calibration_data:
        print(f"  Calibration:    {args.calibration_data}")
    if args.validate:
        print(f"  Validate:       Yes")
        print(f"  Test images:    {args.test_images}")
    print(f"{'='*60}")
    print()

    try:
        # Step 1: Create exporter
        print("[1/3] Initializing exporter...")
        exporter = TensorRTExporter(args.model, args.output_dir)

        # Step 2: Export to ONNX
        print(f"[2/3] Exporting to ONNX (opset={args.opset})...")
        onnx_path = exporter.export_onnx(opset=args.opset, dynamic_batch=True)
        print(f"  ✓ ONNX exported: {onnx_path}")
        print()

        # Step 3: Build TensorRT engine
        print(
            f"[3/3] Building TensorRT engine (precision={args.precision}, "
            f"batch_size={args.max_batch_size})..."
        )
        engine_path = exporter.build_engine(
            precision=args.precision,
            max_batch_size=args.max_batch_size,
            workspace_mb=args.workspace_mb,
            calibration_data=args.calibration_data,
        )
        print(f"  ✓ Engine built: {engine_path}")
        print()

        # Optional: Validate
        if args.validate:
            print("[Validation] Running accuracy validation...")
            # Collect test images from directory
            test_images_dir = Path(args.test_images)
            if not test_images_dir.is_dir():
                print(f"  ✗ Error: Test images directory not found: {args.test_images}")
                sys.exit(1)

            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            test_images = [
                str(p)
                for p in sorted(test_images_dir.iterdir())
                if p.suffix.lower() in image_extensions
            ]

            if not test_images:
                print(f"  ✗ Error: No images found in {args.test_images}")
                sys.exit(1)

            print(f"  Found {len(test_images)} test images")
            metrics = exporter.validate(test_images=test_images)

            print()
            print("  Validation Results:")
            print(f"  {'─'*40}")
            print(f"  Engine mAP:          {metrics['mAP']:.4f}")
            print(f"  Baseline mAP:        {metrics['baseline_mAP']:.4f}")
            print(f"  mAP drop:            {metrics['mAP_drop']:.4f}")
            print(f"  Precision:           {metrics['precision']:.4f}")
            print(f"  Recall:              {metrics['recall']:.4f}")
            print(f"  Images evaluated:    {metrics['num_images']}")
            print(f"  Matches baseline:    {'✓ Yes' if metrics['engine_matches_baseline'] else '✗ No'}")
            print(f"  {'─'*40}")

            if not metrics["engine_matches_baseline"]:
                print()
                print("  ⚠ WARNING: Engine accuracy exceeds tolerance threshold!")
                sys.exit(1)

        print()
        print(f"{'='*60}")
        print("Export complete!")
        print(f"  ONNX:   {onnx_path}")
        print(f"  Engine: {engine_path}")
        print(f"{'='*60}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExport cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error during export")
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
