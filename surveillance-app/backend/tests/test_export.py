"""Comprehensive unit tests for TensorRT export pipeline.

Tests cover:
- validate() method (engine accuracy validation)
- Helper methods (_compute_iou, _nms, _compute_map, _parse_yolov8_output)
- CLI (main function) argument parsing

All tests mock TensorRT, pycuda, and Ultralytics so they run in CI without GPU.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.export_tensorrt import TensorRTExporter, main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_exporter(tmp_path):
    """Create a TensorRTExporter with a valid .pt file and output dir."""
    model_file = tmp_path / "yolov8m.pt"
    model_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    exporter = TensorRTExporter(str(model_file), str(output_dir))
    return exporter


@pytest.fixture
def exporter_with_engine(tmp_path):
    """Create a TensorRTExporter with engine_path set (simulating build_engine done)."""
    model_file = tmp_path / "yolov8m.pt"
    model_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    exporter = TensorRTExporter(str(model_file), str(output_dir))
    # Create a fake engine file
    engine_file = output_dir / "yolov8m_fp16.engine"
    engine_file.write_bytes(b"fake_engine")
    exporter.engine_path = str(engine_file)
    return exporter


@pytest.fixture
def test_images(tmp_path):
    """Create temporary test image files."""
    images = []
    for i in range(3):
        img = tmp_path / f"test_image_{i}.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG header
        images.append(str(img))
    return images


# ---------------------------------------------------------------------------
# Tests for validate() method
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for the validate() method."""

    def test_raises_runtime_error_when_engine_path_is_none(self, tmp_exporter, test_images):
        """validate() raises RuntimeError when engine has not been built."""
        assert tmp_exporter.engine_path is None
        with pytest.raises(RuntimeError, match="TensorRT engine has not been built"):
            tmp_exporter.validate(test_images=test_images)

    def test_raises_file_not_found_for_missing_test_images(self, exporter_with_engine):
        """validate() raises FileNotFoundError for non-existent test images."""
        missing_images = ["/nonexistent/path/image.jpg"]
        with pytest.raises(FileNotFoundError, match="Test image not found"):
            exporter_with_engine.validate(test_images=missing_images)

    def test_validate_with_mocked_inference(self, exporter_with_engine, test_images):
        """validate() runs end-to-end with mocked YOLO and TensorRT inference."""
        # Mock Ultralytics YOLO predictions
        mock_ultralytics = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        # Create mock result with boxes
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__ = MagicMock(return_value=1)
        mock_boxes.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.array([100.0, 100.0, 200.0, 200.0])))))]
        mock_boxes.conf = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.float32(0.9)))))]
        mock_boxes.cls = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.int32(0)))))]
        mock_result.boxes = mock_boxes
        mock_yolo_instance.predict.return_value = [mock_result]

        # Mock _infer_tensorrt to return similar detections (within tolerance)
        engine_detections = [
            {"bbox": [100.0, 100.0, 200.0, 200.0], "confidence": 0.88, "class_id": 0}
        ]

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            with patch.object(exporter_with_engine, "_infer_tensorrt", return_value=engine_detections):
                result = exporter_with_engine.validate(test_images=test_images)

        assert "mAP" in result
        assert "precision" in result
        assert "recall" in result
        assert "num_images" in result
        assert "engine_matches_baseline" in result
        assert "baseline_mAP" in result
        assert "mAP_drop" in result
        assert result["num_images"] == 3

    def test_engine_matches_baseline_true_within_tolerance(self, exporter_with_engine, test_images):
        """engine_matches_baseline is True when mAP drop is within 1% tolerance."""
        mock_ultralytics = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        # Baseline: one detection per image
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__ = MagicMock(return_value=1)
        mock_boxes.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.array([50.0, 50.0, 150.0, 150.0])))))]
        mock_boxes.conf = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.float32(0.95)))))]
        mock_boxes.cls = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.int32(0)))))]
        mock_result.boxes = mock_boxes
        mock_yolo_instance.predict.return_value = [mock_result]

        # Engine returns identical detections (0% drop)
        engine_detections = [
            {"bbox": [50.0, 50.0, 150.0, 150.0], "confidence": 0.95, "class_id": 0}
        ]

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            with patch.object(exporter_with_engine, "_infer_tensorrt", return_value=engine_detections):
                result = exporter_with_engine.validate(test_images=test_images)

        assert result["engine_matches_baseline"] is True
        assert result["mAP_drop"] < 0.01

    def test_engine_matches_baseline_false_exceeds_tolerance(self, exporter_with_engine, test_images):
        """engine_matches_baseline is False when mAP drop exceeds tolerance."""
        mock_ultralytics = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        # Baseline: one detection per image
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__ = MagicMock(return_value=1)
        mock_boxes.xyxy = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.array([50.0, 50.0, 150.0, 150.0])))))]
        mock_boxes.conf = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.float32(0.95)))))]
        mock_boxes.cls = [MagicMock(cpu=MagicMock(return_value=MagicMock(
            numpy=MagicMock(return_value=np.int32(0)))))]
        mock_result.boxes = mock_boxes
        mock_yolo_instance.predict.return_value = [mock_result]

        # Engine returns completely wrong detections (different location, no IoU overlap)
        engine_detections = [
            {"bbox": [400.0, 400.0, 500.0, 500.0], "confidence": 0.9, "class_id": 0}
        ]

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            with patch.object(exporter_with_engine, "_infer_tensorrt", return_value=engine_detections):
                result = exporter_with_engine.validate(test_images=test_images)

        assert result["engine_matches_baseline"] is False
        assert result["mAP_drop"] > 0.01

    def test_validate_with_empty_predictions(self, exporter_with_engine, test_images):
        """validate() handles empty predictions (no detections) gracefully."""
        mock_ultralytics = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_yolo_instance

        # Baseline: no detections
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_yolo_instance.predict.return_value = [mock_result]

        # Engine also returns no detections
        engine_detections = []

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            with patch.object(exporter_with_engine, "_infer_tensorrt", return_value=engine_detections):
                result = exporter_with_engine.validate(test_images=test_images)

        # With no ground truth, _compute_map returns mAP=1.0
        assert result["mAP"] == 1.0
        assert result["engine_matches_baseline"] is True
        assert result["num_images"] == 3


# ---------------------------------------------------------------------------
# Tests for helper methods
# ---------------------------------------------------------------------------


class TestComputeIoU:
    """Tests for _compute_iou static method."""

    def test_overlapping_boxes(self):
        """IoU of partially overlapping boxes is between 0 and 1."""
        box1 = [0.0, 0.0, 100.0, 100.0]
        box2 = [50.0, 50.0, 150.0, 150.0]
        iou = TensorRTExporter._compute_iou(box1, box2)
        # Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        expected = 2500.0 / 17500.0
        assert abs(iou - expected) < 1e-6

    def test_non_overlapping_boxes(self):
        """IoU of non-overlapping boxes is 0."""
        box1 = [0.0, 0.0, 50.0, 50.0]
        box2 = [100.0, 100.0, 200.0, 200.0]
        iou = TensorRTExporter._compute_iou(box1, box2)
        assert iou == 0.0

    def test_identical_boxes(self):
        """IoU of identical boxes is 1.0."""
        box = [10.0, 20.0, 110.0, 120.0]
        iou = TensorRTExporter._compute_iou(box, box)
        assert iou == 1.0

    def test_one_box_inside_another(self):
        """IoU when one box is completely inside another."""
        outer = [0.0, 0.0, 200.0, 200.0]
        inner = [50.0, 50.0, 100.0, 100.0]
        iou = TensorRTExporter._compute_iou(outer, inner)
        # Intersection = area of inner = 50*50 = 2500
        # Union = 200*200 + 50*50 - 2500 = 40000 + 2500 - 2500 = 40000
        expected = 2500.0 / 40000.0
        assert abs(iou - expected) < 1e-6

    def test_touching_boxes_zero_iou(self):
        """Boxes that share an edge but don't overlap have IoU = 0."""
        box1 = [0.0, 0.0, 50.0, 50.0]
        box2 = [50.0, 0.0, 100.0, 50.0]
        iou = TensorRTExporter._compute_iou(box1, box2)
        assert iou == 0.0


class TestNMS:
    """Tests for _nms static method."""

    def test_overlapping_boxes_suppressed(self):
        """NMS suppresses overlapping boxes with lower scores."""
        boxes = np.array([
            [0.0, 0.0, 100.0, 100.0],
            [10.0, 10.0, 110.0, 110.0],  # High overlap with first
        ], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)

        keep = TensorRTExporter._nms(boxes, scores, iou_threshold=0.5)

        # Only the higher-scoring box should be kept
        assert keep == [0]

    def test_non_overlapping_boxes_all_kept(self):
        """NMS keeps all boxes when they don't overlap."""
        boxes = np.array([
            [0.0, 0.0, 50.0, 50.0],
            [200.0, 200.0, 300.0, 300.0],
            [400.0, 400.0, 500.0, 500.0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        keep = TensorRTExporter._nms(boxes, scores, iou_threshold=0.5)

        assert sorted(keep) == [0, 1, 2]

    def test_nms_empty_input(self):
        """NMS returns empty list for empty input."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)

        keep = TensorRTExporter._nms(boxes, scores, iou_threshold=0.5)

        assert keep == []

    def test_nms_single_box(self):
        """NMS keeps the single box."""
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.8], dtype=np.float32)

        keep = TensorRTExporter._nms(boxes, scores, iou_threshold=0.5)

        assert keep == [0]

    def test_nms_high_threshold_keeps_more(self):
        """Higher IoU threshold keeps more overlapping boxes."""
        boxes = np.array([
            [0.0, 0.0, 100.0, 100.0],
            [10.0, 10.0, 110.0, 110.0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)

        # With very high threshold, overlapping boxes are kept
        keep = TensorRTExporter._nms(boxes, scores, iou_threshold=0.99)

        assert sorted(keep) == [0, 1]


class TestComputeMap:
    """Tests for _compute_map method."""

    def test_perfect_predictions(self, tmp_exporter):
        """mAP is 1.0 when predictions exactly match ground truth."""
        predictions = [
            [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0}],
            [{"bbox": [20.0, 20.0, 80.0, 80.0], "confidence": 0.85, "class_id": 1}],
        ]
        ground_truth = [
            [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0}],
            [{"bbox": [20.0, 20.0, 80.0, 80.0], "confidence": 0.85, "class_id": 1}],
        ]

        result = tmp_exporter._compute_map(predictions, ground_truth, iou_threshold=0.5)

        assert result["mAP"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_no_predictions(self, tmp_exporter):
        """mAP is 0 when there are no predictions but ground truth exists."""
        predictions = [[], []]
        ground_truth = [
            [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0}],
            [{"bbox": [20.0, 20.0, 80.0, 80.0], "confidence": 0.85, "class_id": 1}],
        ]

        result = tmp_exporter._compute_map(predictions, ground_truth, iou_threshold=0.5)

        assert result["mAP"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_no_ground_truth(self, tmp_exporter):
        """mAP is 1.0 when there is no ground truth (vacuously true)."""
        predictions = [
            [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0}],
        ]
        ground_truth = [[]]

        result = tmp_exporter._compute_map(predictions, ground_truth, iou_threshold=0.5)

        assert result["mAP"] == 1.0

    def test_partial_match(self, tmp_exporter):
        """mAP is between 0 and 1 for partial matches."""
        predictions = [
            [
                {"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0},
                {"bbox": [300.0, 300.0, 400.0, 400.0], "confidence": 0.8, "class_id": 0},  # FP
            ],
        ]
        ground_truth = [
            [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.9, "class_id": 0}],
        ]

        result = tmp_exporter._compute_map(predictions, ground_truth, iou_threshold=0.5)

        assert 0.0 < result["mAP"] <= 1.0
        assert result["recall"] == 1.0  # The GT box was matched
        assert result["precision"] == 0.5  # 1 TP, 1 FP


class TestParseYolov8Output:
    """Tests for _parse_yolov8_output method."""

    def test_parse_with_detections(self, tmp_exporter):
        """_parse_yolov8_output correctly parses tensor with detections."""
        # Create a mock output tensor: shape (1, 84, 8400)
        # 84 = 4 (bbox: cx, cy, w, h) + 80 (class scores)
        # Use enough predictions so shape[0] < shape[1] after batch dim removal
        # triggering the transpose logic in the method
        num_classes = 80
        num_predictions = 100  # Must be > 84 so transpose is triggered
        raw_output = np.zeros((1, 4 + num_classes, num_predictions), dtype=np.float32)

        # Set one detection: center at (320, 320), size 100x100, class 0 with score 0.9
        raw_output[0, 0, 0] = 320.0  # cx
        raw_output[0, 1, 0] = 320.0  # cy
        raw_output[0, 2, 0] = 100.0  # w
        raw_output[0, 3, 0] = 100.0  # h
        raw_output[0, 4, 0] = 0.9    # class 0 score

        detections = tmp_exporter._parse_yolov8_output(
            raw_output, orig_w=640, orig_h=640, conf_threshold=0.25
        )

        assert len(detections) == 1
        det = detections[0]
        assert det["class_id"] == 0
        assert abs(det["confidence"] - 0.9) < 1e-5
        # bbox should be xyxy: (320-50, 320-50, 320+50, 320+50) = (270, 270, 370, 370)
        assert abs(det["bbox"][0] - 270.0) < 1e-3
        assert abs(det["bbox"][1] - 270.0) < 1e-3
        assert abs(det["bbox"][2] - 370.0) < 1e-3
        assert abs(det["bbox"][3] - 370.0) < 1e-3

    def test_parse_with_no_detections_above_threshold(self, tmp_exporter):
        """_parse_yolov8_output returns empty list when all scores below threshold."""
        num_classes = 80
        num_predictions = 100
        raw_output = np.zeros((1, 4 + num_classes, num_predictions), dtype=np.float32)
        # All class scores are 0 (below threshold)

        detections = tmp_exporter._parse_yolov8_output(
            raw_output, orig_w=640, orig_h=640, conf_threshold=0.25
        )

        assert detections == []

    def test_parse_scales_to_original_image_size(self, tmp_exporter):
        """_parse_yolov8_output scales boxes from 640x640 to original dimensions."""
        num_classes = 80
        num_predictions = 100
        raw_output = np.zeros((1, 4 + num_classes, num_predictions), dtype=np.float32)

        # Detection at center of 640x640 image
        raw_output[0, 0, 0] = 320.0  # cx
        raw_output[0, 1, 0] = 320.0  # cy
        raw_output[0, 2, 0] = 100.0  # w
        raw_output[0, 3, 0] = 100.0  # h
        raw_output[0, 4, 0] = 0.8    # class 0 score

        # Original image is 1280x960 (2x width, 1.5x height)
        detections = tmp_exporter._parse_yolov8_output(
            raw_output, orig_w=1280, orig_h=960, conf_threshold=0.25
        )

        assert len(detections) == 1
        det = detections[0]
        # x coords scaled by 1280/640 = 2.0
        # y coords scaled by 960/640 = 1.5
        assert abs(det["bbox"][0] - 540.0) < 1e-3  # (320-50)*2 = 540
        assert abs(det["bbox"][1] - 405.0) < 1e-3  # (320-50)*1.5 = 405
        assert abs(det["bbox"][2] - 740.0) < 1e-3  # (320+50)*2 = 740
        assert abs(det["bbox"][3] - 555.0) < 1e-3  # (320+50)*1.5 = 555

    def test_parse_multiple_detections_with_nms(self, tmp_exporter):
        """_parse_yolov8_output applies NMS to overlapping detections."""
        num_classes = 80
        num_predictions = 100
        raw_output = np.zeros((1, 4 + num_classes, num_predictions), dtype=np.float32)

        # Two overlapping detections (same class, same location)
        raw_output[0, 0, 0] = 320.0
        raw_output[0, 1, 0] = 320.0
        raw_output[0, 2, 0] = 100.0
        raw_output[0, 3, 0] = 100.0
        raw_output[0, 4, 0] = 0.9  # class 0, high score

        raw_output[0, 0, 1] = 325.0  # Slightly offset
        raw_output[0, 1, 1] = 325.0
        raw_output[0, 2, 1] = 100.0
        raw_output[0, 3, 1] = 100.0
        raw_output[0, 4, 1] = 0.7  # class 0, lower score

        # One non-overlapping detection
        raw_output[0, 0, 2] = 100.0
        raw_output[0, 1, 2] = 100.0
        raw_output[0, 2, 2] = 50.0
        raw_output[0, 3, 2] = 50.0
        raw_output[0, 4, 2] = 0.8  # class 0

        detections = tmp_exporter._parse_yolov8_output(
            raw_output, orig_w=640, orig_h=640, conf_threshold=0.25
        )

        # NMS should suppress the lower-scoring overlapping box
        # Keeping the high-score overlapping one and the non-overlapping one
        assert len(detections) == 2


# ---------------------------------------------------------------------------
# Tests for CLI (main function)
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the CLI main() function."""

    def test_help_exits_with_zero(self):
        """--help flag exits with code 0."""
        with patch("sys.argv", ["export_tensorrt", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_missing_model_argument_exits_with_error(self):
        """Missing required --model argument exits with error code."""
        with patch("sys.argv", ["export_tensorrt", "--precision", "fp16"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0

    def test_int8_without_calibration_data_exits_with_error(self, tmp_path):
        """--precision int8 without --calibration-data exits with error."""
        model_file = tmp_path / "model.pt"
        model_file.touch()

        with patch("sys.argv", [
            "export_tensorrt",
            "--model", str(model_file),
            "--precision", "int8",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0

    def test_valid_args_calls_exporter(self, tmp_path):
        """Valid arguments proceed to create exporter and call export methods."""
        model_file = tmp_path / "model.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        mock_exporter_instance = MagicMock()
        mock_exporter_instance.export_onnx.return_value = str(output_dir / "model.onnx")
        mock_exporter_instance.build_engine.return_value = str(output_dir / "model_fp16.engine")

        with patch("sys.argv", [
            "export_tensorrt",
            "--model", str(model_file),
            "--output-dir", str(output_dir),
            "--precision", "fp16",
        ]):
            with patch("metropolis.export_tensorrt.TensorRTExporter", return_value=mock_exporter_instance):
                main()

        mock_exporter_instance.export_onnx.assert_called_once()
        mock_exporter_instance.build_engine.assert_called_once_with(
            precision="fp16",
            max_batch_size=8,
            workspace_mb=4096,
            calibration_data=None,
        )

    def test_nonexistent_model_exits_with_error(self, tmp_path):
        """Non-existent model file causes exit with error code 1."""
        with patch("sys.argv", [
            "export_tensorrt",
            "--model", "/nonexistent/model.pt",
            "--precision", "fp16",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
