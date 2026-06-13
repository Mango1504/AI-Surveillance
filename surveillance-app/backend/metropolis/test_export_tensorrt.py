"""Unit tests for TensorRTExporter class (task 2.2: export_onnx and __init__ validation)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from .export_tensorrt import TensorRTExporter


class TestTensorRTExporterInit:
    """Tests for __init__ validation logic."""

    def test_raises_file_not_found_for_missing_model(self, tmp_path):
        """__init__ raises FileNotFoundError when model_path doesn't exist."""
        fake_model = str(tmp_path / "nonexistent.pt")
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            TensorRTExporter(fake_model, str(tmp_path / "output"))

    def test_raises_value_error_for_non_pt_extension(self, tmp_path):
        """__init__ raises ValueError when model_path doesn't have .pt extension."""
        bad_model = tmp_path / "model.onnx"
        bad_model.touch()
        with pytest.raises(ValueError, match="must have .pt extension"):
            TensorRTExporter(str(bad_model), str(tmp_path / "output"))

    def test_raises_value_error_for_no_extension(self, tmp_path):
        """__init__ raises ValueError when model_path has no extension."""
        bad_model = tmp_path / "model"
        bad_model.touch()
        with pytest.raises(ValueError, match="must have .pt extension"):
            TensorRTExporter(str(bad_model), str(tmp_path / "output"))

    def test_valid_model_path_initializes_successfully(self, tmp_path):
        """__init__ succeeds with a valid .pt file that exists."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        exporter = TensorRTExporter(str(model_file), str(output_dir))

        assert exporter.model_path == str(model_file)
        assert exporter.output_dir == str(output_dir)
        assert exporter.onnx_path is None
        assert exporter.engine_path is None

    def test_creates_output_directory_if_not_exists(self, tmp_path):
        """__init__ creates the output directory when it doesn't exist."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "nested" / "output" / "dir"

        TensorRTExporter(str(model_file), str(output_dir))

        assert output_dir.exists()


class TestExportOnnx:
    """Tests for export_onnx() method."""

    def _mock_ultralytics(self):
        """Create a mock ultralytics module that can be imported."""
        mock_ultralytics = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_yolo_instance
        return mock_ultralytics, mock_yolo_instance

    def test_export_onnx_calls_ultralytics_with_correct_params(self, tmp_path):
        """export_onnx uses correct parameters for Ultralytics export."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        # Create a fake ONNX output file that Ultralytics would produce
        fake_onnx = tmp_path / "yolov8m.onnx"
        fake_onnx.touch()

        mock_ultralytics, mock_yolo_instance = self._mock_ultralytics()
        mock_yolo_instance.export.return_value = str(fake_onnx)

        exporter = TensorRTExporter(str(model_file), str(output_dir))

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            result = exporter.export_onnx(opset=17, dynamic_batch=True)

        # Verify YOLO was instantiated with the model path
        mock_ultralytics.YOLO.assert_called_once_with(str(model_file))

        # Verify export was called with correct parameters
        mock_yolo_instance.export.assert_called_once_with(
            format="onnx",
            opset=17,
            dynamic=True,
            simplify=True,
        )

        # Verify the result path is in the output directory
        assert Path(result).parent == output_dir
        assert result.endswith(".onnx")
        assert exporter.onnx_path == result

    def test_export_onnx_with_custom_opset(self, tmp_path):
        """export_onnx passes custom opset value to Ultralytics."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        fake_onnx = tmp_path / "yolov8m.onnx"
        fake_onnx.touch()

        mock_ultralytics, mock_yolo_instance = self._mock_ultralytics()
        mock_yolo_instance.export.return_value = str(fake_onnx)

        exporter = TensorRTExporter(str(model_file), str(output_dir))

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            exporter.export_onnx(opset=13, dynamic_batch=False)

        mock_yolo_instance.export.assert_called_once_with(
            format="onnx",
            opset=13,
            dynamic=False,
            simplify=True,
        )

    def test_export_onnx_moves_file_to_output_dir(self, tmp_path):
        """export_onnx moves the exported file to the configured output_dir."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        # Simulate Ultralytics exporting to the same directory as the model
        fake_onnx = tmp_path / "yolov8m.onnx"
        fake_onnx.write_text("fake onnx content")

        mock_ultralytics, mock_yolo_instance = self._mock_ultralytics()
        mock_yolo_instance.export.return_value = str(fake_onnx)

        exporter = TensorRTExporter(str(model_file), str(output_dir))

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            result = exporter.export_onnx()

        # File should now be in output_dir
        expected_path = str(output_dir / "yolov8m.onnx")
        assert result == expected_path
        assert Path(expected_path).exists()

    def test_export_onnx_stores_path_in_onnx_path_attribute(self, tmp_path):
        """export_onnx sets self.onnx_path after successful export."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        fake_onnx = tmp_path / "yolov8m.onnx"
        fake_onnx.touch()

        mock_ultralytics, mock_yolo_instance = self._mock_ultralytics()
        mock_yolo_instance.export.return_value = str(fake_onnx)

        exporter = TensorRTExporter(str(model_file), str(output_dir))
        assert exporter.onnx_path is None

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            result = exporter.export_onnx()

        assert exporter.onnx_path is not None
        assert exporter.onnx_path == result

    def test_export_onnx_raises_runtime_error_on_missing_output(self, tmp_path):
        """export_onnx raises RuntimeError if ONNX file doesn't exist after export."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        # Return a path that doesn't actually exist (simulating failed export)
        nonexistent_onnx = tmp_path / "nonexistent.onnx"

        mock_ultralytics, mock_yolo_instance = self._mock_ultralytics()
        mock_yolo_instance.export.return_value = str(nonexistent_onnx)

        exporter = TensorRTExporter(str(model_file), str(output_dir))

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            with pytest.raises(RuntimeError, match="ONNX export failed"):
                exporter.export_onnx()

    def test_export_onnx_no_move_when_already_in_output_dir(self, tmp_path):
        """export_onnx skips move when file is already in output_dir."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Simulate Ultralytics exporting directly to output_dir
        fake_onnx = output_dir / "yolov8m.onnx"
        fake_onnx.write_text("fake onnx content")

        mock_ultralytics, mock_yolo_instance = self._mock_ultralytics()
        mock_yolo_instance.export.return_value = str(fake_onnx)

        exporter = TensorRTExporter(str(model_file), str(output_dir))

        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            result = exporter.export_onnx()

        assert result == str(fake_onnx)
        assert Path(result).exists()


class TestBuildEngine:
    """Tests for build_engine() method (task 2.3)."""

    def _create_exporter_with_onnx(self, tmp_path):
        """Helper to create an exporter with onnx_path already set."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        exporter = TensorRTExporter(str(model_file), str(output_dir))
        # Simulate that export_onnx was already called
        fake_onnx = output_dir / "yolov8m.onnx"
        fake_onnx.write_bytes(b"fake onnx data")
        exporter.onnx_path = str(fake_onnx)
        return exporter

    def test_raises_value_error_for_invalid_precision(self, tmp_path):
        """build_engine raises ValueError for unsupported precision values."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        with pytest.raises(ValueError, match="Precision must be one of"):
            exporter.build_engine(precision="fp64")

    def test_raises_value_error_for_empty_precision(self, tmp_path):
        """build_engine raises ValueError for empty precision string."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        with pytest.raises(ValueError, match="Precision must be one of"):
            exporter.build_engine(precision="")

    def test_raises_runtime_error_when_onnx_not_exported(self, tmp_path):
        """build_engine raises RuntimeError if export_onnx hasn't been called."""
        model_file = tmp_path / "yolov8m.pt"
        model_file.touch()
        output_dir = tmp_path / "output"

        exporter = TensorRTExporter(str(model_file), str(output_dir))
        # onnx_path is None by default

        with pytest.raises(RuntimeError, match="ONNX model has not been exported"):
            exporter.build_engine(precision="fp16")

    def test_raises_value_error_for_int8_without_calibration_data(self, tmp_path):
        """build_engine raises ValueError when int8 is used without calibration_data."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        with pytest.raises(ValueError, match="calibration_data is required"):
            exporter.build_engine(precision="int8", calibration_data=None)

    def test_int8_with_calibration_data_does_not_raise_validation_error(self, tmp_path):
        """build_engine does not raise ValueError for int8 with calibration_data provided."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        # Mock tensorrt to avoid needing actual GPU
        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0
        mock_trt.BuilderFlag.INT8 = 1
        mock_trt.BuilderFlag.FP16 = 0

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "images"
        mock_network.get_input.return_value = mock_input_tensor

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        mock_builder.build_serialized_network.return_value = b"fake_engine_data"

        mock_calibrator_module = MagicMock()

        with patch.dict(sys.modules, {
            "tensorrt": mock_trt,
            "metropolis": MagicMock(),
            "metropolis.calibrator": mock_calibrator_module,
        }):
            # Patch the import inside the method
            with patch("builtins.open", MagicMock()):
                with patch("builtins.open") as mock_open:
                    mock_open.return_value.__enter__ = MagicMock(
                        return_value=MagicMock(read=MagicMock(return_value=b"onnx"))
                    )
                    mock_open.return_value.__exit__ = MagicMock(return_value=False)
                    # This should not raise a ValueError about calibration_data
                    # It may raise other errors due to mocking, but not the validation error
                    try:
                        exporter.build_engine(
                            precision="int8",
                            calibration_data="/path/to/calibration",
                        )
                    except ValueError as e:
                        if "calibration_data" in str(e):
                            pytest.fail(
                                "Should not raise calibration_data ValueError "
                                "when calibration_data is provided"
                            )

    def test_build_engine_fp16_calls_tensorrt_correctly(self, tmp_path):
        """build_engine with fp16 sets the FP16 builder flag."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        # Create comprehensive TensorRT mock
        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0
        mock_trt.BuilderFlag.FP16 = "FP16_FLAG"
        mock_trt.BuilderFlag.INT8 = "INT8_FLAG"

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "images"
        mock_network.get_input.return_value = mock_input_tensor

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        mock_builder.build_serialized_network.return_value = b"fake_engine_data"

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            result = exporter.build_engine(precision="fp16", max_batch_size=8)

        # Verify FP16 flag was set
        mock_config.set_flag.assert_called_once_with("FP16_FLAG")

        # Verify optimization profile was configured
        mock_profile.set_shape.assert_called_once_with(
            "images",
            min=(1, 3, 640, 640),
            opt=(4, 3, 640, 640),
            max=(8, 3, 640, 640),
        )

        # Verify engine was written and path stored
        assert exporter.engine_path is not None
        assert result == exporter.engine_path
        assert result.endswith("_fp16.engine")

    def test_build_engine_fp32_does_not_set_precision_flags(self, tmp_path):
        """build_engine with fp32 does not set FP16 or INT8 flags."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0
        mock_trt.BuilderFlag.FP16 = "FP16_FLAG"
        mock_trt.BuilderFlag.INT8 = "INT8_FLAG"

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "images"
        mock_network.get_input.return_value = mock_input_tensor

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        mock_builder.build_serialized_network.return_value = b"fake_engine_data"

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            exporter.build_engine(precision="fp32", max_batch_size=4)

        # Verify no precision flags were set
        mock_config.set_flag.assert_not_called()

    def test_build_engine_sets_workspace_memory(self, tmp_path):
        """build_engine configures workspace memory correctly."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = "WORKSPACE_TYPE"
        mock_trt.BuilderFlag.FP16 = "FP16_FLAG"

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "images"
        mock_network.get_input.return_value = mock_input_tensor

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        mock_builder.build_serialized_network.return_value = b"fake_engine_data"

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            exporter.build_engine(precision="fp16", workspace_mb=2048)

        # Verify workspace was set: 2048 * (1 << 20) = 2048 MB
        mock_config.set_memory_pool_limit.assert_called_once_with(
            "WORKSPACE_TYPE", 2048 * (1 << 20)
        )

    def test_build_engine_raises_on_parse_failure(self, tmp_path):
        """build_engine raises RuntimeError when ONNX parsing fails."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network

        mock_parser = MagicMock()
        mock_parser.parse.return_value = False
        mock_parser.num_errors = 1
        mock_parser.get_error.return_value = "Invalid ONNX node"
        mock_trt.OnnxParser.return_value = mock_parser

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            with pytest.raises(RuntimeError, match="Failed to parse ONNX model"):
                exporter.build_engine(precision="fp16")

    def test_build_engine_raises_on_build_failure(self, tmp_path):
        """build_engine raises RuntimeError when engine build returns None."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0
        mock_trt.BuilderFlag.FP16 = "FP16_FLAG"

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "images"
        mock_network.get_input.return_value = mock_input_tensor

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        # Simulate build failure
        mock_builder.build_serialized_network.return_value = None

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            with pytest.raises(RuntimeError, match="TensorRT engine build failed"):
                exporter.build_engine(precision="fp16")

    def test_build_engine_optimization_profile_with_batch_size_1(self, tmp_path):
        """build_engine handles max_batch_size=1 correctly (opt=1)."""
        exporter = self._create_exporter_with_onnx(tmp_path)

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0
        mock_trt.BuilderFlag.FP16 = "FP16_FLAG"

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "images"
        mock_network.get_input.return_value = mock_input_tensor

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        mock_builder.build_serialized_network.return_value = b"fake_engine_data"

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            exporter.build_engine(precision="fp16", max_batch_size=1)

        # With max_batch_size=1, opt should be 1 (not 0)
        mock_profile.set_shape.assert_called_once_with(
            "images",
            min=(1, 3, 640, 640),
            opt=(1, 3, 640, 640),
            max=(1, 3, 640, 640),
        )

    def test_build_engine_output_filename_uses_model_stem(self, tmp_path):
        """build_engine names the output file using model stem and precision."""
        model_file = tmp_path / "custom_model.pt"
        model_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        exporter = TensorRTExporter(str(model_file), str(output_dir))
        fake_onnx = output_dir / "custom_model.onnx"
        fake_onnx.write_bytes(b"fake onnx data")
        exporter.onnx_path = str(fake_onnx)

        mock_trt = MagicMock()
        mock_trt.Logger.WARNING = 0
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
        mock_trt.MemoryPoolType.WORKSPACE = 0
        mock_trt.BuilderFlag.FP16 = "FP16_FLAG"

        mock_builder = MagicMock()
        mock_trt.Builder.return_value = mock_builder
        mock_network = MagicMock()
        mock_builder.create_network.return_value = mock_network
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "images"
        mock_network.get_input.return_value = mock_input_tensor

        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_profile = MagicMock()
        mock_builder.create_optimization_profile.return_value = mock_profile

        mock_builder.build_serialized_network.return_value = b"fake_engine_data"

        with patch.dict(sys.modules, {"tensorrt": mock_trt}):
            result = exporter.build_engine(precision="fp16")

        expected_path = str(output_dir / "custom_model_fp16.engine")
        assert result == expected_path
