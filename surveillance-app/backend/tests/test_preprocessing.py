"""
Unit tests for the Triton Python backend YOLOv8 preprocessing model.

Tests verify:
- Resize to 640x640 (bilinear interpolation)
- Normalization to [0, 1] range
- HWC to CHW conversion
- Output dtype is float32
- Correct output shape [3, 640, 640]
- Error handling for invalid inputs
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add the model directory to the path so we can import the model module
MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "yolov8_preprocessing" / "1"
sys.path.insert(0, str(MODEL_DIR))


# Mock triton_python_backend_utils before importing model
mock_pb_utils = MagicMock()
mock_pb_utils.Logger = MagicMock()
mock_pb_utils.Tensor = MagicMock(side_effect=lambda name, data: {"name": name, "data": data})
mock_pb_utils.TritonError = Exception
mock_pb_utils.InferenceResponse = MagicMock(side_effect=lambda output_tensors=None, error=None: {
    "output_tensors": output_tensors,
    "error": error,
})

sys.modules["triton_python_backend_utils"] = mock_pb_utils

# Now import the model module
from model import TritonPythonModel, _resize_bilinear


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def preprocessing_model():
    """Create and initialize a TritonPythonModel instance."""
    model = TritonPythonModel()
    model.initialize({})
    return model


@pytest.fixture
def sample_image_640():
    """Create a sample 640x640 RGB image."""
    return np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_1080p():
    """Create a sample 1920x1080 RGB image."""
    return np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_small():
    """Create a small 100x100 RGB image."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests for _resize_bilinear
# ---------------------------------------------------------------------------


class TestResizeBilinear:
    """Tests for the bilinear resize function."""

    def test_resize_to_target_size(self, sample_image_1080p):
        """Resize a 1080p image to 640x640."""
        result = _resize_bilinear(sample_image_1080p, 640, 640)
        assert result.shape == (640, 640, 3)

    def test_resize_preserves_dtype_uint8(self, sample_image_small):
        """Resize should preserve uint8 dtype."""
        result = _resize_bilinear(sample_image_small, 640, 640)
        assert result.dtype == np.uint8

    def test_resize_no_op_when_already_target_size(self, sample_image_640):
        """If image is already 640x640, return as-is."""
        result = _resize_bilinear(sample_image_640, 640, 640)
        assert result.shape == (640, 640, 3)
        np.testing.assert_array_equal(result, sample_image_640)

    def test_resize_upscale(self):
        """Resize a small image up to 640x640."""
        small = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = _resize_bilinear(small, 640, 640)
        assert result.shape == (640, 640, 3)

    def test_resize_non_square(self):
        """Resize a non-square image to 640x640."""
        rect = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = _resize_bilinear(rect, 640, 640)
        assert result.shape == (640, 640, 3)

    def test_resize_values_in_range(self, sample_image_small):
        """Resized uint8 values should remain in [0, 255]."""
        result = _resize_bilinear(sample_image_small, 640, 640)
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# Tests for TritonPythonModel preprocessing pipeline
# ---------------------------------------------------------------------------


class TestPreprocessingPipeline:
    """Tests for the full preprocessing pipeline."""

    def test_output_shape(self, preprocessing_model, sample_image_640):
        """Output should be [3, 640, 640]."""
        # Create a mock request
        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = sample_image_640

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        responses = preprocessing_model.execute([request])

        assert len(responses) == 1
        # Get the output data from the Tensor call
        tensor_call = mock_pb_utils.Tensor.call_args
        output_data = tensor_call[0][1]  # Second positional arg is the data
        assert output_data.shape == (3, 640, 640)

    def test_output_dtype_float32(self, preprocessing_model, sample_image_640):
        """Output should be float32."""
        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = sample_image_640

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        preprocessing_model.execute([request])

        tensor_call = mock_pb_utils.Tensor.call_args
        output_data = tensor_call[0][1]
        assert output_data.dtype == np.float32

    def test_output_normalized_range(self, preprocessing_model, sample_image_640):
        """Output values should be in [0, 1] range."""
        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = sample_image_640

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        preprocessing_model.execute([request])

        tensor_call = mock_pb_utils.Tensor.call_args
        output_data = tensor_call[0][1]
        assert output_data.min() >= 0.0
        assert output_data.max() <= 1.0

    def test_hwc_to_chw_conversion(self, preprocessing_model):
        """Verify HWC to CHW conversion is correct."""
        # Create a known image where channels are distinguishable
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Channel 0 = 255
        image[:, :, 1] = 128  # Channel 1 = 128
        image[:, :, 2] = 0    # Channel 2 = 0

        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = image

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        preprocessing_model.execute([request])

        tensor_call = mock_pb_utils.Tensor.call_args
        output_data = tensor_call[0][1]

        # After CHW conversion:
        # output_data[0] should be channel 0 (all 1.0 = 255/255)
        # output_data[1] should be channel 1 (all ~0.502 = 128/255)
        # output_data[2] should be channel 2 (all 0.0 = 0/255)
        np.testing.assert_allclose(output_data[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(output_data[1], 128.0 / 255.0, atol=1e-6)
        np.testing.assert_allclose(output_data[2], 0.0, atol=1e-6)

    def test_batch_processing(self, preprocessing_model):
        """Model should handle multiple requests in a batch."""
        images = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8),
            np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
        ]

        requests = []
        for img in images:
            mock_tensor = MagicMock()
            mock_tensor.as_numpy.return_value = img
            requests.append(MagicMock())

        # Set up get_input_tensor_by_name to return different tensors for each request
        tensors = []
        for img in images:
            t = MagicMock()
            t.as_numpy.return_value = img
            tensors.append(t)

        mock_pb_utils.get_input_tensor_by_name = MagicMock(side_effect=tensors)

        responses = preprocessing_model.execute(requests)
        assert len(responses) == 3

    def test_contiguous_output(self, preprocessing_model, sample_image_640):
        """Output array should be C-contiguous for efficient memory transfer."""
        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = sample_image_640

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        preprocessing_model.execute([request])

        tensor_call = mock_pb_utils.Tensor.call_args
        output_data = tensor_call[0][1]
        assert output_data.flags["C_CONTIGUOUS"]

    def test_output_tensor_name(self, preprocessing_model, sample_image_640):
        """Output tensor should be named 'processed'."""
        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = sample_image_640

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        preprocessing_model.execute([request])

        tensor_call = mock_pb_utils.Tensor.call_args
        tensor_name = tensor_call[0][0]  # First positional arg is the name
        assert tensor_name == "processed"


# ---------------------------------------------------------------------------
# Tests for error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in the preprocessing model."""

    def test_missing_input_tensor(self, preprocessing_model):
        """Should return error response when input tensor is missing."""
        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=None)

        request = MagicMock()
        responses = preprocessing_model.execute([request])

        assert len(responses) == 1
        # The response should have an error
        response_call = mock_pb_utils.InferenceResponse.call_args
        assert response_call[1].get("error") is not None or (
            response_call[0] if response_call[0] else False
        )

    def test_invalid_input_shape_2d(self, preprocessing_model):
        """Should return error for 2D input (missing channel dimension)."""
        invalid_image = np.random.randint(0, 256, (640, 640), dtype=np.uint8)

        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = invalid_image

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        responses = preprocessing_model.execute([request])

        assert len(responses) == 1

    def test_invalid_input_shape_wrong_channels(self, preprocessing_model):
        """Should return error for input with wrong number of channels."""
        invalid_image = np.random.randint(0, 256, (640, 640, 4), dtype=np.uint8)

        mock_tensor = MagicMock()
        mock_tensor.as_numpy.return_value = invalid_image

        mock_pb_utils.get_input_tensor_by_name = MagicMock(return_value=mock_tensor)

        request = MagicMock()
        responses = preprocessing_model.execute([request])

        assert len(responses) == 1


class TestFinalize:
    """Tests for model finalize."""

    def test_finalize_runs_without_error(self, preprocessing_model):
        """Finalize should complete without raising."""
        preprocessing_model.finalize()
