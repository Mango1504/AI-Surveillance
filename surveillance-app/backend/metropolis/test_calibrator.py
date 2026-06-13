"""Unit tests for the EntropyCalibrator class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from metropolis.calibrator import EntropyCalibrator, _IMAGE_EXTENSIONS


@pytest.fixture
def calibration_dir(tmp_path):
    """Create a temporary directory with fake calibration images."""
    # Create small test images using numpy (saved as raw .png via cv2)
    try:
        import cv2

        for i in range(10):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"image_{i:03d}.jpg"), img)
    except ImportError:
        # If cv2 not available, create dummy files with image extensions
        for i in range(10):
            (tmp_path / f"image_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff" + os.urandom(100))

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    """Create an empty temporary directory."""
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty


@pytest.fixture
def mixed_dir(tmp_path):
    """Create a directory with both image and non-image files."""
    try:
        import cv2

        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
    except ImportError:
        for i in range(3):
            (tmp_path / f"img_{i}.png").write_bytes(b"\x89PNG" + os.urandom(50))

    # Non-image files
    (tmp_path / "readme.txt").write_text("not an image")
    (tmp_path / "data.csv").write_text("col1,col2\n1,2")
    (tmp_path / "config.yaml").write_text("key: value")

    return tmp_path


class TestEntropyCalibrator:
    """Tests for EntropyCalibrator initialization and configuration."""

    def test_init_valid_directory(self, calibration_dir):
        """Calibrator initializes with a valid image directory."""
        cal = EntropyCalibrator(
            str(calibration_dir), batch_size=4, input_shape=(3, 640, 640)
        )
        assert cal.batch_size == 4
        assert cal.input_shape == (3, 640, 640)
        assert cal.calibration_data == str(calibration_dir)

    def test_init_nonexistent_directory(self):
        """Calibrator raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError, match="not found"):
            EntropyCalibrator("/nonexistent/path/to/images")

    def test_init_empty_directory(self, empty_dir):
        """Calibrator raises ValueError for directory with no images."""
        with pytest.raises(ValueError, match="No valid image files"):
            EntropyCalibrator(str(empty_dir))

    def test_init_filters_non_image_files(self, mixed_dir):
        """Calibrator only picks up files with image extensions."""
        cal = EntropyCalibrator(str(mixed_dir), batch_size=2)
        # Should find exactly 3 .png files, not the .txt/.csv/.yaml
        assert len(cal._image_paths) == 3

    def test_init_not_a_directory(self, tmp_path):
        """Calibrator raises FileNotFoundError if path is a file, not dir."""
        file_path = tmp_path / "somefile.txt"
        file_path.write_text("hello")
        with pytest.raises(FileNotFoundError, match="not a directory"):
            EntropyCalibrator(str(file_path))

    def test_cache_file_location(self, calibration_dir):
        """Cache file is stored in the calibration directory."""
        cal = EntropyCalibrator(str(calibration_dir))
        expected = str(calibration_dir / "calibration.cache")
        assert cal.cache_file == expected

    def test_get_batch_size(self, calibration_dir):
        """get_batch_size returns the configured batch size."""
        cal = EntropyCalibrator(str(calibration_dir), batch_size=16)
        assert cal.get_batch_size() == 16

    def test_default_input_shape(self, calibration_dir):
        """Default input shape is (3, 640, 640)."""
        cal = EntropyCalibrator(str(calibration_dir))
        assert cal.input_shape == (3, 640, 640)


class TestGetBatch:
    """Tests for the get_batch method."""

    def test_get_batch_returns_pointer_list(self, calibration_dir):
        """get_batch returns a list with a single integer (memory pointer)."""
        cal = EntropyCalibrator(str(calibration_dir), batch_size=4)
        result = cal.get_batch(["input"])
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], int)

    def test_get_batch_exhausts_images(self, calibration_dir):
        """get_batch returns None after all images are consumed."""
        cal = EntropyCalibrator(str(calibration_dir), batch_size=5)
        # 10 images / 5 per batch = 2 batches
        batch1 = cal.get_batch(["input"])
        assert batch1 is not None
        batch2 = cal.get_batch(["input"])
        assert batch2 is not None
        batch3 = cal.get_batch(["input"])
        assert batch3 is None

    def test_get_batch_handles_partial_batch(self, calibration_dir):
        """get_batch handles the last batch with fewer images than batch_size."""
        cal = EntropyCalibrator(str(calibration_dir), batch_size=7)
        # 10 images: first batch=7, second batch=3 (partial)
        batch1 = cal.get_batch(["input"])
        assert batch1 is not None
        batch2 = cal.get_batch(["input"])
        assert batch2 is not None
        batch3 = cal.get_batch(["input"])
        assert batch3 is None

    def test_get_batch_with_single_image(self, tmp_path):
        """get_batch works with fewer images than batch_size."""
        try:
            import cv2

            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / "single.jpg"), img)
        except ImportError:
            (tmp_path / "single.jpg").write_bytes(b"\xff\xd8\xff" + os.urandom(100))

        cal = EntropyCalibrator(str(tmp_path), batch_size=8)
        result = cal.get_batch(["input"])
        assert result is not None
        # Next call should return None
        assert cal.get_batch(["input"]) is None


class TestCalibrationCache:
    """Tests for read/write calibration cache."""

    def test_read_cache_returns_none_when_missing(self, calibration_dir):
        """read_calibration_cache returns None when no cache file exists."""
        cal = EntropyCalibrator(str(calibration_dir))
        assert cal.read_calibration_cache() is None

    def test_write_and_read_cache(self, calibration_dir):
        """write_calibration_cache creates a file that read_calibration_cache can load."""
        cal = EntropyCalibrator(str(calibration_dir))
        test_data = b"fake_calibration_data_12345"

        cal.write_calibration_cache(test_data)

        # Verify file was created
        assert os.path.exists(cal.cache_file)

        # Verify read returns the same data
        result = cal.read_calibration_cache()
        assert result == test_data

    def test_cache_file_is_in_calibration_dir(self, calibration_dir):
        """Cache file is written inside the calibration data directory."""
        cal = EntropyCalibrator(str(calibration_dir))
        cal.write_calibration_cache(b"test_cache")

        cache_path = Path(cal.cache_file)
        assert cache_path.parent == calibration_dir
        assert cache_path.name == "calibration.cache"


class TestImagePreprocessing:
    """Tests for the _load_and_preprocess method."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a single test image."""
        try:
            import cv2

            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            path = str(tmp_path / "test.jpg")
            cv2.imwrite(path, img)
            return path
        except ImportError:
            pytest.skip("cv2 required for preprocessing tests")

    def test_preprocess_output_shape(self, calibration_dir, sample_image):
        """Preprocessed image has correct CHW shape."""
        cal = EntropyCalibrator(str(calibration_dir), input_shape=(3, 640, 640))
        result = cal._load_and_preprocess(sample_image, 640, 640)
        assert result is not None
        assert result.shape == (3, 640, 640)

    def test_preprocess_output_dtype(self, calibration_dir, sample_image):
        """Preprocessed image is float32."""
        cal = EntropyCalibrator(str(calibration_dir))
        result = cal._load_and_preprocess(sample_image, 640, 640)
        assert result is not None
        assert result.dtype == np.float32

    def test_preprocess_output_range(self, calibration_dir, sample_image):
        """Preprocessed image values are normalized to [0, 1]."""
        cal = EntropyCalibrator(str(calibration_dir))
        result = cal._load_and_preprocess(sample_image, 640, 640)
        assert result is not None
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_custom_shape(self, calibration_dir, sample_image):
        """Preprocessing respects custom input shape."""
        cal = EntropyCalibrator(
            str(calibration_dir), input_shape=(3, 320, 320)
        )
        result = cal._load_and_preprocess(sample_image, 320, 320)
        assert result is not None
        assert result.shape == (3, 320, 320)

    def test_preprocess_invalid_path(self, calibration_dir):
        """Preprocessing returns None for non-existent file."""
        cal = EntropyCalibrator(str(calibration_dir))
        result = cal._load_and_preprocess("/nonexistent/image.jpg", 640, 640)
        assert result is None

    def test_preprocess_corrupt_file(self, calibration_dir, tmp_path):
        """Preprocessing returns None for corrupt/unreadable image."""
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"not a real image file content")
        cal = EntropyCalibrator(str(calibration_dir))
        result = cal._load_and_preprocess(str(corrupt), 640, 640)
        assert result is None
