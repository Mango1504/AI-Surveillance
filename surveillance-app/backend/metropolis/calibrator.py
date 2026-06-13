"""INT8 entropy calibrator for TensorRT engine building.

Implements TensorRT's IInt8EntropyCalibrator2 interface to provide calibration
data for INT8 quantization. Reads representative images from a directory,
preprocesses them (resize, normalize, HWC→CHW), and feeds batches to the
TensorRT builder for computing optimal quantization ranges.

Typical usage:
    calibrator = EntropyCalibrator(
        calibration_data="path/to/calibration_images/",
        batch_size=8,
        input_shape=(3, 640, 640),
    )
    config.int8_calibrator = calibrator
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Supported image extensions for calibration
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Try importing pycuda for GPU memory allocation; fall back to numpy-only mode
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — initializes CUDA context

    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    logger.warning(
        "pycuda not available. EntropyCalibrator will use numpy fallback "
        "(suitable for testing only, not for actual TensorRT calibration)."
    )

# Try importing tensorrt for the base class
try:
    import tensorrt as trt

    _TRT_BASE_CLASS = trt.IInt8EntropyCalibrator2
except ImportError:
    # Provide a stub base class so the module can be imported without tensorrt
    _TRT_BASE_CLASS = object


class EntropyCalibrator(_TRT_BASE_CLASS):
    """INT8 entropy calibrator that reads calibration images from a directory.

    Implements TensorRT's IInt8EntropyCalibrator2 interface. Reads images from
    the specified directory, preprocesses them to match model input requirements,
    and provides batches to the TensorRT builder for INT8 calibration.

    The calibration cache is stored as ``calibration.cache`` in the same
    directory as the calibration images for reuse across builds.

    Attributes:
        calibration_data: Path to the directory containing calibration images.
        batch_size: Number of images per calibration batch.
        input_shape: Model input shape as (C, H, W).
        cache_file: Path to the calibration cache file.
    """

    def __init__(
        self,
        calibration_data: str,
        batch_size: int = 8,
        input_shape: tuple = (3, 640, 640),
    ) -> None:
        """Initialize the calibrator with image directory and batch parameters.

        Args:
            calibration_data: Path to a directory containing representative
                calibration images (jpg, png, bmp, etc.).
            batch_size: Number of images to process per calibration batch.
            input_shape: Model input tensor shape as (channels, height, width).
                Default is (3, 640, 640) for YOLOv8.

        Raises:
            FileNotFoundError: If calibration_data directory does not exist.
            ValueError: If no valid image files are found in the directory.
        """
        # Initialize the TensorRT base class if available
        if _TRT_BASE_CLASS is not object:
            super().__init__()

        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.input_shape = input_shape

        # Validate calibration directory exists
        cal_dir = Path(calibration_data)
        if not cal_dir.exists():
            raise FileNotFoundError(
                f"Calibration data directory not found: {calibration_data}"
            )
        if not cal_dir.is_dir():
            raise FileNotFoundError(
                f"Calibration data path is not a directory: {calibration_data}"
            )

        # Collect image file paths
        self._image_paths = sorted(
            [
                str(p)
                for p in cal_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            ]
        )

        if not self._image_paths:
            raise ValueError(
                f"No valid image files found in calibration directory: "
                f"{calibration_data}. Supported extensions: {_IMAGE_EXTENSIONS}"
            )

        logger.info(
            "EntropyCalibrator initialized: %d images, batch_size=%d, "
            "input_shape=%s",
            len(self._image_paths),
            batch_size,
            input_shape,
        )

        # Cache file stored alongside calibration images
        self.cache_file = str(cal_dir / "calibration.cache")

        # Batch tracking state
        self._current_index = 0
        self._batch_count = 0

        # Allocate device memory for one batch
        batch_nbytes = (
            batch_size
            * int(np.prod(input_shape))
            * np.dtype(np.float32).itemsize
        )

        if PYCUDA_AVAILABLE:
            self._device_input = cuda.mem_alloc(batch_nbytes)
            logger.debug("Allocated %d bytes of GPU memory for calibration.", batch_nbytes)
        else:
            # Fallback: store a numpy buffer and a fake pointer for testing
            self._host_buffer = np.zeros(
                (batch_size, *input_shape), dtype=np.float32
            )
            self._device_input = self._host_buffer.ctypes.data
            logger.debug(
                "Using numpy fallback buffer (%d bytes) for calibration.",
                batch_nbytes,
            )

    def get_batch_size(self) -> int:
        """Return the calibration batch size.

        Returns:
            The number of images per calibration batch.
        """
        return self.batch_size

    def get_batch(self, names: list) -> Optional[list]:
        """Read and preprocess the next batch of calibration images.

        Reads up to ``batch_size`` images from the calibration directory,
        preprocesses each (resize, BGR→RGB, normalize, HWC→CHW), and copies
        the batch to device memory.

        Args:
            names: List of input tensor names (provided by TensorRT, unused).

        Returns:
            A list containing the device memory pointer for the batch, or
            None when all images have been consumed.
        """
        if self._current_index >= len(self._image_paths):
            logger.info(
                "Calibration complete: processed %d batches (%d images total).",
                self._batch_count,
                len(self._image_paths),
            )
            return None

        # Determine batch slice
        batch_end = min(
            self._current_index + self.batch_size, len(self._image_paths)
        )
        batch_paths = self._image_paths[self._current_index:batch_end]
        actual_batch_size = len(batch_paths)

        # Preprocess images
        target_h = self.input_shape[1]
        target_w = self.input_shape[2]
        batch_data = np.zeros(
            (self.batch_size, *self.input_shape), dtype=np.float32
        )

        for i, img_path in enumerate(batch_paths):
            img = self._load_and_preprocess(img_path, target_h, target_w)
            if img is not None:
                batch_data[i] = img
            else:
                logger.warning(
                    "Failed to load calibration image: %s (using zeros)",
                    img_path,
                )

        # Copy to device memory
        if PYCUDA_AVAILABLE:
            cuda.memcpy_htod(self._device_input, batch_data.ravel())
        else:
            # Numpy fallback: just store the data
            np.copyto(self._host_buffer, batch_data)
            self._device_input = self._host_buffer.ctypes.data

        self._current_index = batch_end
        self._batch_count += 1

        logger.debug(
            "Calibration batch %d: %d images (index %d-%d of %d)",
            self._batch_count,
            actual_batch_size,
            self._current_index - actual_batch_size,
            self._current_index - 1,
            len(self._image_paths),
        )

        return [int(self._device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        """Read cached calibration data from file if it exists.

        TensorRT calls this before calibration begins. If a valid cache
        exists, calibration is skipped and the cached ranges are used.

        Returns:
            The cached calibration data as bytes, or None if no cache exists.
        """
        if os.path.exists(self.cache_file):
            logger.info("Reading calibration cache from: %s", self.cache_file)
            with open(self.cache_file, "rb") as f:
                cache_data = f.read()
            logger.info(
                "Loaded calibration cache (%d bytes).", len(cache_data)
            )
            return cache_data

        logger.info("No calibration cache found at: %s", self.cache_file)
        return None

    def write_calibration_cache(self, cache) -> None:
        """Write calibration data to file for reuse in future builds.

        TensorRT calls this after calibration completes. The cache file
        allows subsequent engine builds to skip the calibration step.

        Args:
            cache: Calibration data bytes provided by TensorRT.
        """
        logger.info("Writing calibration cache to: %s", self.cache_file)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.info(
            "Calibration cache written (%d bytes).", len(cache)
        )

    def _load_and_preprocess(
        self, image_path: str, target_h: int, target_w: int
    ) -> Optional[np.ndarray]:
        """Load and preprocess a single image for calibration.

        Preprocessing steps:
            1. Read image with OpenCV (BGR format)
            2. Resize to target dimensions (target_w × target_h)
            3. Convert BGR → RGB
            4. Normalize pixel values to [0, 1] (divide by 255.0)
            5. Transpose from HWC to CHW format
            6. Convert to float32

        Args:
            image_path: Path to the image file.
            target_h: Target height for resizing.
            target_w: Target width for resizing.

        Returns:
            Preprocessed image as a float32 numpy array with shape (C, H, W),
            or None if the image could not be loaded.
        """
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                logger.warning("cv2.imread returned None for: %s", image_path)
                return None

            # Resize to target dimensions
            img = cv2.resize(img, (target_w, target_h))

            # BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            # HWC → CHW
            img = img.transpose(2, 0, 1)

            return img

        except ImportError:
            logger.error(
                "OpenCV (cv2) is required for image preprocessing but not installed."
            )
            return None
        except Exception as e:
            logger.error("Error preprocessing image %s: %s", image_path, e)
            return None
