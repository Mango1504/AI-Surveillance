"""Appearance embedding extraction for cross-camera re-identification.

Provides an EmbeddingExtractor that produces fixed-size feature vectors
(128 dimensions) from cropped object regions. Supports a lightweight
fallback based on color histograms and spatial average pooling when no
dedicated ReID model is available.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Fixed crop size for embedding extraction
_CROP_HEIGHT = 128
_CROP_WIDTH = 64

# Histogram parameters: 16 bins per channel × 3 channels = 48 values
_HIST_BINS = 16
_HIST_FEATURES = _HIST_BINS * 3  # 48

# Spatial pooling grid: 4 columns × 5 rows × 4 features per cell
# We use a 4×5 grid with average R, G, B, and luminance per cell = 80 values
_GRID_COLS = 4
_GRID_ROWS = 5
_SPATIAL_FEATURES_PER_CELL = 4  # R, G, B, luminance
_SPATIAL_FEATURES = _GRID_COLS * _GRID_ROWS * _SPATIAL_FEATURES_PER_CELL  # 80

# Total embedding dimension: 48 + 80 = 128
EMBEDDING_DIM = _HIST_FEATURES + _SPATIAL_FEATURES  # 128


class EmbeddingExtractor:
    """Extracts appearance embeddings from cropped object regions.

    Produces a 128-dimensional L2-normalized feature vector for each object.
    When no ReID model is provided, uses a lightweight fallback combining
    color histograms and spatial average pooling.

    Args:
        model_path: Optional path to a ReID model file. If None, uses the
            histogram + spatial pooling fallback.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the embedding extractor.

        Args:
            model_path: Optional path to a ReID model (ONNX or TensorRT).
                If None, the lightweight fallback method is used.
        """
        self._model = None
        self._model_path = model_path

        if model_path is not None:
            self._model = self._load_model(model_path)
            if self._model is not None:
                logger.info(
                    "EmbeddingExtractor initialized with ReID model: %s",
                    model_path,
                )
            else:
                logger.warning(
                    "Failed to load ReID model from %s, using fallback.",
                    model_path,
                )
        else:
            logger.info(
                "EmbeddingExtractor initialized with histogram+pooling fallback "
                "(embedding_dim=%d).",
                EMBEDDING_DIM,
            )

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of produced embeddings."""
        return EMBEDDING_DIM

    def _load_model(self, model_path: str) -> object | None:
        """Attempt to load a ReID model from the given path.

        Returns the loaded model object, or None if loading fails.
        """
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(model_path)
            logger.info("Loaded ONNX ReID model: %s", model_path)
            return session
        except Exception as exc:
            logger.warning("Could not load ReID model: %s", exc)
            return None

    def extract(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> list[float]:
        """Extract an appearance embedding for a single object.

        Args:
            frame: The full video frame as a BGR numpy array (H, W, 3).
            bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            A 128-dimensional L2-normalized feature vector as a list of floats.
        """
        crop = self._crop_and_resize(frame, bbox)

        if self._model is not None:
            return self._extract_with_model(crop)

        return self._extract_fallback(crop)

    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[list[float]]:
        """Extract appearance embeddings for multiple objects in a frame.

        Args:
            frame: The full video frame as a BGR numpy array (H, W, 3).
            bboxes: List of bounding boxes as (x1, y1, x2, y2).

        Returns:
            List of 128-dimensional L2-normalized feature vectors.
        """
        if not bboxes:
            return []

        embeddings: list[list[float]] = []
        for bbox in bboxes:
            embeddings.append(self.extract(frame, bbox))

        return embeddings

    def _crop_and_resize(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop the object region from the frame and resize to fixed size.

        Args:
            frame: Full frame as BGR numpy array.
            bbox: Bounding box (x1, y1, x2, y2).

        Returns:
            Cropped and resized image of shape (_CROP_HEIGHT, _CROP_WIDTH, 3).
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Clamp coordinates to frame boundaries
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))

        crop = frame[y1:y2, x1:x2]

        # Resize to fixed dimensions using OpenCV
        try:
            import cv2

            resized = cv2.resize(
                crop, (_CROP_WIDTH, _CROP_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
        except ImportError:
            # Fallback: nearest-neighbor resize with numpy
            resized = self._numpy_resize(crop, _CROP_HEIGHT, _CROP_WIDTH)

        return resized

    @staticmethod
    def _numpy_resize(
        img: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        """Simple nearest-neighbor resize using numpy (fallback for no cv2)."""
        src_h, src_w = img.shape[:2]
        row_indices = (np.arange(target_h) * src_h // target_h).astype(int)
        col_indices = (np.arange(target_w) * src_w // target_w).astype(int)
        return img[np.ix_(row_indices, col_indices)]

    def _extract_fallback(self, crop: np.ndarray) -> list[float]:
        """Extract embedding using color histogram + spatial average pooling.

        The fallback produces a 128-dimensional vector:
        - 48 values from per-channel color histograms (16 bins × 3 channels)
        - 80 values from spatial average pooling (4×5 grid × 4 features/cell)

        The final vector is L2-normalized.

        Args:
            crop: Resized object crop of shape (_CROP_HEIGHT, _CROP_WIDTH, 3).

        Returns:
            128-dimensional L2-normalized feature vector.
        """
        # Part 1: Color histogram features (48 dimensions)
        hist_features = self._compute_color_histogram(crop)

        # Part 2: Spatial average pooling features (80 dimensions)
        spatial_features = self._compute_spatial_pooling(crop)

        # Concatenate: 48 + 80 = 128
        embedding = np.concatenate([hist_features, spatial_features])

        # L2 normalize
        embedding = self._l2_normalize(embedding)

        return embedding.tolist()

    def _compute_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """Compute per-channel color histogram.

        Args:
            crop: BGR image of shape (H, W, 3), dtype uint8.

        Returns:
            Numpy array of shape (48,) with normalized histogram values.
        """
        hist_features = np.zeros(_HIST_FEATURES, dtype=np.float32)

        for ch in range(3):
            channel_data = crop[:, :, ch].ravel()
            # Compute histogram with 16 bins in range [0, 256)
            hist, _ = np.histogram(channel_data, bins=_HIST_BINS, range=(0, 256))
            # Normalize histogram to sum to 1
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist.astype(np.float32) / hist_sum
            hist_features[ch * _HIST_BINS : (ch + 1) * _HIST_BINS] = hist

        return hist_features

    def _compute_spatial_pooling(self, crop: np.ndarray) -> np.ndarray:
        """Compute spatial average pooling over a grid.

        Divides the crop into a 4×5 grid and computes average R, G, B,
        and luminance for each cell.

        Args:
            crop: BGR image of shape (_CROP_HEIGHT, _CROP_WIDTH, 3).

        Returns:
            Numpy array of shape (80,) with spatial pooling features.
        """
        h, w = crop.shape[:2]
        cell_h = h // _GRID_ROWS
        cell_w = w // _GRID_COLS

        spatial_features = np.zeros(_SPATIAL_FEATURES, dtype=np.float32)

        # Convert to float for computation
        crop_float = crop.astype(np.float32) / 255.0

        idx = 0
        for row in range(_GRID_ROWS):
            for col in range(_GRID_COLS):
                y_start = row * cell_h
                y_end = (row + 1) * cell_h if row < _GRID_ROWS - 1 else h
                x_start = col * cell_w
                x_end = (col + 1) * cell_w if col < _GRID_COLS - 1 else w

                cell = crop_float[y_start:y_end, x_start:x_end]

                # Average B, G, R channels
                avg_b = cell[:, :, 0].mean()
                avg_g = cell[:, :, 1].mean()
                avg_r = cell[:, :, 2].mean()

                # Luminance (approximate)
                luminance = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b

                spatial_features[idx] = avg_r
                spatial_features[idx + 1] = avg_g
                spatial_features[idx + 2] = avg_b
                spatial_features[idx + 3] = luminance
                idx += 4

        return spatial_features

    def _extract_with_model(self, crop: np.ndarray) -> list[float]:
        """Extract embedding using the loaded ReID model.

        Args:
            crop: Resized object crop of shape (_CROP_HEIGHT, _CROP_WIDTH, 3).

        Returns:
            128-dimensional L2-normalized feature vector.
        """
        try:
            # Prepare input: normalize and convert to NCHW float32
            input_tensor = crop.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
            input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim

            input_name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {input_name: input_tensor})

            embedding = outputs[0].flatten()

            # Truncate or pad to EMBEDDING_DIM
            if len(embedding) > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            elif len(embedding) < EMBEDDING_DIM:
                padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                padded[: len(embedding)] = embedding
                embedding = padded

            embedding = self._l2_normalize(embedding)
            return embedding.tolist()

        except Exception as exc:
            logger.warning(
                "ReID model inference failed, using fallback: %s", exc
            )
            return self._extract_fallback(crop)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector. Returns zero vector if norm is zero."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec
