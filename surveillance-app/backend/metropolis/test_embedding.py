"""Unit tests for the EmbeddingExtractor module."""

import numpy as np
import pytest

from .embedding import EMBEDDING_DIM, EmbeddingExtractor


@pytest.fixture
def extractor() -> EmbeddingExtractor:
    """Create an EmbeddingExtractor with no model (fallback mode)."""
    return EmbeddingExtractor(model_path=None)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a synthetic BGR frame (480x640x3)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox() -> tuple[int, int, int, int]:
    """A valid bounding box within the sample frame."""
    return (100, 50, 250, 300)


class TestEmbeddingExtractorInit:
    """Tests for EmbeddingExtractor initialization."""

    def test_init_no_model(self) -> None:
        """Extractor initializes without a model path (fallback mode)."""
        ext = EmbeddingExtractor(model_path=None)
        assert ext._model is None
        assert ext.embedding_dim == EMBEDDING_DIM

    def test_init_invalid_model_path(self) -> None:
        """Extractor falls back gracefully when model path is invalid."""
        ext = EmbeddingExtractor(model_path="/nonexistent/model.onnx")
        assert ext._model is None
        assert ext.embedding_dim == EMBEDDING_DIM


class TestExtractSingle:
    """Tests for single-object embedding extraction."""

    def test_output_dimension(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray, sample_bbox: tuple
    ) -> None:
        """Embedding has exactly EMBEDDING_DIM dimensions."""
        embedding = extractor.extract(sample_frame, sample_bbox)
        assert len(embedding) == EMBEDDING_DIM

    def test_output_is_list_of_floats(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray, sample_bbox: tuple
    ) -> None:
        """Embedding is a list of float values."""
        embedding = extractor.extract(sample_frame, sample_bbox)
        assert isinstance(embedding, list)
        assert all(isinstance(v, float) for v in embedding)

    def test_l2_normalized(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray, sample_bbox: tuple
    ) -> None:
        """Embedding vector is L2-normalized (unit length)."""
        embedding = extractor.extract(sample_frame, sample_bbox)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray, sample_bbox: tuple
    ) -> None:
        """Same input produces the same embedding."""
        emb1 = extractor.extract(sample_frame, sample_bbox)
        emb2 = extractor.extract(sample_frame, sample_bbox)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_different_regions_produce_different_embeddings(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray
    ) -> None:
        """Different bounding boxes produce different embeddings."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (300, 200, 500, 400)
        emb1 = extractor.extract(sample_frame, bbox1)
        emb2 = extractor.extract(sample_frame, bbox2)
        # They should not be identical
        assert emb1 != emb2


class TestExtractBatch:
    """Tests for batch embedding extraction."""

    def test_empty_batch(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray
    ) -> None:
        """Empty bbox list returns empty result."""
        result = extractor.extract_batch(sample_frame, [])
        assert result == []

    def test_batch_dimensions(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray
    ) -> None:
        """Batch extraction returns correct number of embeddings."""
        bboxes = [(10, 10, 100, 100), (200, 200, 400, 400), (50, 50, 150, 300)]
        result = extractor.extract_batch(sample_frame, bboxes)
        assert len(result) == 3
        for emb in result:
            assert len(emb) == EMBEDDING_DIM

    def test_batch_matches_individual(
        self, extractor: EmbeddingExtractor, sample_frame: np.ndarray
    ) -> None:
        """Batch extraction produces same results as individual extraction."""
        bboxes = [(10, 10, 100, 100), (200, 200, 400, 400)]
        batch_result = extractor.extract_batch(sample_frame, bboxes)

        for i, bbox in enumerate(bboxes):
            individual = extractor.extract(sample_frame, bbox)
            np.testing.assert_array_almost_equal(batch_result[i], individual)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_bbox_at_frame_boundary(self, extractor: EmbeddingExtractor) -> None:
        """Bbox extending beyond frame boundaries is clamped."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bbox extends beyond frame
        bbox = (80, 80, 150, 150)
        embedding = extractor.extract(frame, bbox)
        assert len(embedding) == EMBEDDING_DIM
        assert not any(np.isnan(embedding))

    def test_very_small_bbox(self, extractor: EmbeddingExtractor) -> None:
        """Very small bounding box (1x1 pixel) still produces valid embedding."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bbox = (50, 50, 51, 51)
        embedding = extractor.extract(frame, bbox)
        assert len(embedding) == EMBEDDING_DIM
        assert not any(np.isnan(embedding))

    def test_uniform_color_frame(self, extractor: EmbeddingExtractor) -> None:
        """Uniform color frame produces a valid normalized embedding."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 100
        bbox = (10, 10, 190, 190)
        embedding = extractor.extract(frame, bbox)
        assert len(embedding) == EMBEDDING_DIM
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_negative_bbox_coords_clamped(self, extractor: EmbeddingExtractor) -> None:
        """Negative bbox coordinates are clamped to zero."""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 50
        bbox = (-10, -10, 50, 50)
        embedding = extractor.extract(frame, bbox)
        assert len(embedding) == EMBEDDING_DIM
        assert not any(np.isnan(embedding))
