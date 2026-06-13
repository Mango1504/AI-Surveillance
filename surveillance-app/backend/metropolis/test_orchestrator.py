"""Unit tests for PipelineOrchestrator, Capabilities, and Detection."""

import time
from unittest.mock import patch, MagicMock

import pytest

from .config import MetropolisConfig
from .orchestrator import Capabilities, Detection, PipelineOrchestrator


# ---------------------------------------------------------------------------
# Detection dataclass tests
# ---------------------------------------------------------------------------


class TestDetection:
    def test_creation_with_required_fields(self):
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=(10, 20, 100, 200),
            camera_id=1,
            timestamp=1000.0,
        )
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.95
        assert det.bbox == (10, 20, 100, 200)
        assert det.camera_id == 1
        assert det.timestamp == 1000.0
        assert det.track_id is None
        assert det.embedding is None

    def test_creation_with_optional_fields(self):
        det = Detection(
            class_id=1,
            class_name="phone",
            confidence=0.88,
            bbox=(50, 60, 150, 160),
            camera_id=2,
            timestamp=2000.0,
            track_id=42,
            embedding=[0.1, 0.2, 0.3],
        )
        assert det.track_id == 42
        assert det.embedding == [0.1, 0.2, 0.3]

    def test_to_dict(self):
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9567,
            bbox=(10, 20, 100, 200),
            camera_id=1,
            timestamp=1000.0,
            track_id=5,
        )
        d = det.to_dict()
        assert d["class"] == "person"
        assert d["class_id"] == 0
        assert d["confidence"] == 0.957  # rounded to 3 decimals
        assert d["bbox"] == [10, 20, 100, 200]
        assert d["camera_id"] == 1
        assert d["track_id"] == 5


# ---------------------------------------------------------------------------
# Capabilities dataclass tests
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_defaults(self):
        caps = Capabilities()
        assert caps.has_gpu is False
        assert caps.gpu_name is None
        assert caps.has_tensorrt is False
        assert caps.has_deepstream is False
        assert caps.has_triton is False
        assert caps.triton_url is None

    def test_custom_values(self):
        caps = Capabilities(
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            has_tensorrt=True,
            has_deepstream=True,
            has_triton=True,
            triton_url="localhost:8001",
        )
        assert caps.has_gpu is True
        assert caps.gpu_name == "NVIDIA RTX 4090"


# ---------------------------------------------------------------------------
# PipelineOrchestrator tests
# ---------------------------------------------------------------------------


class TestPipelineOrchestrator:
    def _make_orchestrator(self, pipeline_mode: str = "auto") -> PipelineOrchestrator:
        config = MetropolisConfig(pipeline_mode=pipeline_mode)
        return PipelineOrchestrator(config)

    def test_init(self):
        orch = self._make_orchestrator()
        assert orch.config.pipeline_mode == "auto"
        assert orch.capabilities is None
        assert orch.active_pipeline is None
        assert orch.is_running is False

    # --- Capability detection ---

    @patch.object(PipelineOrchestrator, "_detect_gpu", return_value=(True, "RTX 4090"))
    @patch.object(PipelineOrchestrator, "_detect_tensorrt", return_value=True)
    @patch.object(PipelineOrchestrator, "_detect_deepstream", return_value=True)
    @patch.object(PipelineOrchestrator, "_detect_triton", return_value=True)
    def test_detect_capabilities_all_available(self, mock_triton, mock_ds, mock_trt, mock_gpu):
        orch = self._make_orchestrator()
        caps = orch.detect_capabilities()
        assert caps.has_gpu is True
        assert caps.gpu_name == "RTX 4090"
        assert caps.has_tensorrt is True
        assert caps.has_deepstream is True
        assert caps.has_triton is True

    @patch.object(PipelineOrchestrator, "_detect_gpu", return_value=(False, None))
    @patch.object(PipelineOrchestrator, "_detect_tensorrt", return_value=False)
    @patch.object(PipelineOrchestrator, "_detect_deepstream", return_value=False)
    @patch.object(PipelineOrchestrator, "_detect_triton", return_value=False)
    def test_detect_capabilities_none_available(self, mock_triton, mock_ds, mock_trt, mock_gpu):
        orch = self._make_orchestrator()
        caps = orch.detect_capabilities()
        assert caps.has_gpu is False
        assert caps.has_tensorrt is False
        assert caps.has_deepstream is False
        assert caps.has_triton is False

    # --- Pipeline selection: auto mode ---

    @patch.object(PipelineOrchestrator, "_detect_gpu", return_value=(True, "RTX 4090"))
    @patch.object(PipelineOrchestrator, "_detect_tensorrt", return_value=True)
    @patch.object(PipelineOrchestrator, "_detect_deepstream", return_value=True)
    @patch.object(PipelineOrchestrator, "_detect_triton", return_value=True)
    def test_select_pipeline_auto_metropolis(self, *mocks):
        orch = self._make_orchestrator("auto")
        pipeline = orch.select_pipeline()
        assert pipeline == "metropolis"

    @patch.object(PipelineOrchestrator, "_detect_gpu", return_value=(True, "RTX 3080"))
    @patch.object(PipelineOrchestrator, "_detect_tensorrt", return_value=True)
    @patch.object(PipelineOrchestrator, "_detect_deepstream", return_value=False)
    @patch.object(PipelineOrchestrator, "_detect_triton", return_value=False)
    def test_select_pipeline_auto_hybrid(self, *mocks):
        orch = self._make_orchestrator("auto")
        pipeline = orch.select_pipeline()
        assert pipeline == "hybrid"

    @patch.object(PipelineOrchestrator, "_detect_gpu", return_value=(False, None))
    @patch.object(PipelineOrchestrator, "_detect_tensorrt", return_value=False)
    @patch.object(PipelineOrchestrator, "_detect_deepstream", return_value=False)
    @patch.object(PipelineOrchestrator, "_detect_triton", return_value=False)
    def test_select_pipeline_auto_legacy(self, *mocks):
        orch = self._make_orchestrator("auto")
        pipeline = orch.select_pipeline()
        assert pipeline == "legacy"

    # --- Pipeline selection: explicit mode ---

    def test_select_pipeline_explicit_metropolis(self):
        orch = self._make_orchestrator("metropolis")
        pipeline = orch.select_pipeline()
        assert pipeline == "metropolis"

    def test_select_pipeline_explicit_legacy(self):
        orch = self._make_orchestrator("legacy")
        pipeline = orch.select_pipeline()
        assert pipeline == "legacy"

    def test_select_pipeline_explicit_hybrid(self):
        orch = self._make_orchestrator("hybrid")
        pipeline = orch.select_pipeline()
        assert pipeline == "hybrid"

    def test_select_pipeline_invalid_mode_falls_back_to_legacy(self):
        orch = self._make_orchestrator("invalid_mode")
        pipeline = orch.select_pipeline()
        assert pipeline == "legacy"

    # --- Start / Stop ---

    def test_start_selects_pipeline_if_not_selected(self):
        orch = self._make_orchestrator("legacy")
        orch.start()
        assert orch.is_running is True
        assert orch.active_pipeline == "legacy"

    def test_start_raises_if_already_running(self):
        orch = self._make_orchestrator("legacy")
        orch.start()
        with pytest.raises(RuntimeError, match="already running"):
            orch.start()

    def test_stop(self):
        orch = self._make_orchestrator("legacy")
        orch.start()
        orch.stop()
        assert orch.is_running is False

    def test_stop_raises_if_not_running(self):
        orch = self._make_orchestrator("legacy")
        with pytest.raises(RuntimeError, match="No pipeline is currently running"):
            orch.stop()

    # --- Runtime switching ---

    def test_switch_pipeline_while_running(self):
        orch = self._make_orchestrator("legacy")
        orch.start()
        assert orch.active_pipeline == "legacy"

        orch.switch_pipeline("hybrid")
        assert orch.active_pipeline == "hybrid"
        assert orch.is_running is True

    def test_switch_pipeline_while_stopped(self):
        orch = self._make_orchestrator("legacy")
        orch.select_pipeline()
        orch.switch_pipeline("metropolis")
        assert orch.active_pipeline == "metropolis"
        assert orch.is_running is False

    def test_switch_pipeline_invalid_raises(self):
        orch = self._make_orchestrator("legacy")
        with pytest.raises(ValueError, match="Invalid pipeline"):
            orch.switch_pipeline("nonexistent")

    # --- Detections ---

    def test_get_detections_returns_empty_list_for_unknown_camera(self):
        orch = self._make_orchestrator("legacy")
        orch.start()
        assert orch.get_detections(99) == []

    def test_get_detections_raises_if_not_running(self):
        orch = self._make_orchestrator("legacy")
        with pytest.raises(RuntimeError, match="not running"):
            orch.get_detections(1)

    def test_set_and_get_detections(self):
        orch = self._make_orchestrator("legacy")
        orch.start()

        detections = [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(10, 20, 100, 200),
                camera_id=1,
                timestamp=time.time(),
            )
        ]
        orch.set_detections(1, detections)
        result = orch.get_detections(1)
        assert len(result) == 1
        assert result[0].class_name == "person"

    def test_detections_identical_schema_across_pipelines(self):
        """Verify Detection objects have the same schema regardless of pipeline."""
        orch = self._make_orchestrator("legacy")
        orch.start()

        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 20, 100, 200),
            camera_id=1,
            timestamp=1000.0,
        )
        orch.set_detections(1, [det])

        # Switch pipeline - detections interface stays the same
        orch.switch_pipeline("hybrid")
        orch.set_detections(1, [det])
        result = orch.get_detections(1)
        assert result[0].to_dict() == det.to_dict()
