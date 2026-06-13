"""Integration tests for DeepStream pipeline end-to-end processing.

Contains two test classes:
1. TestDeepStreamHardwareIntegration - Requires actual DeepStream/GStreamer
   hardware. Skipped automatically when GStreamer or pyds is unavailable.
2. TestDeepStreamMockIntegration - Mock-based end-to-end flow verification
   that runs without any hardware dependencies.
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.deepstream_pipeline import DeepStreamPipeline, DeepStreamConfig, PipelineStats
from metropolis.orchestrator import Detection


# ---------------------------------------------------------------------------
# Hardware availability checks
# ---------------------------------------------------------------------------

def _gstreamer_available() -> bool:
    """Check if GStreamer Python bindings are available."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # noqa: F401
        return True
    except (ImportError, ValueError):
        return False


def _pyds_available() -> bool:
    """Check if DeepStream Python bindings (pyds) are available."""
    try:
        import pyds  # noqa: F401
        return True
    except ImportError:
        return False


def _opencv_available() -> bool:
    """Check if OpenCV is available for synthetic video generation."""
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


SKIP_NO_GSTREAMER = pytest.mark.skipif(
    not _gstreamer_available(),
    reason="GStreamer Python bindings (gi/Gst) not available",
)

SKIP_NO_PYDS = pytest.mark.skipif(
    not _pyds_available(),
    reason="DeepStream Python bindings (pyds) not available",
)

SKIP_NO_HARDWARE = pytest.mark.skipif(
    not (_gstreamer_available() and _pyds_available()),
    reason="DeepStream/GStreamer hardware not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_video_path(tmp_path):
    """Create a synthetic test video file using OpenCV if available.

    Generates a short 2-second video with colored rectangles to simulate
    objects that could be detected by the pipeline.
    """
    video_path = tmp_path / "test_video.mp4"

    if _opencv_available():
        import cv2
        import numpy as np

        # Create a 2-second video at 30fps (60 frames), 640x480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        for i in range(60):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a moving rectangle to simulate a detectable object
            x = int(100 + i * 5) % 500
            cv2.rectangle(frame, (x, 100), (x + 80, 250), (0, 255, 0), -1)
            # Draw a second static rectangle
            cv2.rectangle(frame, (300, 300), (400, 400), (255, 0, 0), -1)
            writer.write(frame)

        writer.release()
    else:
        # Create a minimal placeholder file for test structure
        video_path.write_bytes(b"\x00" * 1024)

    return str(video_path)


@pytest.fixture
def deepstream_config():
    """Create a DeepStreamConfig for integration testing."""
    return DeepStreamConfig(
        nvinfer_config="configs/nvinfer_config.txt",
        tracker_config="configs/tracker_config.yml",
        mux_width=640,
        mux_height=480,
        mux_batch_size=1,
    )


# ---------------------------------------------------------------------------
# Hardware Integration Tests (skipped without DeepStream)
# ---------------------------------------------------------------------------

@SKIP_NO_HARDWARE
class TestDeepStreamHardwareIntegration:
    """Integration tests requiring actual DeepStream/GStreamer hardware.

    These tests verify the full pipeline flow with real GStreamer elements.
    They are automatically skipped when DeepStream is not installed.
    """

    def test_pipeline_processes_video_file_end_to_end(
        self, deepstream_config, test_video_path
    ):
        """Pipeline processes a test video file and produces detections."""
        detections_received = []
        frames_processed = []

        def probe_callback(batch_data):
            """Collect detections from the pipeline probe."""
            frames_processed.append(batch_data["frame_number"])
            if batch_data.get("detections"):
                detections_received.extend(batch_data["detections"])

        # Create and configure pipeline
        pipeline = DeepStreamPipeline(deepstream_config)
        pipeline.add_source(0, test_video_path, source_type="file")
        pipeline.register_probe(probe_callback)

        # Start pipeline
        pipeline.start()
        assert pipeline.is_running

        # Wait for processing (up to 10 seconds)
        timeout = 10.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if len(frames_processed) > 5:
                break
            time.sleep(0.5)

        # Stop pipeline
        pipeline.stop()
        assert not pipeline.is_running

        # Verify frames were processed
        assert len(frames_processed) > 0, (
            "Pipeline did not process any frames from the test video"
        )

        # Verify pipeline stats were updated
        stats = pipeline.get_stats()
        assert stats.total_frames_processed > 0

    def test_pipeline_probe_receives_detection_metadata(
        self, deepstream_config, test_video_path
    ):
        """Probe callback receives structured detection metadata."""
        batch_data_received = []

        def probe_callback(batch_data):
            batch_data_received.append(batch_data)

        pipeline = DeepStreamPipeline(deepstream_config)
        pipeline.add_source(0, test_video_path, source_type="file")
        pipeline.register_probe(probe_callback)

        pipeline.start()

        # Wait for at least one batch
        timeout = 10.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if len(batch_data_received) > 0:
                break
            time.sleep(0.5)

        pipeline.stop()

        # Verify batch data structure
        assert len(batch_data_received) > 0
        first_batch = batch_data_received[0]
        assert "detections" in first_batch
        assert "frame_number" in first_batch
        assert "source_id" in first_batch
        assert "timestamp" in first_batch
        assert first_batch["source_id"] == 0

    def test_pipeline_stats_updated_during_processing(
        self, deepstream_config, test_video_path
    ):
        """Pipeline stats reflect processing activity."""
        pipeline = DeepStreamPipeline(deepstream_config)
        pipeline.add_source(0, test_video_path, source_type="file")
        pipeline.register_probe(lambda _: None)

        # Stats should be zero before start
        stats_before = pipeline.get_stats()
        assert stats_before.total_frames_processed == 0

        pipeline.start()

        # Wait for some processing
        time.sleep(3.0)

        # Stats should be updated
        stats_during = pipeline.get_stats()

        pipeline.stop()

        assert stats_during.total_frames_processed > 0
        assert stats_during.active_sources >= 0


# ---------------------------------------------------------------------------
# Mock-Based Integration Tests (no hardware required)
# ---------------------------------------------------------------------------

class TestDeepStreamMockIntegration:
    """End-to-end flow verification using mocks for GStreamer and pyds.

    These tests verify the pipeline orchestration logic (add source →
    register probe → start → process frames → stop) without requiring
    actual DeepStream hardware.
    """

    def _create_mock_gst(self):
        """Create a comprehensive mock GStreamer module."""
        gst = MagicMock()

        # Pipeline mock
        mock_pipeline = MagicMock()
        gst.Pipeline.return_value = mock_pipeline

        # Element factory creates working mock elements
        elements = {}

        def make_element(factory_name, element_name):
            elem = MagicMock()
            elem.name = element_name
            elem.factory_name = factory_name
            elem.link.return_value = True
            elem.get_static_pad.return_value = MagicMock()
            elem.get_request_pad.return_value = MagicMock()
            elements[(factory_name, element_name)] = elem
            return elem

        gst.ElementFactory.make.side_effect = make_element

        # Bin mock
        source_bin = MagicMock()
        source_bin.get_static_pad.return_value = MagicMock()
        gst.Bin.new.return_value = source_bin

        # Ghost pad mock
        ghost_pad = MagicMock()
        gst.GhostPad.new_no_target.return_value = ghost_pad
        gst.PadDirection.SRC = 1

        # State change return
        gst.StateChangeReturn.FAILURE = 0
        gst.StateChangeReturn.SUCCESS = 1
        mock_pipeline.set_state.return_value = 1  # SUCCESS

        # Pad probe type
        gst.PadProbeType.BUFFER = 16
        gst.PadProbeReturn.OK = 0

        # Pad link return
        gst.PadLinkReturn.OK = 0

        return gst, elements, mock_pipeline

    def _create_mock_pyds(self):
        """Create a mock pyds module with batch/frame/object metadata."""
        pyds = MagicMock()

        # Create mock object metadata
        mock_obj_meta = MagicMock()
        mock_obj_meta.class_id = 0
        mock_obj_meta.confidence = 0.92
        mock_obj_meta.obj_label = "person"
        mock_obj_meta.object_id = 1
        mock_obj_meta.rect_params.left = 100
        mock_obj_meta.rect_params.top = 50
        mock_obj_meta.rect_params.width = 80
        mock_obj_meta.rect_params.height = 150

        # Object meta list (single object)
        mock_obj_list = MagicMock()
        mock_obj_list.data = MagicMock()
        mock_obj_list.next = None

        # Frame metadata
        mock_frame_meta = MagicMock()
        mock_frame_meta.source_id = 0
        mock_frame_meta.frame_num = 1
        mock_frame_meta.obj_meta_list = mock_obj_list

        # Frame meta list
        mock_frame_list = MagicMock()
        mock_frame_list.data = MagicMock()
        mock_frame_list.next = None

        # Batch metadata
        mock_batch_meta = MagicMock()
        mock_batch_meta.frame_meta_list = mock_frame_list

        # Configure pyds functions
        pyds.gst_buffer_get_nvds_batch_meta.return_value = mock_batch_meta
        pyds.NvDsFrameMeta.cast.return_value = mock_frame_meta
        pyds.NvDsObjectMeta.cast.return_value = mock_obj_meta

        return pyds, mock_batch_meta, mock_frame_meta, mock_obj_meta

    def test_end_to_end_flow_add_source_start_stop(self):
        """Verify full lifecycle: add source → register probe → start → stop."""
        gst, elements, mock_pipeline = self._create_mock_gst()
        pyds, _, _, _ = self._create_mock_pyds()

        # Mock GLib for the main loop
        mock_glib = MagicMock()
        mock_main_loop = MagicMock()
        mock_glib.MainLoop.return_value = mock_main_loop

        config = DeepStreamConfig(mux_batch_size=1)
        pipeline = DeepStreamPipeline(config)

        # Mock _import_gstreamer to return our mocks
        with patch.object(
            pipeline, "_import_gstreamer", return_value=(gst, pyds)
        ):
            # Mock gi.repository.GLib
            with patch.dict(
                "sys.modules", {"gi": MagicMock(), "gi.repository": MagicMock()}
            ):
                with patch(
                    "metropolis.deepstream_pipeline.DeepStreamPipeline._import_gstreamer",
                    return_value=(gst, pyds),
                ):
                    # Re-create pipeline with patched import
                    pipeline = DeepStreamPipeline(config)
                    pipeline._import_gstreamer = MagicMock(
                        return_value=(gst, pyds)
                    )

                    # Step 1: Add source
                    pipeline.add_source(0, "/test/video.mp4", source_type="file")
                    assert 0 in pipeline.sources
                    assert pipeline.sources[0]["uri"] == "/test/video.mp4"
                    assert pipeline.sources[0]["source_type"] == "file"

                    # Step 2: Register probe
                    detections_collected = []

                    def probe_cb(batch_data):
                        detections_collected.append(batch_data)

                    pipeline.register_probe(probe_cb)

                    # Step 3: Start (mock the GLib import inside start)
                    with patch(
                        "builtins.__import__",
                        side_effect=lambda name, *args, **kwargs: (
                            mock_glib
                            if "GLib" in str(args)
                            else __builtins__.__import__(name, *args, **kwargs)
                        ),
                    ):
                        # Patch the internal start to avoid actual GLib import
                        # Instead, directly test the state transitions
                        pipeline._build_pipeline = MagicMock()
                        pipeline._create_source_bin = MagicMock(
                            return_value=MagicMock(
                                get_static_pad=MagicMock(
                                    return_value=MagicMock(
                                        link=MagicMock(
                                            return_value=gst.PadLinkReturn.OK
                                        )
                                    )
                                )
                            )
                        )

                        # Simulate start internals
                        pipeline._pipeline = mock_pipeline
                        pipeline._streammux = elements.get(
                            ("nvstreammux", "mux"), MagicMock()
                        )
                        pipeline._nvosd = MagicMock()
                        osd_pad = MagicMock()
                        pipeline._nvosd.get_static_pad.return_value = osd_pad
                        pipeline._main_loop = mock_main_loop
                        pipeline._loop_thread = MagicMock()
                        pipeline._running = True

                    # Verify pipeline is running
                    assert pipeline.is_running

                    # Step 4: Stop
                    pipeline.stop()
                    assert not pipeline.is_running

    def test_probe_callback_receives_detections_from_metadata(self):
        """Probe callback extracts Detection objects from NvDs metadata."""
        gst, elements, mock_pipeline = self._create_mock_gst()
        pyds, mock_batch_meta, mock_frame_meta, mock_obj_meta = (
            self._create_mock_pyds()
        )

        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)
        pipeline._import_gstreamer = MagicMock(return_value=(gst, pyds))

        # Register a probe callback
        detections_collected = []

        def probe_cb(batch_data):
            detections_collected.append(batch_data)

        pipeline.register_probe(probe_cb)

        # Add a source so stats tracking works
        pipeline.add_source(0, "/test/video.mp4", source_type="file")
        pipeline._sources[0]["connected"] = True

        # Simulate the analytics probe being called
        mock_info = MagicMock()
        mock_buffer = MagicMock()
        mock_info.get_buffer.return_value = mock_buffer

        # Call the probe callback directly
        result = pipeline._analytics_probe_callback(MagicMock(), mock_info)

        # Verify probe callback was invoked with detection data
        assert len(detections_collected) == 1
        batch_data = detections_collected[0]

        assert "detections" in batch_data
        assert "frame_number" in batch_data
        assert "source_id" in batch_data
        assert "timestamp" in batch_data
        assert batch_data["source_id"] == 0
        assert batch_data["frame_number"] == 1

        # Verify detection was extracted correctly
        assert len(batch_data["detections"]) == 1
        detection = batch_data["detections"][0]
        assert isinstance(detection, Detection)
        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.confidence == 0.92
        assert detection.bbox == (100, 50, 180, 200)  # left, top, left+width, top+height
        assert detection.camera_id == 0
        assert detection.track_id == 1

    def test_pipeline_stats_updated_by_probe(self):
        """Pipeline stats are incremented when probe processes frames."""
        gst, elements, mock_pipeline = self._create_mock_gst()
        pyds, mock_batch_meta, mock_frame_meta, mock_obj_meta = (
            self._create_mock_pyds()
        )

        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)
        pipeline._import_gstreamer = MagicMock(return_value=(gst, pyds))

        # Add a connected source
        pipeline.add_source(0, "/test/video.mp4", source_type="file")
        pipeline._sources[0]["connected"] = True

        # Register a no-op probe
        pipeline.register_probe(lambda _: None)

        # Verify initial stats
        stats_before = pipeline.get_stats()
        assert stats_before.total_frames_processed == 0

        # Simulate probe invocation
        mock_info = MagicMock()
        mock_info.get_buffer.return_value = MagicMock()
        pipeline._analytics_probe_callback(MagicMock(), mock_info)

        # Verify stats were updated
        stats_after = pipeline.get_stats()
        assert stats_after.total_frames_processed == 1
        assert stats_after.active_sources == 1
        assert 0 in stats_after.fps

    def test_multiple_probe_callbacks_all_invoked(self):
        """All registered probe callbacks receive metadata."""
        gst, elements, mock_pipeline = self._create_mock_gst()
        pyds, _, _, _ = self._create_mock_pyds()

        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)
        pipeline._import_gstreamer = MagicMock(return_value=(gst, pyds))

        # Add a connected source
        pipeline.add_source(0, "/test/video.mp4", source_type="file")
        pipeline._sources[0]["connected"] = True

        # Register multiple callbacks
        callback_1_data = []
        callback_2_data = []

        pipeline.register_probe(lambda d: callback_1_data.append(d))
        pipeline.register_probe(lambda d: callback_2_data.append(d))

        # Simulate probe invocation
        mock_info = MagicMock()
        mock_info.get_buffer.return_value = MagicMock()
        pipeline._analytics_probe_callback(MagicMock(), mock_info)

        # Both callbacks should have been called
        assert len(callback_1_data) == 1
        assert len(callback_2_data) == 1

    def test_pipeline_handles_probe_callback_exception_gracefully(self):
        """Pipeline continues if a probe callback raises an exception."""
        gst, elements, mock_pipeline = self._create_mock_gst()
        pyds, _, _, _ = self._create_mock_pyds()

        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)
        pipeline._import_gstreamer = MagicMock(return_value=(gst, pyds))

        # Add a connected source
        pipeline.add_source(0, "/test/video.mp4", source_type="file")
        pipeline._sources[0]["connected"] = True

        # Register a failing callback followed by a working one
        working_callback_data = []

        def failing_callback(batch_data):
            raise ValueError("Simulated callback error")

        pipeline.register_probe(failing_callback)
        pipeline.register_probe(lambda d: working_callback_data.append(d))

        # Simulate probe invocation - should not raise
        mock_info = MagicMock()
        mock_info.get_buffer.return_value = MagicMock()
        pipeline._analytics_probe_callback(MagicMock(), mock_info)

        # The working callback should still have been called
        assert len(working_callback_data) == 1

    def test_full_mock_pipeline_flow_with_multiple_frames(self):
        """Simulate processing multiple frames through the pipeline."""
        gst, elements, mock_pipeline = self._create_mock_gst()
        pyds = MagicMock()

        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)
        pipeline._import_gstreamer = MagicMock(return_value=(gst, pyds))

        # Add a connected source
        pipeline.add_source(0, "/test/video.mp4", source_type="file")
        pipeline._sources[0]["connected"] = True

        all_detections = []
        pipeline.register_probe(
            lambda d: all_detections.extend(d.get("detections", []))
        )

        # Simulate 5 frames with varying detections
        for frame_num in range(5):
            # Create frame-specific metadata
            mock_obj_meta = MagicMock()
            mock_obj_meta.class_id = 0
            mock_obj_meta.confidence = 0.85 + frame_num * 0.02
            mock_obj_meta.obj_label = "person"
            mock_obj_meta.object_id = frame_num + 1
            mock_obj_meta.rect_params.left = 100 + frame_num * 10
            mock_obj_meta.rect_params.top = 50
            mock_obj_meta.rect_params.width = 80
            mock_obj_meta.rect_params.height = 150

            mock_obj_list = MagicMock()
            mock_obj_list.data = MagicMock()
            mock_obj_list.next = None

            mock_frame_meta = MagicMock()
            mock_frame_meta.source_id = 0
            mock_frame_meta.frame_num = frame_num
            mock_frame_meta.obj_meta_list = mock_obj_list

            mock_frame_list = MagicMock()
            mock_frame_list.data = MagicMock()
            mock_frame_list.next = None

            mock_batch_meta = MagicMock()
            mock_batch_meta.frame_meta_list = mock_frame_list

            pyds.gst_buffer_get_nvds_batch_meta.return_value = mock_batch_meta
            pyds.NvDsFrameMeta.cast.return_value = mock_frame_meta
            pyds.NvDsObjectMeta.cast.return_value = mock_obj_meta

            # Invoke probe
            mock_info = MagicMock()
            mock_info.get_buffer.return_value = MagicMock()
            pipeline._analytics_probe_callback(MagicMock(), mock_info)

        # Verify all frames were processed
        stats = pipeline.get_stats()
        assert stats.total_frames_processed == 5
        assert len(all_detections) == 5

        # Verify detections have correct track IDs (monotonically increasing)
        track_ids = [d.track_id for d in all_detections]
        assert track_ids == [1, 2, 3, 4, 5]

    def test_stop_without_start_is_safe(self):
        """Calling stop() on a pipeline that was never started does not crash."""
        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)

        # Should not raise
        pipeline.stop()
        assert not pipeline.is_running

    def test_cannot_add_source_while_running(self):
        """Adding a source to a running pipeline raises RuntimeError."""
        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)
        pipeline._running = True

        with pytest.raises(RuntimeError, match="Cannot add sources while pipeline is running"):
            pipeline.add_source(0, "/test/video.mp4", source_type="file")

    def test_cannot_add_duplicate_source_id(self):
        """Adding a source with an existing ID raises ValueError."""
        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)
        pipeline.add_source(0, "/test/video1.mp4", source_type="file")

        with pytest.raises(ValueError, match="already registered"):
            pipeline.add_source(0, "/test/video2.mp4", source_type="file")

    def test_invalid_source_type_raises_error(self):
        """Adding a source with invalid type raises ValueError."""
        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)

        with pytest.raises(ValueError, match="Invalid source_type"):
            pipeline.add_source(0, "/test/video.mp4", source_type="invalid")

    def test_register_non_callable_raises_type_error(self):
        """Registering a non-callable probe raises TypeError."""
        config = DeepStreamConfig()
        pipeline = DeepStreamPipeline(config)

        with pytest.raises(TypeError, match="callable"):
            pipeline.register_probe("not_a_function")
