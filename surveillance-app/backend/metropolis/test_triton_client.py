"""Unit tests for TritonClient.infer() method."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from .orchestrator import Detection


class TestTritonClientInfer:
    """Tests for the TritonClient.infer() method."""

    def _make_client_disconnected(self):
        """Create a TritonClient instance that is not connected."""
        with patch.dict("sys.modules", {"tritonclient": None, "tritonclient.grpc": None}):
            from .triton_client import TritonClient

            client = TritonClient.__new__(TritonClient)
            client.server_url = "localhost:8001"
            client.model_name = "yolov8_ensemble"
            client._connected = False
            client._model_ready = False
            client._client = None
            client._grpcclient = None
            client._fallback_enabled = False
            client._using_fallback = False
            client._fallback_engine_path = None
            return client

    def _make_client_connected(self, mock_grpc_client=None, mock_grpcclient_module=None):
        """Create a TritonClient instance that appears connected with mocks."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = True
        client._client = mock_grpc_client or MagicMock()
        client._grpcclient = mock_grpcclient_module or MagicMock()
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def test_infer_raises_runtime_error_when_not_connected(self):
        """infer() should raise RuntimeError if client is not connected."""
        client = self._make_client_disconnected()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]

        with pytest.raises(RuntimeError, match="not connected"):
            client.infer(frames)

    def test_infer_single_frame_returns_detections(self):
        """infer() should return Detection objects for a single frame."""
        mock_grpc_module = MagicMock()
        mock_infer_input = MagicMock()
        mock_grpc_module.InferInput.return_value = mock_infer_input

        mock_grpc_client = MagicMock()
        # Simulate Triton returning 2 detections: [x1, y1, x2, y2, confidence, class_id]
        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.array([
            [10.0, 20.0, 100.0, 200.0, 0.95, 0.0],
            [50.0, 60.0, 150.0, 160.0, 0.88, 1.0],
        ], dtype=np.float32)
        mock_grpc_client.infer.return_value = mock_result

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = client.infer([frame])

        assert len(detections) == 2
        assert isinstance(detections[0], Detection)
        assert detections[0].class_id == 0
        assert detections[0].confidence == pytest.approx(0.95)
        assert detections[0].bbox == (10, 20, 100, 200)
        assert detections[0].camera_id == 0
        assert detections[1].class_id == 1
        assert detections[1].confidence == pytest.approx(0.88)
        assert detections[1].bbox == (50, 60, 150, 160)

    def test_infer_multiple_frames(self):
        """infer() should process multiple frames and return all detections."""
        mock_grpc_module = MagicMock()
        mock_grpc_module.InferInput.return_value = MagicMock()

        mock_grpc_client = MagicMock()
        # First frame: 1 detection, second frame: 2 detections
        result1 = MagicMock()
        result1.as_numpy.return_value = np.array([
            [10.0, 20.0, 100.0, 200.0, 0.9, 0.0],
        ], dtype=np.float32)
        result2 = MagicMock()
        result2.as_numpy.return_value = np.array([
            [30.0, 40.0, 130.0, 240.0, 0.85, 1.0],
            [60.0, 70.0, 160.0, 270.0, 0.75, 2.0],
        ], dtype=np.float32)
        mock_grpc_client.infer.side_effect = [result1, result2]

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        detections = client.infer(frames)

        assert len(detections) == 3
        assert detections[0].class_id == 0
        assert detections[1].class_id == 1
        assert detections[2].class_id == 2

    def test_infer_empty_output(self):
        """infer() should return empty list when model returns no detections."""
        mock_grpc_module = MagicMock()
        mock_grpc_module.InferInput.return_value = MagicMock()

        mock_grpc_client = MagicMock()
        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.array([], dtype=np.float32)
        mock_grpc_client.infer.return_value = mock_result

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = client.infer([frame])

        assert detections == []

    def test_infer_creates_correct_input_tensor(self):
        """infer() should create InferInput with correct shape and dtype."""
        mock_grpc_module = MagicMock()
        mock_infer_input = MagicMock()
        mock_grpc_module.InferInput.return_value = mock_infer_input

        mock_grpc_client = MagicMock()
        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.array([], dtype=np.float32)
        mock_grpc_client.infer.return_value = mock_result

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        client.infer([frame])

        # Verify InferInput was created with correct parameters
        mock_grpc_module.InferInput.assert_called_once_with(
            "raw_image", [480, 640, 3], "UINT8"
        )
        mock_infer_input.set_data_from_numpy.assert_called_once()

    def test_infer_converts_non_uint8_frame(self):
        """infer() should convert non-uint8 frames to uint8."""
        mock_grpc_module = MagicMock()
        mock_infer_input = MagicMock()
        mock_grpc_module.InferInput.return_value = mock_infer_input

        mock_grpc_client = MagicMock()
        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.array([], dtype=np.float32)
        mock_grpc_client.infer.return_value = mock_result

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        # Pass a float32 frame
        frame = np.zeros((480, 640, 3), dtype=np.float32)
        client.infer([frame])

        # Should still create input with UINT8 type
        mock_grpc_module.InferInput.assert_called_once_with(
            "raw_image", [480, 640, 3], "UINT8"
        )

    def test_infer_grpc_error_raises_runtime_error(self):
        """infer() should raise RuntimeError when gRPC call fails."""
        mock_grpc_module = MagicMock()
        mock_grpc_module.InferInput.return_value = MagicMock()

        mock_grpc_client = MagicMock()
        mock_grpc_client.infer.side_effect = Exception("gRPC connection refused")

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Triton inference failed"):
            client.infer([frame])

    def test_infer_detection_has_timestamp(self):
        """infer() should set timestamp on Detection objects using time.time()."""
        mock_grpc_module = MagicMock()
        mock_grpc_module.InferInput.return_value = MagicMock()

        mock_grpc_client = MagicMock()
        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.array([
            [10.0, 20.0, 100.0, 200.0, 0.9, 0.0],
        ], dtype=np.float32)
        mock_grpc_client.infer.return_value = mock_result

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        before = time.time()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = client.infer([frame])
        after = time.time()

        assert len(detections) == 1
        assert before <= detections[0].timestamp <= after

    def test_infer_single_detection_1d_output(self):
        """infer() should handle 1D output array (single detection)."""
        mock_grpc_module = MagicMock()
        mock_grpc_module.InferInput.return_value = MagicMock()

        mock_grpc_client = MagicMock()
        mock_result = MagicMock()
        # Some models might return a 1D array for a single detection
        mock_result.as_numpy.return_value = np.array(
            [10.0, 20.0, 100.0, 200.0, 0.9, 0.0], dtype=np.float32
        )
        mock_grpc_client.infer.return_value = mock_result

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = client.infer([frame])

        assert len(detections) == 1
        assert detections[0].bbox == (10, 20, 100, 200)

    def test_infer_empty_frames_list(self):
        """infer() should return empty list when given no frames."""
        mock_grpc_module = MagicMock()
        mock_grpc_client = MagicMock()

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        detections = client.infer([])
        assert detections == []


class TestTritonClientHealthCheck:
    """Tests for the TritonClient.health_check() method."""

    def _make_client_connected(self, mock_grpc_client=None):
        """Create a TritonClient instance that appears connected with mocks."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = False
        client._client = mock_grpc_client or MagicMock()
        client._grpcclient = MagicMock()
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def _make_client_no_client(self):
        """Create a TritonClient instance with no underlying client."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = False
        client._model_ready = False
        client._client = None
        client._grpcclient = None
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def test_health_check_returns_true_when_server_live(self):
        """health_check() should return True when server is live."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        result = client.health_check()

        assert result is True
        assert client._connected is True

    def test_health_check_returns_false_when_server_not_live(self):
        """health_check() should return False when server is not live."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = False

        client = self._make_client_connected(mock_grpc_client)
        result = client.health_check()

        assert result is False
        assert client._connected is False

    def test_health_check_returns_false_on_exception(self):
        """health_check() should return False and not crash on exception."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.side_effect = Exception("Connection refused")

        client = self._make_client_connected(mock_grpc_client)
        result = client.health_check()

        assert result is False
        assert client._connected is False

    def test_health_check_returns_false_when_client_is_none(self):
        """health_check() should return False when _client is None."""
        client = self._make_client_no_client()
        result = client.health_check()

        assert result is False
        assert client._connected is False


class TestTritonClientIsModelReady:
    """Tests for the TritonClient.is_model_ready() method."""

    def _make_client_connected(self, mock_grpc_client=None):
        """Create a TritonClient instance that appears connected with mocks."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = False
        client._client = mock_grpc_client or MagicMock()
        client._grpcclient = MagicMock()
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def _make_client_no_client(self):
        """Create a TritonClient instance with no underlying client."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = False
        client._model_ready = False
        client._client = None
        client._grpcclient = None
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def test_is_model_ready_returns_true_when_model_ready(self):
        """is_model_ready() should return True when model is ready."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_model_ready.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        result = client.is_model_ready()

        assert result is True
        assert client._model_ready is True
        mock_grpc_client.is_model_ready.assert_called_once_with(
            model_name="yolov8_ensemble"
        )

    def test_is_model_ready_returns_false_when_model_not_ready(self):
        """is_model_ready() should return False when model is not ready."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_model_ready.return_value = False

        client = self._make_client_connected(mock_grpc_client)
        result = client.is_model_ready()

        assert result is False
        assert client._model_ready is False

    def test_is_model_ready_returns_false_on_exception(self):
        """is_model_ready() should return False and not crash on exception."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_model_ready.side_effect = Exception("Model not found")

        client = self._make_client_connected(mock_grpc_client)
        result = client.is_model_ready()

        assert result is False
        assert client._model_ready is False

    def test_is_model_ready_returns_false_when_client_is_none(self):
        """is_model_ready() should return False when _client is None."""
        client = self._make_client_no_client()
        result = client.is_model_ready()

        assert result is False
        assert client._model_ready is False


class TestTritonClientHealthPolling:
    """Tests for the background health polling thread."""

    def _make_client_connected(self, mock_grpc_client=None):
        """Create a TritonClient instance that appears connected with mocks."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = False
        client._client = mock_grpc_client or MagicMock()
        client._grpcclient = MagicMock()
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def test_start_health_polling_creates_daemon_thread(self):
        """start_health_polling() should create a daemon thread."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True
        mock_grpc_client.is_model_ready.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        client.start_health_polling(interval=0.1)

        try:
            assert hasattr(client, "_polling_thread")
            assert client._polling_thread.daemon is True
            assert client._polling_thread.is_alive()
        finally:
            client.stop_health_polling()

    def test_stop_health_polling_stops_thread(self):
        """stop_health_polling() should stop the polling thread."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True
        mock_grpc_client.is_model_ready.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        client.start_health_polling(interval=0.1)
        client.stop_health_polling()

        assert not client._polling_thread.is_alive()

    def test_polling_calls_health_check_and_is_model_ready(self):
        """Polling thread should call health_check and is_model_ready."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True
        mock_grpc_client.is_model_ready.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        client.start_health_polling(interval=0.1)

        # Wait for at least one poll cycle
        time.sleep(0.3)
        client.stop_health_polling()

        assert mock_grpc_client.is_server_live.call_count >= 1
        assert mock_grpc_client.is_model_ready.call_count >= 1

    def test_polling_updates_state_on_failure(self):
        """Polling thread should update state when server becomes unavailable."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = False

        client = self._make_client_connected(mock_grpc_client)
        client.start_health_polling(interval=0.1)

        # Wait for at least one poll cycle
        time.sleep(0.3)
        client.stop_health_polling()

        assert client._connected is False

    def test_stop_health_polling_safe_when_not_started(self):
        """stop_health_polling() should not crash if polling was never started."""
        mock_grpc_client = MagicMock()
        client = self._make_client_connected(mock_grpc_client)

        # Should not raise
        client.stop_health_polling()

    def test_start_health_polling_ignores_duplicate_start(self):
        """start_health_polling() should not create duplicate threads."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True
        mock_grpc_client.is_model_ready.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        client.start_health_polling(interval=0.1)
        first_thread = client._polling_thread

        client.start_health_polling(interval=0.1)
        second_thread = client._polling_thread

        try:
            # Should be the same thread (duplicate start ignored)
            assert first_thread is second_thread
        finally:
            client.stop_health_polling()


class TestTritonClientClose:
    """Tests for the TritonClient.close() method."""

    def _make_client_connected(self, mock_grpc_client=None):
        """Create a TritonClient instance that appears connected with mocks."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = True
        client._client = mock_grpc_client or MagicMock()
        client._grpcclient = MagicMock()
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def test_close_stops_polling_and_cleans_up(self):
        """close() should stop polling and reset state."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True
        mock_grpc_client.is_model_ready.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        client.start_health_polling(interval=0.1)
        time.sleep(0.2)

        client.close()

        assert client._connected is False
        assert client._model_ready is False
        assert client._client is None
        assert not client._polling_thread.is_alive()

    def test_close_safe_without_polling(self):
        """close() should work even if polling was never started."""
        mock_grpc_client = MagicMock()
        client = self._make_client_connected(mock_grpc_client)

        client.close()

        assert client._connected is False
        assert client._model_ready is False
        assert client._client is None


class TestTritonClientFallback:
    """Tests for the fallback logic in TritonClient."""

    def _make_client_connected(self, mock_grpc_client=None, mock_grpcclient_module=None):
        """Create a TritonClient instance that appears connected with mocks."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = True
        client._client = mock_grpc_client or MagicMock()
        client._grpcclient = mock_grpcclient_module or MagicMock()
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def _make_client_disconnected(self):
        """Create a TritonClient instance that is not connected."""
        from .triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = False
        client._model_ready = False
        client._client = None
        client._grpcclient = None
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None
        return client

    def test_is_using_fallback_initially_false(self):
        """is_using_fallback should be False initially."""
        client = self._make_client_connected()
        assert client.is_using_fallback is False

    def test_set_fallback_engine_raises_on_missing_file(self):
        """set_fallback_engine() should raise FileNotFoundError for missing file."""
        client = self._make_client_connected()
        with pytest.raises(FileNotFoundError, match="not found"):
            client.set_fallback_engine("/nonexistent/path/model.engine")

    def test_set_fallback_engine_enables_fallback(self, tmp_path):
        """set_fallback_engine() should enable fallback when file exists."""
        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"fake engine data")

        client = self._make_client_connected()
        client.set_fallback_engine(str(engine_file))

        assert client._fallback_enabled is True
        assert client._fallback_engine_path == str(engine_file)

    def test_infer_falls_back_on_grpc_error_when_enabled(self, tmp_path):
        """infer() should switch to fallback on gRPC error when fallback is enabled."""
        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"fake engine data")

        mock_grpc_module = MagicMock()
        mock_grpc_module.InferInput.return_value = MagicMock()

        mock_grpc_client = MagicMock()
        mock_grpc_client.infer.side_effect = Exception("gRPC connection refused")

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)
        client._fallback_enabled = True
        client._fallback_engine_path = str(engine_file)

        # Mock the _infer_local_tensorrt method to avoid needing real TensorRT
        with patch.object(client, "_infer_local_tensorrt", return_value=[]) as mock_local:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = client.infer([frame])

            assert client._using_fallback is True
            assert client.is_using_fallback is True
            mock_local.assert_called_once_with([frame])
            assert result == []

    def test_infer_raises_when_no_fallback_configured(self):
        """infer() should raise RuntimeError on gRPC error when no fallback."""
        mock_grpc_module = MagicMock()
        mock_grpc_module.InferInput.return_value = MagicMock()

        mock_grpc_client = MagicMock()
        mock_grpc_client.infer.side_effect = Exception("gRPC connection refused")

        client = self._make_client_connected(mock_grpc_client, mock_grpc_module)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Triton inference failed"):
            client.infer([frame])

    def test_infer_uses_fallback_directly_when_already_in_fallback(self, tmp_path):
        """infer() should go directly to local inference when already in fallback."""
        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"fake engine data")

        client = self._make_client_connected()
        client._fallback_enabled = True
        client._using_fallback = True
        client._fallback_engine_path = str(engine_file)

        with patch.object(client, "_infer_local_tensorrt", return_value=[]) as mock_local:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = client.infer([frame])

            mock_local.assert_called_once_with([frame])
            assert result == []

    def test_infer_disconnected_with_fallback_switches_to_local(self, tmp_path):
        """infer() should switch to fallback when disconnected and fallback enabled."""
        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"fake engine data")

        client = self._make_client_disconnected()
        client._fallback_enabled = True
        client._fallback_engine_path = str(engine_file)

        with patch.object(client, "_infer_local_tensorrt", return_value=[]) as mock_local:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = client.infer([frame])

            assert client._using_fallback is True
            mock_local.assert_called_once_with([frame])

    def test_infer_local_tensorrt_raises_when_not_configured(self):
        """_infer_local_tensorrt() should raise RuntimeError when no engine configured."""
        client = self._make_client_connected()

        with pytest.raises(RuntimeError, match="No fallback TensorRT engine configured"):
            client._infer_local_tensorrt([np.zeros((480, 640, 3), dtype=np.uint8)])

    def test_polling_auto_recovers_from_fallback(self):
        """Health polling should set _using_fallback=False when Triton comes back."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True
        mock_grpc_client.is_model_ready.return_value = True

        client = self._make_client_connected(mock_grpc_client)
        client._using_fallback = True
        client._fallback_enabled = True

        client.start_health_polling(interval=0.1)

        # Wait for at least one poll cycle
        time.sleep(0.3)
        client.stop_health_polling()

        assert client._using_fallback is False
        assert client.is_using_fallback is False

    def test_polling_does_not_recover_when_server_down(self):
        """Health polling should keep fallback active when server is still down."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = False

        client = self._make_client_connected(mock_grpc_client)
        client._using_fallback = True
        client._fallback_enabled = True

        client.start_health_polling(interval=0.1)

        # Wait for at least one poll cycle
        time.sleep(0.3)
        client.stop_health_polling()

        assert client._using_fallback is True
