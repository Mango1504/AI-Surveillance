"""Additional unit tests for TritonClient covering gaps in package-level tests.

Tests cover:
- get_model_metadata() success and error cases
- _parse_detections() with malformed rows (less than 6 values)
- __init__ method with mocked tritonclient import (success and ImportError)
- close() resets all state properly (comprehensive state verification)

All tests mock tritonclient.grpc so they run without a Triton server.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.orchestrator import Detection


class TestTritonClientGetModelMetadata:
    """Tests for the TritonClient.get_model_metadata() method."""

    def _make_client_connected(self, mock_grpc_client=None):
        """Create a TritonClient instance that appears connected with mocks."""
        from metropolis.triton_client import TritonClient

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

    def _make_client_disconnected(self):
        """Create a TritonClient instance that is not connected."""
        from metropolis.triton_client import TritonClient

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

    def test_get_model_metadata_success(self):
        """get_model_metadata() should return structured metadata dict on success."""
        mock_grpc_client = MagicMock()

        # Create mock metadata response
        mock_input = MagicMock()
        mock_input.name = "raw_image"
        mock_input.shape = [-1, -1, 3]
        mock_input.datatype = "UINT8"

        mock_output = MagicMock()
        mock_output.name = "detections"
        mock_output.shape = [-1, 6]
        mock_output.datatype = "FP32"

        mock_metadata = MagicMock()
        mock_metadata.name = "yolov8_ensemble"
        mock_metadata.versions = ["1"]
        mock_metadata.inputs = [mock_input]
        mock_metadata.outputs = [mock_output]

        mock_grpc_client.get_model_metadata.return_value = mock_metadata

        client = self._make_client_connected(mock_grpc_client)
        result = client.get_model_metadata()

        assert result["name"] == "yolov8_ensemble"
        assert result["versions"] == ["1"]
        assert len(result["inputs"]) == 1
        assert result["inputs"][0]["name"] == "raw_image"
        assert result["inputs"][0]["shape"] == [-1, -1, 3]
        assert result["inputs"][0]["datatype"] == "UINT8"
        assert len(result["outputs"]) == 1
        assert result["outputs"][0]["name"] == "detections"
        assert result["outputs"][0]["shape"] == [-1, 6]
        assert result["outputs"][0]["datatype"] == "FP32"

        mock_grpc_client.get_model_metadata.assert_called_once_with(
            model_name="yolov8_ensemble"
        )

    def test_get_model_metadata_raises_when_not_connected(self):
        """get_model_metadata() should raise RuntimeError when client is not connected."""
        client = self._make_client_disconnected()

        with pytest.raises(RuntimeError, match="not connected"):
            client.get_model_metadata()

    def test_get_model_metadata_raises_on_grpc_error(self):
        """get_model_metadata() should raise RuntimeError when gRPC call fails."""
        mock_grpc_client = MagicMock()
        mock_grpc_client.get_model_metadata.side_effect = Exception(
            "Model not found in repository"
        )

        client = self._make_client_connected(mock_grpc_client)

        with pytest.raises(RuntimeError, match="Failed to retrieve model metadata"):
            client.get_model_metadata()

    def test_get_model_metadata_multiple_inputs_outputs(self):
        """get_model_metadata() should handle models with multiple inputs/outputs."""
        mock_grpc_client = MagicMock()

        mock_input1 = MagicMock()
        mock_input1.name = "raw_image"
        mock_input1.shape = [-1, -1, 3]
        mock_input1.datatype = "UINT8"

        mock_input2 = MagicMock()
        mock_input2.name = "image_info"
        mock_input2.shape = [3]
        mock_input2.datatype = "FP32"

        mock_output1 = MagicMock()
        mock_output1.name = "detections"
        mock_output1.shape = [-1, 6]
        mock_output1.datatype = "FP32"

        mock_output2 = MagicMock()
        mock_output2.name = "num_detections"
        mock_output2.shape = [1]
        mock_output2.datatype = "INT32"

        mock_metadata = MagicMock()
        mock_metadata.name = "yolov8_ensemble"
        mock_metadata.versions = ["1", "2"]
        mock_metadata.inputs = [mock_input1, mock_input2]
        mock_metadata.outputs = [mock_output1, mock_output2]

        mock_grpc_client.get_model_metadata.return_value = mock_metadata

        client = self._make_client_connected(mock_grpc_client)
        result = client.get_model_metadata()

        assert result["versions"] == ["1", "2"]
        assert len(result["inputs"]) == 2
        assert len(result["outputs"]) == 2
        assert result["inputs"][1]["name"] == "image_info"
        assert result["outputs"][1]["name"] == "num_detections"


class TestTritonClientParseDetections:
    """Tests for the TritonClient._parse_detections() method with malformed data."""

    def _make_client(self):
        """Create a TritonClient instance for testing _parse_detections."""
        from metropolis.triton_client import TritonClient

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

    def test_parse_detections_skips_malformed_rows_less_than_6_values(self):
        """_parse_detections() should skip rows with fewer than 6 values."""
        client = self._make_client()

        # Mix of valid and malformed rows
        output = np.array([
            [10.0, 20.0, 100.0, 200.0, 0.95, 0.0],  # Valid
            [30.0, 40.0, 50.0],                        # Malformed: only 3 values
            [60.0, 70.0, 160.0, 270.0, 0.8, 1.0],    # Valid
        ], dtype=object)

        # numpy won't create a proper 2D array with ragged rows, so test with
        # a proper 2D array where we simulate short rows
        # Instead, test with a 2D array that has rows with less than 6 columns
        output_short = np.array([
            [10.0, 20.0, 100.0, 200.0, 0.95],  # Only 5 values
        ], dtype=np.float32)

        detections = client._parse_detections(output_short)
        assert len(detections) == 0  # Should skip the malformed row

    def test_parse_detections_handles_empty_array(self):
        """_parse_detections() should return empty list for empty array."""
        client = self._make_client()

        output = np.array([], dtype=np.float32)
        detections = client._parse_detections(output)
        assert detections == []

    def test_parse_detections_handles_none_input(self):
        """_parse_detections() should return empty list for None input."""
        client = self._make_client()

        detections = client._parse_detections(None)
        assert detections == []

    def test_parse_detections_valid_2d_array(self):
        """_parse_detections() should correctly parse a valid 2D detection array."""
        client = self._make_client()

        output = np.array([
            [10.0, 20.0, 100.0, 200.0, 0.95, 0.0],
            [50.0, 60.0, 150.0, 160.0, 0.88, 1.0],
        ], dtype=np.float32)

        detections = client._parse_detections(output, camera_id=2)

        assert len(detections) == 2
        assert isinstance(detections[0], Detection)
        assert detections[0].bbox == (10, 20, 100, 200)
        assert detections[0].confidence == pytest.approx(0.95)
        assert detections[0].class_id == 0
        assert detections[0].camera_id == 2
        assert detections[1].class_id == 1

    def test_parse_detections_1d_single_detection(self):
        """_parse_detections() should handle 1D array (single detection)."""
        client = self._make_client()

        output = np.array([10.0, 20.0, 100.0, 200.0, 0.9, 2.0], dtype=np.float32)
        detections = client._parse_detections(output)

        assert len(detections) == 1
        assert detections[0].class_id == 2
        assert detections[0].bbox == (10, 20, 100, 200)

    def test_parse_detections_rows_with_extra_values_uses_first_six(self):
        """_parse_detections() should use only first 6 values if row has more."""
        client = self._make_client()

        # Row with 8 values (extra data beyond the 6 expected)
        output = np.array([
            [10.0, 20.0, 100.0, 200.0, 0.95, 0.0, 99.0, 88.0],
        ], dtype=np.float32)

        detections = client._parse_detections(output)

        assert len(detections) == 1
        assert detections[0].bbox == (10, 20, 100, 200)
        assert detections[0].confidence == pytest.approx(0.95)
        assert detections[0].class_id == 0

    def test_parse_detections_all_rows_malformed(self):
        """_parse_detections() should return empty list when all rows are malformed."""
        client = self._make_client()

        # All rows have fewer than 6 values (uniform shape so numpy can create 2D array)
        output = np.array([
            [10.0, 20.0, 100.0, 0.0, 0.0],
            [30.0, 40.0, 50.0, 0.0, 0.0],
        ], dtype=np.float32)

        detections = client._parse_detections(output)
        assert len(detections) == 0


class TestTritonClientInit:
    """Tests for the TritonClient.__init__() method."""

    def test_init_success_with_mocked_tritonclient(self):
        """__init__ should connect successfully when tritonclient.grpc is available."""
        import importlib

        mock_grpcclient = MagicMock()
        mock_inference_client = MagicMock()
        mock_grpcclient.InferenceServerClient.return_value = mock_inference_client

        # The parent module must have .grpc attribute pointing to our mock
        mock_tritonclient_parent = MagicMock()
        mock_tritonclient_parent.grpc = mock_grpcclient

        with patch.dict(
            "sys.modules",
            {
                "tritonclient": mock_tritonclient_parent,
                "tritonclient.grpc": mock_grpcclient,
            },
        ):
            import metropolis.triton_client as tc_module

            importlib.reload(tc_module)
            TritonClient = tc_module.TritonClient

            client = TritonClient(
                server_url="triton-server:8001",
                model_name="custom_model",
            )

            assert client.server_url == "triton-server:8001"
            assert client.model_name == "custom_model"
            assert client._connected is True
            assert client._client is mock_inference_client
            assert client._grpcclient is mock_grpcclient
            mock_grpcclient.InferenceServerClient.assert_called_once_with(
                url="triton-server:8001",
                verbose=False,
            )

    def test_init_import_error_sets_disconnected_state(self):
        """__init__ should handle ImportError gracefully and set disconnected state."""
        import importlib

        # Simulate tritonclient not being installed
        with patch.dict(
            "sys.modules",
            {"tritonclient": None, "tritonclient.grpc": None},
        ):
            import metropolis.triton_client as tc_module

            importlib.reload(tc_module)
            TritonClient = tc_module.TritonClient

            client = TritonClient(
                server_url="localhost:8001",
                model_name="yolov8_ensemble",
            )

            assert client._connected is False
            assert client._client is None
            assert client._grpcclient is None

    def test_init_connection_error_sets_disconnected_state(self):
        """__init__ should handle connection errors gracefully."""
        import importlib

        mock_grpcclient = MagicMock()
        mock_grpcclient.InferenceServerClient.side_effect = Exception(
            "Connection refused"
        )

        mock_tritonclient_parent = MagicMock()
        mock_tritonclient_parent.grpc = mock_grpcclient

        with patch.dict(
            "sys.modules",
            {
                "tritonclient": mock_tritonclient_parent,
                "tritonclient.grpc": mock_grpcclient,
            },
        ):
            import metropolis.triton_client as tc_module

            importlib.reload(tc_module)
            TritonClient = tc_module.TritonClient

            client = TritonClient(
                server_url="unreachable:8001",
                model_name="yolov8_ensemble",
            )

            assert client.server_url == "unreachable:8001"
            assert client._connected is False
            assert client._client is None
            assert client._grpcclient is None

    def test_init_default_parameters(self):
        """__init__ should use default server_url and model_name."""
        import importlib

        mock_grpcclient = MagicMock()
        mock_grpcclient.InferenceServerClient.return_value = MagicMock()

        mock_tritonclient_parent = MagicMock()
        mock_tritonclient_parent.grpc = mock_grpcclient

        with patch.dict(
            "sys.modules",
            {
                "tritonclient": mock_tritonclient_parent,
                "tritonclient.grpc": mock_grpcclient,
            },
        ):
            import metropolis.triton_client as tc_module

            importlib.reload(tc_module)
            TritonClient = tc_module.TritonClient

            client = TritonClient()

            assert client.server_url == "localhost:8001"
            assert client.model_name == "yolov8_ensemble"
            assert client._fallback_enabled is False
            assert client._using_fallback is False
            assert client._fallback_engine_path is None


class TestTritonClientCloseStateReset:
    """Tests verifying close() resets all state properly."""

    def _make_client_fully_active(self):
        """Create a TritonClient with all state set to active values."""
        from metropolis.triton_client import TritonClient

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = True
        client._client = MagicMock()
        client._grpcclient = MagicMock()
        client._fallback_enabled = True
        client._using_fallback = True
        client._fallback_engine_path = "/path/to/engine.engine"
        return client

    def test_close_resets_connected_state(self):
        """close() should set _connected to False."""
        client = self._make_client_fully_active()
        client.close()
        assert client._connected is False

    def test_close_resets_model_ready_state(self):
        """close() should set _model_ready to False."""
        client = self._make_client_fully_active()
        client.close()
        assert client._model_ready is False

    def test_close_sets_client_to_none(self):
        """close() should set _client to None."""
        client = self._make_client_fully_active()
        assert client._client is not None
        client.close()
        assert client._client is None

    def test_close_with_active_polling_stops_thread(self):
        """close() should stop the polling thread if it's running."""
        from metropolis.triton_client import TritonClient

        mock_grpc_client = MagicMock()
        mock_grpc_client.is_server_live.return_value = True
        mock_grpc_client.is_model_ready.return_value = True

        client = TritonClient.__new__(TritonClient)
        client.server_url = "localhost:8001"
        client.model_name = "yolov8_ensemble"
        client._connected = True
        client._model_ready = True
        client._client = mock_grpc_client
        client._grpcclient = MagicMock()
        client._fallback_enabled = False
        client._using_fallback = False
        client._fallback_engine_path = None

        # Start polling
        client.start_health_polling(interval=0.1)
        assert client._polling_thread.is_alive()

        # Close should stop polling
        client.close()
        assert not client._polling_thread.is_alive()
        assert client._connected is False
        assert client._model_ready is False
        assert client._client is None

    def test_close_idempotent(self):
        """close() should be safe to call multiple times."""
        client = self._make_client_fully_active()

        # Call close twice - should not raise
        client.close()
        client.close()

        assert client._connected is False
        assert client._model_ready is False
        assert client._client is None
