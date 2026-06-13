"""Unit tests for DeepStreamPipeline._build_pipeline() method.

Tests verify pipeline construction logic including element creation,
property setting, element addition to pipeline, and element linking.
All tests mock GStreamer/DeepStream bindings so they run without GPU hardware.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure the backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metropolis.deepstream_pipeline import DeepStreamPipeline, DeepStreamConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Create a default DeepStreamConfig."""
    return DeepStreamConfig()


@pytest.fixture
def custom_config():
    """Create a custom DeepStreamConfig with non-default values."""
    return DeepStreamConfig(
        nvinfer_config="custom/nvinfer.txt",
        tracker_lib="/custom/lib/tracker.so",
        tracker_config="custom/tracker.yml",
        mux_width=1280,
        mux_height=720,
        mux_batch_size=2,
        mux_batched_push_timeout=50000,
        gpu_id=1,
    )


@pytest.fixture
def mock_gst():
    """Create a mock Gst module with working ElementFactory."""
    gst = MagicMock()

    # Track created elements by name for verification
    elements = {}

    def make_element(factory_name, element_name):
        elem = MagicMock()
        elem.name = element_name
        elem.factory_name = factory_name
        elem.link.return_value = True
        elements[(factory_name, element_name)] = elem
        return elem

    gst.ElementFactory.make.side_effect = make_element
    gst.Pipeline.return_value = MagicMock()

    return gst, elements


# ---------------------------------------------------------------------------
# Tests: Successful pipeline construction
# ---------------------------------------------------------------------------


class TestBuildPipelineSuccess:
    """Tests for successful _build_pipeline() execution."""

    def test_initializes_gstreamer(self, default_config, mock_gst):
        """_build_pipeline calls Gst.init(None)."""
        gst, _ = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        gst.init.assert_called_once_with(None)

    def test_creates_gst_pipeline(self, default_config, mock_gst):
        """_build_pipeline creates a Gst.Pipeline instance."""
        gst, _ = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        gst.Pipeline.assert_called_once()

    def test_creates_all_elements(self, default_config, mock_gst):
        """_build_pipeline creates all required pipeline elements."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        expected_elements = [
            call("nvstreammux", "mux"),
            call("nvinfer", "primary-inference"),
            call("nvtracker", "tracker"),
            call("nvdsosd", "osd"),
            call("fakesink", "sink"),
        ]
        gst.ElementFactory.make.assert_has_calls(expected_elements, any_order=False)

    def test_sets_streammux_properties(self, default_config, mock_gst):
        """_build_pipeline sets correct properties on nvstreammux."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        mux = elements[("nvstreammux", "mux")]
        mux.set_property.assert_any_call("batch-size", 4)
        mux.set_property.assert_any_call("width", 1920)
        mux.set_property.assert_any_call("height", 1080)
        mux.set_property.assert_any_call("batched-push-timeout", 40000)
        mux.set_property.assert_any_call("gpu-id", 0)

    def test_sets_streammux_custom_properties(self, custom_config, mock_gst):
        """_build_pipeline uses config values for streammux properties."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(custom_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        mux = elements[("nvstreammux", "mux")]
        mux.set_property.assert_any_call("batch-size", 2)
        mux.set_property.assert_any_call("width", 1280)
        mux.set_property.assert_any_call("height", 720)
        mux.set_property.assert_any_call("batched-push-timeout", 50000)
        mux.set_property.assert_any_call("gpu-id", 1)

    def test_sets_nvinfer_properties(self, default_config, mock_gst):
        """_build_pipeline sets correct properties on nvinfer."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        nvinfer = elements[("nvinfer", "primary-inference")]
        nvinfer.set_property.assert_any_call("config-file-path", "configs/nvinfer_config.txt")
        nvinfer.set_property.assert_any_call("gpu-id", 0)

    def test_sets_tracker_properties(self, default_config, mock_gst):
        """_build_pipeline sets correct properties on nvtracker."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        tracker = elements[("nvtracker", "tracker")]
        tracker.set_property.assert_any_call("tracker-width", 640)
        tracker.set_property.assert_any_call("tracker-height", 384)
        tracker.set_property.assert_any_call(
            "ll-lib-file",
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        )
        tracker.set_property.assert_any_call("ll-config-file", "configs/tracker_config.yml")
        tracker.set_property.assert_any_call("gpu-id", 0)

    def test_sets_fakesink_sync_false(self, default_config, mock_gst):
        """_build_pipeline sets sync=False on fakesink."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        sink = elements[("fakesink", "sink")]
        sink.set_property.assert_any_call("sync", False)

    def test_adds_all_elements_to_pipeline(self, default_config, mock_gst):
        """_build_pipeline adds all elements to the Gst.Pipeline."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        gst_pipeline = gst.Pipeline.return_value
        assert gst_pipeline.add.call_count == 5

    def test_links_elements_in_order(self, default_config, mock_gst):
        """_build_pipeline links elements: streammux → nvinfer → tracker → osd → sink."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        mux = elements[("nvstreammux", "mux")]
        nvinfer = elements[("nvinfer", "primary-inference")]
        tracker = elements[("nvtracker", "tracker")]
        osd = elements[("nvdsosd", "osd")]

        mux.link.assert_called_once_with(nvinfer)
        nvinfer.link.assert_called_once_with(tracker)
        tracker.link.assert_called_once_with(osd)
        osd.link.assert_called_once_with(elements[("fakesink", "sink")])

    def test_stores_element_references(self, default_config, mock_gst):
        """_build_pipeline stores element references on self."""
        gst, elements = mock_gst
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._build_pipeline()

        assert pipeline._streammux is elements[("nvstreammux", "mux")]
        assert pipeline._nvinfer is elements[("nvinfer", "primary-inference")]
        assert pipeline._tracker is elements[("nvtracker", "tracker")]
        assert pipeline._nvosd is elements[("nvdsosd", "osd")]
        assert pipeline._fakesink is elements[("fakesink", "sink")]
        assert pipeline._pipeline is gst.Pipeline.return_value


# ---------------------------------------------------------------------------
# Tests: Element creation failures
# ---------------------------------------------------------------------------


class TestBuildPipelineFailures:
    """Tests for _build_pipeline() error handling."""

    def test_raises_on_streammux_creation_failure(self, default_config):
        """RuntimeError raised when nvstreammux fails to create."""
        gst = MagicMock()
        gst.Pipeline.return_value = MagicMock()
        gst.ElementFactory.make.return_value = None

        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="nvstreammux"):
                pipeline._build_pipeline()

    def test_raises_on_nvinfer_creation_failure(self, default_config):
        """RuntimeError raised when nvinfer fails to create."""
        gst = MagicMock()
        gst.Pipeline.return_value = MagicMock()

        call_count = [0]

        def make_side_effect(factory_name, element_name):
            call_count[0] += 1
            if factory_name == "nvinfer":
                return None
            elem = MagicMock()
            elem.link.return_value = True
            return elem

        gst.ElementFactory.make.side_effect = make_side_effect
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="nvinfer"):
                pipeline._build_pipeline()

    def test_raises_on_tracker_creation_failure(self, default_config):
        """RuntimeError raised when nvtracker fails to create."""
        gst = MagicMock()
        gst.Pipeline.return_value = MagicMock()

        def make_side_effect(factory_name, element_name):
            if factory_name == "nvtracker":
                return None
            elem = MagicMock()
            elem.link.return_value = True
            return elem

        gst.ElementFactory.make.side_effect = make_side_effect
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="nvtracker"):
                pipeline._build_pipeline()

    def test_raises_on_osd_creation_failure(self, default_config):
        """RuntimeError raised when nvdsosd fails to create."""
        gst = MagicMock()
        gst.Pipeline.return_value = MagicMock()

        def make_side_effect(factory_name, element_name):
            if factory_name == "nvdsosd":
                return None
            elem = MagicMock()
            elem.link.return_value = True
            return elem

        gst.ElementFactory.make.side_effect = make_side_effect
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="nvdsosd"):
                pipeline._build_pipeline()

    def test_raises_on_fakesink_creation_failure(self, default_config):
        """RuntimeError raised when fakesink fails to create."""
        gst = MagicMock()
        gst.Pipeline.return_value = MagicMock()

        def make_side_effect(factory_name, element_name):
            if factory_name == "fakesink":
                return None
            elem = MagicMock()
            elem.link.return_value = True
            return elem

        gst.ElementFactory.make.side_effect = make_side_effect
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="fakesink"):
                pipeline._build_pipeline()

    def test_raises_on_link_failure(self, default_config):
        """RuntimeError raised when element linking fails."""
        gst = MagicMock()
        gst.Pipeline.return_value = MagicMock()

        def make_side_effect(factory_name, element_name):
            elem = MagicMock()
            # Make the first link call fail
            elem.link.return_value = False
            return elem

        gst.ElementFactory.make.side_effect = make_side_effect
        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="Failed to link"):
                pipeline._build_pipeline()

    def test_raises_on_pipeline_creation_failure(self, default_config):
        """RuntimeError raised when Gst.Pipeline() returns falsy."""
        gst = MagicMock()
        gst.Pipeline.return_value = None

        pipeline = DeepStreamPipeline(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="Failed to create GStreamer Pipeline"):
                pipeline._build_pipeline()


# ---------------------------------------------------------------------------
# Tests: _create_source_bin()
# ---------------------------------------------------------------------------


class TestCreateSourceBinRTSP:
    """Tests for _create_source_bin() with RTSP sources."""

    def test_creates_bin_with_correct_name(self, default_config):
        """Source bin is named 'source-bin-{source_id}'."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        gst.ElementFactory.make.return_value = MagicMock()
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera1:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        gst.Bin.new.assert_called_once_with("source-bin-0")

    def test_creates_uridecodebin_for_rtsp(self, default_config):
        """RTSP source uses uridecodebin element."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        uri_decode = MagicMock()
        gst.ElementFactory.make.return_value = uri_decode
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera1:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        gst.ElementFactory.make.assert_called_once_with("uridecodebin", "uri-decode-0")

    def test_sets_uri_property_for_rtsp(self, default_config):
        """RTSP URI is set directly on uridecodebin."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        uri_decode = MagicMock()
        gst.ElementFactory.make.return_value = uri_decode
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        rtsp_uri = "rtsp://camera1:554/stream"
        source_info = {"uri": rtsp_uri, "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        uri_decode.set_property.assert_called_once_with("uri", rtsp_uri)

    def test_creates_ghost_pad_with_no_target(self, default_config):
        """RTSP source bin has a ghost pad named 'src' with no initial target."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        gst.ElementFactory.make.return_value = MagicMock()
        ghost_pad = MagicMock()
        gst.GhostPad.new_no_target.return_value = ghost_pad
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera1:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        gst.GhostPad.new_no_target.assert_called_once_with("src", 1)
        ghost_pad.set_active.assert_called_once_with(True)
        source_bin.add_pad.assert_called_once_with(ghost_pad)

    def test_connects_pad_added_signal(self, default_config):
        """RTSP source connects pad-added signal on uridecodebin."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        uri_decode = MagicMock()
        gst.ElementFactory.make.return_value = uri_decode
        ghost_pad = MagicMock()
        gst.GhostPad.new_no_target.return_value = ghost_pad
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera1:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        uri_decode.connect.assert_called_once_with(
            "pad-added",
            DeepStreamPipeline._on_source_bin_pad_added,
            ghost_pad,
        )

    def test_returns_source_bin(self, default_config):
        """_create_source_bin returns the created bin."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        gst.ElementFactory.make.return_value = MagicMock()
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera1:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            result = pipeline._create_source_bin(0, source_info)

        assert result is source_bin


class TestCreateSourceBinFile:
    """Tests for _create_source_bin() with file sources."""

    def test_prepends_file_prefix_to_uri(self, default_config):
        """File source prepends 'file://' to URI if not present."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        uri_decode = MagicMock()
        gst.ElementFactory.make.return_value = uri_decode
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/path/to/video.mp4", "source_type": "file"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        uri_decode.set_property.assert_called_once_with("uri", "file:///path/to/video.mp4")

    def test_does_not_double_prefix_file_uri(self, default_config):
        """File source does not add 'file://' if already present."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        uri_decode = MagicMock()
        gst.ElementFactory.make.return_value = uri_decode
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "file:///path/to/video.mp4", "source_type": "file"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        uri_decode.set_property.assert_called_once_with("uri", "file:///path/to/video.mp4")

    def test_uses_uridecodebin_for_file(self, default_config):
        """File source uses uridecodebin element."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin
        gst.ElementFactory.make.return_value = MagicMock()
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/path/to/video.mp4", "source_type": "file"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(1, source_info)

        gst.ElementFactory.make.assert_called_once_with("uridecodebin", "uri-decode-1")


class TestCreateSourceBinUSB:
    """Tests for _create_source_bin() with USB sources."""

    def test_creates_v4l2src_videoconvert_nvvideoconvert(self, default_config):
        """USB source creates v4l2src, videoconvert, and nvvideoconvert."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin

        elements = {}

        def make_element(factory_name, element_name):
            elem = MagicMock()
            elem.link.return_value = True
            elem.get_static_pad.return_value = MagicMock()
            elements[(factory_name, element_name)] = elem
            return elem

        gst.ElementFactory.make.side_effect = make_element
        gst.GhostPad.new.return_value = MagicMock()

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        expected_calls = [
            call("v4l2src", "v4l2src-0"),
            call("videoconvert", "videoconvert-0"),
            call("nvvideoconvert", "nvvideoconvert-0"),
        ]
        gst.ElementFactory.make.assert_has_calls(expected_calls, any_order=False)

    def test_sets_device_property_on_v4l2src(self, default_config):
        """USB source sets device property to the URI."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin

        v4l2src = MagicMock()
        v4l2src.link.return_value = True

        def make_element(factory_name, element_name):
            if factory_name == "v4l2src":
                return v4l2src
            elem = MagicMock()
            elem.link.return_value = True
            elem.get_static_pad.return_value = MagicMock()
            return elem

        gst.ElementFactory.make.side_effect = make_element
        gst.GhostPad.new.return_value = MagicMock()

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        v4l2src.set_property.assert_called_once_with("device", "/dev/video0")

    def test_links_usb_elements_in_order(self, default_config):
        """USB source links: v4l2src → videoconvert → nvvideoconvert."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin

        elements = {}

        def make_element(factory_name, element_name):
            elem = MagicMock()
            elem.link.return_value = True
            elem.get_static_pad.return_value = MagicMock()
            elements[factory_name] = elem
            return elem

        gst.ElementFactory.make.side_effect = make_element
        gst.GhostPad.new.return_value = MagicMock()

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        elements["v4l2src"].link.assert_called_once_with(elements["videoconvert"])
        elements["videoconvert"].link.assert_called_once_with(elements["nvvideoconvert"])

    def test_creates_ghost_pad_from_nvvideoconvert_src(self, default_config):
        """USB source bin ghost pad targets nvvideoconvert's src pad."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin

        nvvideoconvert_src_pad = MagicMock()

        def make_element(factory_name, element_name):
            elem = MagicMock()
            elem.link.return_value = True
            if factory_name == "nvvideoconvert":
                elem.get_static_pad.return_value = nvvideoconvert_src_pad
            return elem

        gst.ElementFactory.make.side_effect = make_element
        ghost_pad = MagicMock()
        gst.GhostPad.new.return_value = ghost_pad

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        gst.GhostPad.new.assert_called_once_with("src", nvvideoconvert_src_pad)
        ghost_pad.set_active.assert_called_once_with(True)
        source_bin.add_pad.assert_called_once_with(ghost_pad)

    def test_adds_all_elements_to_bin(self, default_config):
        """USB source adds all three elements to the bin."""
        gst = MagicMock()
        source_bin = MagicMock()
        gst.Bin.new.return_value = source_bin

        elements_list = []

        def make_element(factory_name, element_name):
            elem = MagicMock()
            elem.link.return_value = True
            elem.get_static_pad.return_value = MagicMock()
            elements_list.append(elem)
            return elem

        gst.ElementFactory.make.side_effect = make_element
        gst.GhostPad.new.return_value = MagicMock()

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline._create_source_bin(0, source_info)

        assert source_bin.add.call_count == 3


class TestCreateSourceBinErrors:
    """Tests for _create_source_bin() error handling."""

    def test_raises_on_bin_creation_failure(self, default_config):
        """RuntimeError raised when Gst.Bin.new returns None."""
        gst = MagicMock()
        gst.Bin.new.return_value = None

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="Failed to create source bin"):
                pipeline._create_source_bin(0, source_info)

    def test_raises_on_uridecodebin_creation_failure(self, default_config):
        """RuntimeError raised when uridecodebin fails to create."""
        gst = MagicMock()
        gst.Bin.new.return_value = MagicMock()
        gst.ElementFactory.make.return_value = None

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="uridecodebin"):
                pipeline._create_source_bin(0, source_info)

    def test_raises_on_v4l2src_creation_failure(self, default_config):
        """RuntimeError raised when v4l2src fails to create."""
        gst = MagicMock()
        gst.Bin.new.return_value = MagicMock()
        gst.ElementFactory.make.return_value = None

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="v4l2src"):
                pipeline._create_source_bin(0, source_info)

    def test_raises_on_videoconvert_creation_failure(self, default_config):
        """RuntimeError raised when videoconvert fails to create."""
        gst = MagicMock()
        gst.Bin.new.return_value = MagicMock()

        def make_element(factory_name, element_name):
            if factory_name == "videoconvert":
                return None
            elem = MagicMock()
            elem.link.return_value = True
            return elem

        gst.ElementFactory.make.side_effect = make_element

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="videoconvert"):
                pipeline._create_source_bin(0, source_info)

    def test_raises_on_nvvideoconvert_creation_failure(self, default_config):
        """RuntimeError raised when nvvideoconvert fails to create."""
        gst = MagicMock()
        gst.Bin.new.return_value = MagicMock()

        def make_element(factory_name, element_name):
            if factory_name == "nvvideoconvert":
                return None
            elem = MagicMock()
            elem.link.return_value = True
            return elem

        gst.ElementFactory.make.side_effect = make_element

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="nvvideoconvert"):
                pipeline._create_source_bin(0, source_info)

    def test_raises_on_usb_link_failure(self, default_config):
        """RuntimeError raised when USB element linking fails."""
        gst = MagicMock()
        gst.Bin.new.return_value = MagicMock()

        def make_element(factory_name, element_name):
            elem = MagicMock()
            elem.link.return_value = False  # All links fail
            return elem

        gst.ElementFactory.make.side_effect = make_element

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "/dev/video0", "source_type": "usb"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="Failed to link"):
                pipeline._create_source_bin(0, source_info)

    def test_raises_on_unsupported_source_type(self, default_config):
        """RuntimeError raised for unsupported source type."""
        gst = MagicMock()
        gst.Bin.new.return_value = MagicMock()

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "something", "source_type": "unknown"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="Unsupported source type"):
                pipeline._create_source_bin(0, source_info)

    def test_raises_on_ghost_pad_creation_failure_rtsp(self, default_config):
        """RuntimeError raised when ghost pad creation fails for RTSP."""
        gst = MagicMock()
        gst.Bin.new.return_value = MagicMock()
        gst.ElementFactory.make.return_value = MagicMock()
        gst.GhostPad.new_no_target.return_value = None
        gst.PadDirection.SRC = 1

        pipeline = DeepStreamPipeline(default_config)
        source_info = {"uri": "rtsp://camera:554/stream", "source_type": "rtsp"}

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with pytest.raises(RuntimeError, match="ghost pad"):
                pipeline._create_source_bin(0, source_info)


class TestOnSourceBinPadAdded:
    """Tests for the _on_source_bin_pad_added static callback."""

    def test_links_video_pad_to_ghost_pad(self):
        """Video pad is linked to ghost pad via set_target."""
        pad = MagicMock()
        caps = MagicMock()
        structure = MagicMock()
        structure.get_name.return_value = "video/x-raw"
        caps.get_structure.return_value = structure
        pad.get_current_caps.return_value = caps

        ghost_pad = MagicMock()
        ghost_pad.set_target.return_value = True

        decodebin = MagicMock()

        DeepStreamPipeline._on_source_bin_pad_added(decodebin, pad, ghost_pad)

        ghost_pad.set_target.assert_called_once_with(pad)

    def test_ignores_audio_pad(self):
        """Audio pad is not linked to ghost pad."""
        pad = MagicMock()
        caps = MagicMock()
        structure = MagicMock()
        structure.get_name.return_value = "audio/x-raw"
        caps.get_structure.return_value = structure
        pad.get_current_caps.return_value = caps

        ghost_pad = MagicMock()
        decodebin = MagicMock()

        DeepStreamPipeline._on_source_bin_pad_added(decodebin, pad, ghost_pad)

        ghost_pad.set_target.assert_not_called()

    def test_uses_query_caps_when_current_caps_none(self):
        """Falls back to query_caps when get_current_caps returns None."""
        pad = MagicMock()
        pad.get_current_caps.return_value = None
        caps = MagicMock()
        structure = MagicMock()
        structure.get_name.return_value = "video/x-h264"
        caps.get_structure.return_value = structure
        pad.query_caps.return_value = caps

        ghost_pad = MagicMock()
        ghost_pad.set_target.return_value = True
        decodebin = MagicMock()

        DeepStreamPipeline._on_source_bin_pad_added(decodebin, pad, ghost_pad)

        pad.query_caps.assert_called_once_with(None)
        ghost_pad.set_target.assert_called_once_with(pad)


# ---------------------------------------------------------------------------
# Tests: start() method
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gi_glib():
    """Mock the gi.repository.GLib module for start() tests."""
    mock_glib = MagicMock()
    mock_gi = MagicMock()
    mock_gi_repository = MagicMock()
    mock_gi_repository.GLib = mock_glib

    with patch.dict(sys.modules, {
        "gi": mock_gi,
        "gi.repository": mock_gi_repository,
        "gi.repository.GLib": mock_glib,
    }):
        yield mock_glib


class TestStartMethod:
    """Tests for DeepStreamPipeline.start() method."""

    def _setup_pipeline_with_source(self, config):
        """Helper to create a pipeline with one source added."""
        pipeline = DeepStreamPipeline(config)
        pipeline.add_source(0, "rtsp://camera1:554/stream", "rtsp")
        return pipeline

    def _mock_gst_for_start(self):
        """Create mock Gst module suitable for start() testing."""
        gst = MagicMock()
        gst.Pipeline.return_value = MagicMock()
        gst.State.PLAYING = "PLAYING"
        gst.State.NULL = "NULL"
        gst.StateChangeReturn.FAILURE = "FAILURE"
        gst.PadProbeType.BUFFER = 1
        gst.PadLinkReturn.OK = 0

        def make_element(factory_name, element_name):
            elem = MagicMock()
            elem.link.return_value = True
            elem.get_static_pad.return_value = MagicMock()
            elem.get_request_pad.return_value = MagicMock()
            return elem

        gst.ElementFactory.make.side_effect = make_element

        # Source bin mocks
        source_bin = MagicMock()
        src_pad = MagicMock()
        src_pad.link.return_value = 0  # Gst.PadLinkReturn.OK
        source_bin.get_static_pad.return_value = src_pad
        gst.Bin.new.return_value = source_bin
        gst.GhostPad.new_no_target.return_value = MagicMock()
        gst.PadDirection.SRC = 1

        # Pipeline set_state returns success
        gst.Pipeline.return_value.set_state.return_value = "SUCCESS"

        return gst

    def test_raises_if_already_running(self, default_config):
        """start() raises RuntimeError if pipeline is already running."""
        pipeline = self._setup_pipeline_with_source(default_config)
        pipeline._running = True

        with pytest.raises(RuntimeError, match="already running"):
            pipeline.start()

    def test_raises_if_no_sources(self, default_config):
        """start() raises RuntimeError if no sources have been added."""
        pipeline = DeepStreamPipeline(default_config)

        with pytest.raises(RuntimeError, match="No sources"):
            pipeline.start()

    def test_calls_build_pipeline(self, default_config, mock_gi_glib):
        """start() calls _build_pipeline() to construct the pipeline."""
        gst = self._mock_gst_for_start()
        pipeline = self._setup_pipeline_with_source(default_config)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch("metropolis.deepstream_pipeline.threading.Thread") as mock_thread:
                mock_thread.return_value = MagicMock()
                with patch.object(pipeline, "_build_pipeline") as mock_build:
                    with patch.object(pipeline, "_create_source_bin") as mock_create:
                        source_bin = MagicMock()
                        src_pad = MagicMock()
                        src_pad.link.return_value = gst.PadLinkReturn.OK
                        source_bin.get_static_pad.return_value = src_pad
                        mock_create.return_value = source_bin

                        def setup_elements():
                            pipeline._streammux = MagicMock()
                            pipeline._streammux.get_request_pad.return_value = MagicMock()
                            pipeline._nvosd = MagicMock()
                            pipeline._nvosd.get_static_pad.return_value = MagicMock()
                            pipeline._pipeline = MagicMock()
                            pipeline._pipeline.set_state.return_value = "SUCCESS"

                        mock_build.side_effect = setup_elements
                        pipeline.start()

                    mock_build.assert_called_once()

    def test_creates_source_bins_for_each_source(self, default_config, mock_gi_glib):
        """start() creates a source bin for each registered source."""
        pipeline = DeepStreamPipeline(default_config)
        pipeline.add_source(0, "rtsp://cam1:554/stream", "rtsp")
        pipeline.add_source(1, "rtsp://cam2:554/stream", "rtsp")

        gst = self._mock_gst_for_start()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch("metropolis.deepstream_pipeline.threading.Thread") as mock_thread:
                mock_thread.return_value = MagicMock()
                with patch.object(pipeline, "_build_pipeline") as mock_build:
                    with patch.object(pipeline, "_create_source_bin") as mock_create:
                        source_bin = MagicMock()
                        src_pad = MagicMock()
                        src_pad.link.return_value = gst.PadLinkReturn.OK
                        source_bin.get_static_pad.return_value = src_pad
                        mock_create.return_value = source_bin

                        def setup_elements():
                            pipeline._streammux = MagicMock()
                            pipeline._streammux.get_request_pad.return_value = MagicMock()
                            pipeline._nvosd = MagicMock()
                            pipeline._nvosd.get_static_pad.return_value = MagicMock()
                            pipeline._pipeline = MagicMock()
                            pipeline._pipeline.set_state.return_value = "SUCCESS"

                        mock_build.side_effect = setup_elements
                        pipeline.start()

                    assert mock_create.call_count == 2

    def test_sets_running_true_on_success(self, default_config, mock_gi_glib):
        """start() sets self._running = True on successful start."""
        pipeline = self._setup_pipeline_with_source(default_config)
        gst = self._mock_gst_for_start()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch("metropolis.deepstream_pipeline.threading.Thread") as mock_thread:
                mock_thread.return_value = MagicMock()
                with patch.object(pipeline, "_build_pipeline") as mock_build:
                    with patch.object(pipeline, "_create_source_bin") as mock_create:
                        source_bin = MagicMock()
                        src_pad = MagicMock()
                        src_pad.link.return_value = gst.PadLinkReturn.OK
                        source_bin.get_static_pad.return_value = src_pad
                        mock_create.return_value = source_bin

                        def setup_elements():
                            pipeline._streammux = MagicMock()
                            pipeline._streammux.get_request_pad.return_value = MagicMock()
                            pipeline._nvosd = MagicMock()
                            pipeline._nvosd.get_static_pad.return_value = MagicMock()
                            pipeline._pipeline = MagicMock()
                            pipeline._pipeline.set_state.return_value = "SUCCESS"

                        mock_build.side_effect = setup_elements
                        pipeline.start()

        assert pipeline._running is True

    def test_starts_glib_mainloop_in_daemon_thread(self, default_config, mock_gi_glib):
        """start() creates a daemon thread running the GLib MainLoop."""
        pipeline = self._setup_pipeline_with_source(default_config)
        gst = self._mock_gst_for_start()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch("metropolis.deepstream_pipeline.threading.Thread") as mock_thread:
                thread_instance = MagicMock()
                mock_thread.return_value = thread_instance
                with patch.object(pipeline, "_build_pipeline") as mock_build:
                    with patch.object(pipeline, "_create_source_bin") as mock_create:
                        source_bin = MagicMock()
                        src_pad = MagicMock()
                        src_pad.link.return_value = gst.PadLinkReturn.OK
                        source_bin.get_static_pad.return_value = src_pad
                        mock_create.return_value = source_bin

                        def setup_elements():
                            pipeline._streammux = MagicMock()
                            pipeline._streammux.get_request_pad.return_value = MagicMock()
                            pipeline._nvosd = MagicMock()
                            pipeline._nvosd.get_static_pad.return_value = MagicMock()
                            pipeline._pipeline = MagicMock()
                            pipeline._pipeline.set_state.return_value = "SUCCESS"

                        mock_build.side_effect = setup_elements
                        pipeline.start()

                # Verify thread was created with daemon=True
                mock_thread.assert_called_once()
                call_kwargs = mock_thread.call_args[1]
                assert call_kwargs["daemon"] is True
                assert call_kwargs["name"] == "glib-mainloop"
                thread_instance.start.assert_called_once()

    def test_sets_pipeline_state_to_playing(self, default_config, mock_gi_glib):
        """start() sets pipeline state to PLAYING."""
        pipeline = self._setup_pipeline_with_source(default_config)
        gst = self._mock_gst_for_start()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch("metropolis.deepstream_pipeline.threading.Thread") as mock_thread:
                mock_thread.return_value = MagicMock()
                with patch.object(pipeline, "_build_pipeline") as mock_build:
                    with patch.object(pipeline, "_create_source_bin") as mock_create:
                        source_bin = MagicMock()
                        src_pad = MagicMock()
                        src_pad.link.return_value = gst.PadLinkReturn.OK
                        source_bin.get_static_pad.return_value = src_pad
                        mock_create.return_value = source_bin

                        mock_pipeline = MagicMock()
                        mock_pipeline.set_state.return_value = "SUCCESS"

                        def setup_elements():
                            pipeline._streammux = MagicMock()
                            pipeline._streammux.get_request_pad.return_value = MagicMock()
                            pipeline._nvosd = MagicMock()
                            pipeline._nvosd.get_static_pad.return_value = MagicMock()
                            pipeline._pipeline = mock_pipeline

                        mock_build.side_effect = setup_elements
                        pipeline.start()

                mock_pipeline.set_state.assert_called_once_with("PLAYING")

    def test_attaches_analytics_probe(self, default_config, mock_gi_glib):
        """start() attaches analytics probe to nvosd sink pad."""
        pipeline = self._setup_pipeline_with_source(default_config)
        gst = self._mock_gst_for_start()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch("metropolis.deepstream_pipeline.threading.Thread") as mock_thread:
                mock_thread.return_value = MagicMock()
                with patch.object(pipeline, "_build_pipeline") as mock_build:
                    with patch.object(pipeline, "_create_source_bin") as mock_create:
                        source_bin = MagicMock()
                        src_pad = MagicMock()
                        src_pad.link.return_value = gst.PadLinkReturn.OK
                        source_bin.get_static_pad.return_value = src_pad
                        mock_create.return_value = source_bin

                        osd_sink_pad = MagicMock()
                        mock_nvosd = MagicMock()
                        mock_nvosd.get_static_pad.return_value = osd_sink_pad

                        def setup_elements():
                            pipeline._streammux = MagicMock()
                            pipeline._streammux.get_request_pad.return_value = MagicMock()
                            pipeline._nvosd = mock_nvosd
                            pipeline._pipeline = MagicMock()
                            pipeline._pipeline.set_state.return_value = "SUCCESS"

                        mock_build.side_effect = setup_elements
                        pipeline.start()

                mock_nvosd.get_static_pad.assert_called_once_with("sink")
                osd_sink_pad.add_probe.assert_called_once_with(
                    gst.PadProbeType.BUFFER,
                    pipeline._analytics_probe_callback,
                    None,
                )


# ---------------------------------------------------------------------------
# Tests: stop() method
# ---------------------------------------------------------------------------


class TestStopMethod:
    """Tests for DeepStreamPipeline.stop() method."""

    def test_returns_early_if_not_running(self, default_config):
        """stop() returns without error if pipeline is not running."""
        pipeline = DeepStreamPipeline(default_config)
        assert pipeline._running is False
        # Should not raise
        pipeline.stop()

    def test_sets_pipeline_state_to_null(self, default_config):
        """stop() sets pipeline state to NULL."""
        gst = MagicMock()
        gst.State.NULL = "NULL"

        pipeline = DeepStreamPipeline(default_config)
        pipeline._running = True
        mock_pipeline = MagicMock()
        pipeline._pipeline = mock_pipeline
        pipeline._main_loop = MagicMock()
        pipeline._loop_thread = MagicMock()
        pipeline._loop_thread.is_alive.return_value = False
        pipeline._streammux = MagicMock()
        pipeline._nvinfer = MagicMock()
        pipeline._tracker = MagicMock()
        pipeline._nvosd = MagicMock()
        pipeline._fakesink = MagicMock()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline.stop()

        mock_pipeline.set_state.assert_called_once_with("NULL")

    def test_quits_main_loop(self, default_config):
        """stop() calls quit() on the GLib MainLoop."""
        gst = MagicMock()
        gst.State.NULL = "NULL"

        pipeline = DeepStreamPipeline(default_config)
        pipeline._running = True
        pipeline._pipeline = MagicMock()
        mock_main_loop = MagicMock()
        pipeline._main_loop = mock_main_loop
        pipeline._loop_thread = MagicMock()
        pipeline._loop_thread.is_alive.return_value = False
        pipeline._streammux = MagicMock()
        pipeline._nvinfer = MagicMock()
        pipeline._tracker = MagicMock()
        pipeline._nvosd = MagicMock()
        pipeline._fakesink = MagicMock()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline.stop()

        mock_main_loop.quit.assert_called_once()

    def test_joins_loop_thread(self, default_config):
        """stop() joins the background thread with timeout."""
        gst = MagicMock()
        gst.State.NULL = "NULL"

        pipeline = DeepStreamPipeline(default_config)
        pipeline._running = True
        pipeline._pipeline = MagicMock()
        pipeline._main_loop = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        pipeline._loop_thread = mock_thread
        pipeline._streammux = MagicMock()
        pipeline._nvinfer = MagicMock()
        pipeline._tracker = MagicMock()
        pipeline._nvosd = MagicMock()
        pipeline._fakesink = MagicMock()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline.stop()

        mock_thread.join.assert_called_once_with(timeout=5.0)

    def test_sets_running_false(self, default_config):
        """stop() sets self._running = False."""
        gst = MagicMock()
        gst.State.NULL = "NULL"

        pipeline = DeepStreamPipeline(default_config)
        pipeline._running = True
        pipeline._pipeline = MagicMock()
        pipeline._main_loop = MagicMock()
        pipeline._loop_thread = MagicMock()
        pipeline._loop_thread.is_alive.return_value = False
        pipeline._streammux = MagicMock()
        pipeline._nvinfer = MagicMock()
        pipeline._tracker = MagicMock()
        pipeline._nvosd = MagicMock()
        pipeline._fakesink = MagicMock()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline.stop()

        assert pipeline._running is False

    def test_cleans_up_references(self, default_config):
        """stop() sets pipeline element references to None."""
        gst = MagicMock()
        gst.State.NULL = "NULL"

        pipeline = DeepStreamPipeline(default_config)
        pipeline._running = True
        pipeline._pipeline = MagicMock()
        pipeline._main_loop = MagicMock()
        pipeline._loop_thread = MagicMock()
        pipeline._loop_thread.is_alive.return_value = False
        pipeline._streammux = MagicMock()
        pipeline._nvinfer = MagicMock()
        pipeline._tracker = MagicMock()
        pipeline._nvosd = MagicMock()
        pipeline._fakesink = MagicMock()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline.stop()

        assert pipeline._pipeline is None
        assert pipeline._main_loop is None
        assert pipeline._loop_thread is None
        assert pipeline._streammux is None
        assert pipeline._nvinfer is None
        assert pipeline._tracker is None
        assert pipeline._nvosd is None
        assert pipeline._fakesink is None

    def test_resets_source_connection_state(self, default_config):
        """stop() resets source element and connected state."""
        gst = MagicMock()
        gst.State.NULL = "NULL"

        pipeline = DeepStreamPipeline(default_config)
        pipeline.add_source(0, "rtsp://cam:554/stream", "rtsp")
        pipeline._running = True
        pipeline._pipeline = MagicMock()
        pipeline._main_loop = MagicMock()
        pipeline._loop_thread = MagicMock()
        pipeline._loop_thread.is_alive.return_value = False
        pipeline._streammux = MagicMock()
        pipeline._nvinfer = MagicMock()
        pipeline._tracker = MagicMock()
        pipeline._nvosd = MagicMock()
        pipeline._fakesink = MagicMock()

        # Simulate that source was connected
        pipeline._sources[0]["element"] = MagicMock()
        pipeline._sources[0]["connected"] = True

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            pipeline.stop()

        assert pipeline._sources[0]["element"] is None
        assert pipeline._sources[0]["connected"] is False

# ---------------------------------------------------------------------------
# Tests: _analytics_probe_callback()
# ---------------------------------------------------------------------------


class TestAnalyticsProbeCallback:
    """Tests for _analytics_probe_callback() metadata extraction."""

    def _make_obj_meta(self, class_id=0, confidence=0.9, left=10, top=20,
                       width=100, height=50, obj_label="person", object_id=1):
        """Create a mock NvDsObjectMeta."""
        obj_meta = MagicMock()
        obj_meta.class_id = class_id
        obj_meta.confidence = confidence
        obj_meta.obj_label = obj_label
        obj_meta.object_id = object_id

        rect_params = MagicMock()
        rect_params.left = left
        rect_params.top = top
        rect_params.width = width
        rect_params.height = height
        obj_meta.rect_params = rect_params

        return obj_meta

    def _make_frame_meta(self, source_id=0, frame_num=1, obj_metas=None):
        """Create a mock NvDsFrameMeta with linked object list."""
        frame_meta = MagicMock()
        frame_meta.source_id = source_id
        frame_meta.frame_num = frame_num

        if obj_metas is None:
            obj_metas = []

        # Build linked list of object metas
        if not obj_metas:
            frame_meta.obj_meta_list = None
        else:
            nodes = []
            for om in obj_metas:
                node = MagicMock()
                node.data = om
                nodes.append(node)
            # Link nodes together
            for i, node in enumerate(nodes):
                if i < len(nodes) - 1:
                    node.next = nodes[i + 1]
                else:
                    node.next = None
            frame_meta.obj_meta_list = nodes[0]

        return frame_meta

    def _make_batch_meta(self, frame_metas):
        """Create a mock batch_meta with linked frame list."""
        batch_meta = MagicMock()

        if not frame_metas:
            batch_meta.frame_meta_list = None
        else:
            nodes = []
            for fm in frame_metas:
                node = MagicMock()
                node.data = fm
                nodes.append(node)
            for i, node in enumerate(nodes):
                if i < len(nodes) - 1:
                    node.next = nodes[i + 1]
                else:
                    node.next = None
            batch_meta.frame_meta_list = nodes[0]

        return batch_meta

    def _setup_pipeline_with_mocks(self, config, batch_meta):
        """Set up a pipeline with mocked GStreamer and pyds."""
        gst = MagicMock()
        gst.PadProbeReturn.OK = 0

        pyds = MagicMock()
        pyds.gst_buffer_get_nvds_batch_meta.return_value = batch_meta

        # Make pyds.NvDsFrameMeta.cast and NvDsObjectMeta.cast return the data directly
        pyds.NvDsFrameMeta.cast.side_effect = lambda data: data
        pyds.NvDsObjectMeta.cast.side_effect = lambda data: data

        pipeline = DeepStreamPipeline(config)
        pipeline._sources = {0: {"connected": True}}

        info = MagicMock()
        info.get_buffer.return_value = MagicMock()

        return pipeline, gst, pyds, info

    def test_returns_ok_on_success(self, default_config):
        """Callback returns Gst.PadProbeReturn.OK on successful processing."""
        obj_meta = self._make_obj_meta()
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                result = pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert result == 0  # Gst.PadProbeReturn.OK

    def test_returns_ok_on_null_buffer(self, default_config):
        """Callback returns OK even when buffer is None."""
        gst = MagicMock()
        gst.PadProbeReturn.OK = 0
        pyds = MagicMock()

        pipeline = DeepStreamPipeline(default_config)
        info = MagicMock()
        info.get_buffer.return_value = None

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            result = pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert result == 0

    def test_returns_ok_on_null_batch_meta(self, default_config):
        """Callback returns OK when batch metadata is None."""
        gst = MagicMock()
        gst.PadProbeReturn.OK = 0
        pyds = MagicMock()
        pyds.gst_buffer_get_nvds_batch_meta.return_value = None

        pipeline = DeepStreamPipeline(default_config)
        info = MagicMock()
        info.get_buffer.return_value = MagicMock()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            result = pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert result == 0

    def test_extracts_detection_with_correct_bbox(self, default_config):
        """Callback converts left/top/width/height to x1/y1/x2/y2."""
        obj_meta = self._make_obj_meta(left=10, top=20, width=100, height=50)
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert len(received_data) == 1
        detections = received_data[0]["detections"]
        assert len(detections) == 1
        assert detections[0].bbox == (10, 20, 110, 70)  # x1, y1, x1+w, y1+h

    def test_extracts_class_id_and_confidence(self, default_config):
        """Callback extracts class_id and confidence from object meta."""
        obj_meta = self._make_obj_meta(class_id=2, confidence=0.85)
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        detection = received_data[0]["detections"][0]
        assert detection.class_id == 2
        assert detection.confidence == 0.85

    def test_extracts_track_id(self, default_config):
        """Callback extracts tracker ID from object meta."""
        obj_meta = self._make_obj_meta(object_id=42)
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        detection = received_data[0]["detections"][0]
        assert detection.track_id == 42

    def test_track_id_none_for_untracked(self, default_config):
        """Callback sets track_id=None when object_id is 0xFFFFFFFFFFFFFFFF."""
        obj_meta = self._make_obj_meta(object_id=0xFFFFFFFFFFFFFFFF)
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        detection = received_data[0]["detections"][0]
        assert detection.track_id is None

    def test_extracts_class_name(self, default_config):
        """Callback extracts class name from obj_label."""
        obj_meta = self._make_obj_meta(obj_label="car")
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        detection = received_data[0]["detections"][0]
        assert detection.class_name == "car"

    def test_invokes_all_registered_callbacks(self, default_config):
        """Callback invokes all registered probe callbacks."""
        obj_meta = self._make_obj_meta()
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        cb1_data = []
        cb2_data = []
        pipeline.register_probe(lambda data: cb1_data.append(data))
        pipeline.register_probe(lambda data: cb2_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert len(cb1_data) == 1
        assert len(cb2_data) == 1

    def test_callback_error_does_not_crash_pipeline(self, default_config):
        """A failing callback does not prevent other callbacks from running."""
        obj_meta = self._make_obj_meta()
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        cb2_data = []
        pipeline.register_probe(lambda data: (_ for _ in ()).throw(ValueError("boom")))
        pipeline.register_probe(lambda data: cb2_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                result = pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert result == 0  # Still returns OK
        assert len(cb2_data) == 1

    def test_updates_total_frames_processed(self, default_config):
        """Callback increments total_frames_processed stat."""
        obj_meta = self._make_obj_meta()
        frame_meta = self._make_frame_meta(obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert pipeline._stats.total_frames_processed == 1

    def test_handles_multiple_objects_in_frame(self, default_config):
        """Callback extracts all objects from a single frame."""
        obj1 = self._make_obj_meta(class_id=0, obj_label="person", object_id=1)
        obj2 = self._make_obj_meta(class_id=1, obj_label="car", object_id=2)
        obj3 = self._make_obj_meta(class_id=2, obj_label="bicycle", object_id=3)
        frame_meta = self._make_frame_meta(obj_metas=[obj1, obj2, obj3])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert len(received_data) == 1
        assert len(received_data[0]["detections"]) == 3

    def test_handles_multiple_frames_in_batch(self, default_config):
        """Callback processes all frames in a batch."""
        obj1 = self._make_obj_meta(obj_label="person")
        obj2 = self._make_obj_meta(obj_label="car")
        frame1 = self._make_frame_meta(source_id=0, obj_metas=[obj1])
        frame2 = self._make_frame_meta(source_id=1, obj_metas=[obj2])
        batch_meta = self._make_batch_meta([frame1, frame2])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)
        pipeline._sources = {0: {"connected": True}, 1: {"connected": True}}

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert len(received_data) == 2
        assert pipeline._stats.total_frames_processed == 2

    def test_batch_data_contains_expected_keys(self, default_config):
        """Callback passes batch_data dict with required keys to callbacks."""
        obj_meta = self._make_obj_meta()
        frame_meta = self._make_frame_meta(source_id=3, frame_num=42, obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)
        pipeline._sources = {3: {"connected": True}}

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1234.5
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        batch_data = received_data[0]
        assert "detections" in batch_data
        assert "frame_number" in batch_data
        assert "source_id" in batch_data
        assert "timestamp" in batch_data
        assert batch_data["frame_number"] == 42
        assert batch_data["source_id"] == 3
        assert batch_data["timestamp"] == 1234.5

    def test_detection_camera_id_matches_source_id(self, default_config):
        """Detection.camera_id is set to the frame's source_id."""
        obj_meta = self._make_obj_meta()
        frame_meta = self._make_frame_meta(source_id=5, obj_metas=[obj_meta])
        batch_meta = self._make_batch_meta([frame_meta])

        pipeline, gst, pyds, info = self._setup_pipeline_with_mocks(default_config, batch_meta)
        pipeline._sources = {5: {"connected": True}}

        received_data = []
        pipeline.register_probe(lambda data: received_data.append(data))

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            with patch("metropolis.deepstream_pipeline.time") as mock_time:
                mock_time.time.return_value = 1000.0
                pipeline._analytics_probe_callback(MagicMock(), info, None)

        detection = received_data[0]["detections"][0]
        assert detection.camera_id == 5

    def test_graceful_error_handling(self, default_config):
        """Callback handles exceptions gracefully and returns OK."""
        gst = MagicMock()
        gst.PadProbeReturn.OK = 0
        pyds = MagicMock()
        # Simulate an error during batch meta extraction
        pyds.gst_buffer_get_nvds_batch_meta.side_effect = RuntimeError("GPU error")

        pipeline = DeepStreamPipeline(default_config)
        info = MagicMock()
        info.get_buffer.return_value = MagicMock()

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, pyds)):
            result = pipeline._analytics_probe_callback(MagicMock(), info, None)

        assert result == 0  # Still returns OK despite error


# ---------------------------------------------------------------------------
# Tests: _handle_source_error() and _reconnect_source()
# ---------------------------------------------------------------------------

from metropolis.deepstream_pipeline import MAX_RETRIES, RECONNECT_INTERVAL_SECS


class TestHandleSourceError:
    """Tests for _handle_source_error() fault tolerance logic."""

    def test_increments_retry_count(self, default_config):
        """_handle_source_error increments retry_count for the source."""
        pipeline = DeepStreamPipeline(default_config)
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": True,
            "retry_count": 0,
        }

        with patch("threading.Timer") as mock_timer:
            mock_timer.return_value = MagicMock()
            pipeline._handle_source_error(0, RuntimeError("Connection lost"))

        assert pipeline._sources[0]["retry_count"] == 1

    def test_schedules_reconnection_on_first_retry(self, default_config):
        """_handle_source_error starts a Timer for reconnection on first failure."""
        pipeline = DeepStreamPipeline(default_config)
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": True,
            "retry_count": 0,
        }

        with patch("metropolis.deepstream_pipeline.threading.Timer") as mock_timer:
            timer_instance = MagicMock()
            mock_timer.return_value = timer_instance
            pipeline._handle_source_error(0, RuntimeError("Connection lost"))

        mock_timer.assert_called_once_with(
            RECONNECT_INTERVAL_SECS,
            pipeline._reconnect_source,
            args=(0,),
        )
        timer_instance.start.assert_called_once()

    def test_timer_is_daemon(self, default_config):
        """Reconnection timer thread is set as daemon."""
        pipeline = DeepStreamPipeline(default_config)
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": True,
            "retry_count": 0,
        }

        with patch("metropolis.deepstream_pipeline.threading.Timer") as mock_timer:
            timer_instance = MagicMock()
            mock_timer.return_value = timer_instance
            pipeline._handle_source_error(0, RuntimeError("Connection lost"))

        assert timer_instance.daemon is True

    def test_marks_source_disconnected_after_max_retries(self, default_config):
        """After MAX_RETRIES, source is marked as permanently disconnected."""
        pipeline = DeepStreamPipeline(default_config)
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": True,
            "retry_count": MAX_RETRIES,  # Already at max
        }

        # This call will increment to MAX_RETRIES + 1, exceeding the limit
        pipeline._handle_source_error(0, RuntimeError("Connection lost"))

        assert pipeline._sources[0]["connected"] is False
        assert pipeline._sources[0]["retry_count"] == MAX_RETRIES + 1

    def test_does_not_schedule_timer_after_max_retries(self, default_config):
        """No reconnection timer is started after MAX_RETRIES exhausted."""
        pipeline = DeepStreamPipeline(default_config)
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": True,
            "retry_count": MAX_RETRIES,
        }

        with patch("metropolis.deepstream_pipeline.threading.Timer") as mock_timer:
            pipeline._handle_source_error(0, RuntimeError("Connection lost"))

        mock_timer.assert_not_called()

    def test_handles_unknown_source_id_gracefully(self, default_config):
        """_handle_source_error returns without error for unknown source_id."""
        pipeline = DeepStreamPipeline(default_config)
        # No sources registered — should not raise
        pipeline._handle_source_error(99, RuntimeError("Unknown source"))

    def test_schedules_reconnection_up_to_max_retries(self, default_config):
        """Reconnection is scheduled for each retry up to MAX_RETRIES."""
        pipeline = DeepStreamPipeline(default_config)
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": True,
            "retry_count": MAX_RETRIES - 1,  # One retry left
        }

        with patch("metropolis.deepstream_pipeline.threading.Timer") as mock_timer:
            timer_instance = MagicMock()
            mock_timer.return_value = timer_instance
            pipeline._handle_source_error(0, RuntimeError("Connection lost"))

        # Should still schedule since retry_count becomes MAX_RETRIES (== MAX_RETRIES)
        mock_timer.assert_called_once()
        timer_instance.start.assert_called_once()

    def test_constants_have_correct_values(self):
        """Module constants MAX_RETRIES and RECONNECT_INTERVAL_SECS are correct."""
        assert MAX_RETRIES == 3
        assert RECONNECT_INTERVAL_SECS == 10


class TestReconnectSource:
    """Tests for _reconnect_source() helper method."""

    def test_successful_reconnection_resets_retry_count(self, default_config):
        """Successful reconnection resets retry_count to 0."""
        gst = MagicMock()
        gst.PadLinkReturn.OK = 0
        source_bin = MagicMock()
        srcpad = MagicMock()
        srcpad.link.return_value = 0  # Gst.PadLinkReturn.OK
        source_bin.get_static_pad.return_value = srcpad

        pipeline = DeepStreamPipeline(default_config)
        pipeline._pipeline = MagicMock()
        pipeline._streammux = MagicMock()
        pipeline._streammux.get_request_pad.return_value = MagicMock()
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": False,
            "retry_count": 2,
        }

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch.object(pipeline, "_create_source_bin", return_value=source_bin):
                pipeline._reconnect_source(0)

        assert pipeline._sources[0]["retry_count"] == 0
        assert pipeline._sources[0]["connected"] is True

    def test_successful_reconnection_updates_element(self, default_config):
        """Successful reconnection updates the element reference."""
        gst = MagicMock()
        gst.PadLinkReturn.OK = 0
        source_bin = MagicMock()
        srcpad = MagicMock()
        srcpad.link.return_value = 0
        source_bin.get_static_pad.return_value = srcpad

        pipeline = DeepStreamPipeline(default_config)
        pipeline._pipeline = MagicMock()
        pipeline._streammux = MagicMock()
        pipeline._streammux.get_request_pad.return_value = MagicMock()
        old_element = MagicMock()
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": old_element,
            "connected": False,
            "retry_count": 1,
        }

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch.object(pipeline, "_create_source_bin", return_value=source_bin):
                pipeline._reconnect_source(0)

        assert pipeline._sources[0]["element"] is source_bin

    def test_removes_old_element_before_reconnection(self, default_config):
        """Old source bin is set to NULL and removed from pipeline."""
        gst = MagicMock()
        gst.State.NULL = 1
        gst.PadLinkReturn.OK = 0
        source_bin = MagicMock()
        srcpad = MagicMock()
        srcpad.link.return_value = 0
        source_bin.get_static_pad.return_value = srcpad

        pipeline = DeepStreamPipeline(default_config)
        mock_pipeline = MagicMock()
        pipeline._pipeline = mock_pipeline
        pipeline._streammux = MagicMock()
        pipeline._streammux.get_request_pad.return_value = MagicMock()
        old_element = MagicMock()
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": old_element,
            "connected": False,
            "retry_count": 1,
        }

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch.object(pipeline, "_create_source_bin", return_value=source_bin):
                pipeline._reconnect_source(0)

        old_element.set_state.assert_called_once_with(1)  # Gst.State.NULL
        mock_pipeline.remove.assert_called_once_with(old_element)

    def test_failed_reconnection_triggers_handle_source_error(self, default_config):
        """Failed reconnection calls _handle_source_error to retry."""
        gst = MagicMock()
        gst.State.NULL = 1

        pipeline = DeepStreamPipeline(default_config)
        pipeline._pipeline = MagicMock()
        pipeline._streammux = MagicMock()
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": False,
            "retry_count": 1,
        }

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch.object(
                pipeline, "_create_source_bin", side_effect=RuntimeError("Cannot create bin")
            ):
                with patch.object(pipeline, "_handle_source_error") as mock_handle:
                    pipeline._reconnect_source(0)

        mock_handle.assert_called_once()
        args = mock_handle.call_args[0]
        assert args[0] == 0
        assert isinstance(args[1], RuntimeError)

    def test_handles_unknown_source_id(self, default_config):
        """_reconnect_source returns without error for unknown source_id."""
        pipeline = DeepStreamPipeline(default_config)
        # No sources registered — should not raise
        pipeline._reconnect_source(99)

    def test_sets_new_source_bin_to_playing(self, default_config):
        """New source bin is set to PLAYING state after reconnection."""
        gst = MagicMock()
        gst.State.NULL = 1
        gst.State.PLAYING = 4
        gst.PadLinkReturn.OK = 0
        source_bin = MagicMock()
        srcpad = MagicMock()
        srcpad.link.return_value = 0
        source_bin.get_static_pad.return_value = srcpad

        pipeline = DeepStreamPipeline(default_config)
        pipeline._pipeline = MagicMock()
        pipeline._streammux = MagicMock()
        pipeline._streammux.get_request_pad.return_value = MagicMock()
        pipeline._sources[0] = {
            "uri": "rtsp://camera:554/stream",
            "source_type": "rtsp",
            "element": MagicMock(),
            "connected": False,
            "retry_count": 1,
        }

        with patch.object(pipeline, "_import_gstreamer", return_value=(gst, MagicMock())):
            with patch.object(pipeline, "_create_source_bin", return_value=source_bin):
                pipeline._reconnect_source(0)

        source_bin.set_state.assert_called_once_with(4)  # Gst.State.PLAYING
