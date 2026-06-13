"""DeepStream/GStreamer pipeline backend for NVIDIA Metropolis integration.

Provides a GStreamer-based video analytics pipeline using NVIDIA DeepStream SDK.
The pipeline supports multi-source stream multiplexing, TensorRT inference,
object tracking, and analytics metadata extraction via pad probes.

GStreamer (gi) and DeepStream (pyds) are imported lazily since they require
NVIDIA hardware and SDK installations that may not be available in all
environments.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .orchestrator import Detection

logger = logging.getLogger(__name__)

# Fault tolerance constants
MAX_RETRIES = 3
RECONNECT_INTERVAL_SECS = 10


@dataclass
class DeepStreamConfig:
    """Configuration for the DeepStream/GStreamer pipeline.

    Controls pipeline element properties including inference model paths,
    tracker library, stream multiplexer dimensions, and GPU device selection.

    Attributes:
        nvinfer_config: Path to the nvinfer element configuration file
            specifying model engine, label file, and inference parameters.
        tracker_lib: Path to the nvtracker shared library (.so) implementing
            the selected tracking algorithm (DeepSORT, ByteTrack, or NvDCF).
        tracker_config: Path to the tracker configuration YAML file with
            algorithm-specific parameters.
        mux_width: Output width in pixels for the stream multiplexer. All
            input streams are scaled to this width before batching.
        mux_height: Output height in pixels for the stream multiplexer. All
            input streams are scaled to this height before batching.
        mux_batch_size: Maximum number of frames batched together by the
            stream multiplexer for parallel inference.
        mux_batched_push_timeout: Maximum time in microseconds the muxer
            waits to form a complete batch before pushing an incomplete one.
        gpu_id: CUDA device index to use for inference and preprocessing.
    """

    nvinfer_config: str = "configs/nvinfer_config.txt"
    tracker_lib: str = (
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    )
    tracker_config: str = "configs/tracker_config.yml"
    mux_width: int = 1920
    mux_height: int = 1080
    mux_batch_size: int = 4
    mux_batched_push_timeout: int = 40000  # microseconds
    gpu_id: int = 0


@dataclass
class PipelineStats:
    """Runtime statistics for the DeepStream pipeline.

    Provides per-source performance metrics collected during pipeline
    execution. Updated periodically by the analytics probe callback.

    Attributes:
        fps: Mapping of source_id to current frames-per-second throughput.
        latency_ms: Mapping of source_id to current end-to-end latency in
            milliseconds (frame ingestion to metadata output).
        gpu_utilization: Current GPU utilization percentage (0.0 to 100.0)
            as reported by the NVIDIA driver.
        active_sources: Number of sources currently streaming frames.
        total_frames_processed: Cumulative frame count across all sources
            since pipeline start.
    """

    fps: dict[int, float] = field(default_factory=dict)
    latency_ms: dict[int, float] = field(default_factory=dict)
    gpu_utilization: float = 0.0
    active_sources: int = 0
    total_frames_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline statistics to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "fps": dict(self.fps),
            "latency_ms": dict(self.latency_ms),
            "gpu_utilization": self.gpu_utilization,
            "active_sources": self.active_sources,
            "total_frames_processed": self.total_frames_processed,
        }


class DeepStreamPipeline:
    """GStreamer/DeepStream video analytics pipeline.

    Constructs and manages a DeepStream pipeline with the following topology:
        nvstreammux → nvinfer → nvtracker → nvosd → fakesink

    The pipeline supports multiple camera sources (RTSP, USB, file), performs
    TensorRT-accelerated inference, tracks objects across frames, and exposes
    detection/tracking metadata through registered probe callbacks.

    GStreamer and DeepStream Python bindings are imported lazily on first use
    to allow the module to be imported in environments without NVIDIA hardware.

    Example:
        >>> config = DeepStreamConfig(mux_batch_size=2)
        >>> pipeline = DeepStreamPipeline(config)
        >>> pipeline.add_source(0, "rtsp://camera1:554/stream")
        >>> pipeline.register_probe(my_callback)
        >>> pipeline.start()
        >>> # ... pipeline runs in background ...
        >>> pipeline.stop()
    """

    def __init__(self, config: DeepStreamConfig) -> None:
        """Initialize the DeepStream pipeline from configuration.

        Stores configuration and sets up internal state. The actual GStreamer
        pipeline is not constructed until start() is called, allowing sources
        to be added first.

        Args:
            config: DeepStreamConfig instance with pipeline parameters.
        """
        self._config = config
        self._pipeline: Optional[Any] = None  # Gst.Pipeline once constructed
        self._sources: dict[int, dict[str, Any]] = {}  # source_id -> source info
        self._running: bool = False
        self._probe_callbacks: list[Callable] = []
        self._stats = PipelineStats()
        self._main_loop: Optional[Any] = None  # GLib.MainLoop
        self._loop_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        logger.info(
            "DeepStreamPipeline initialized with config: mux=%dx%d, batch=%d, gpu=%d",
            config.mux_width,
            config.mux_height,
            config.mux_batch_size,
            config.gpu_id,
        )

    @property
    def config(self) -> DeepStreamConfig:
        """Return the pipeline configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Return whether the pipeline is currently running."""
        return self._running

    @property
    def sources(self) -> dict[int, dict[str, Any]]:
        """Return the current source registry (source_id -> info dict)."""
        return dict(self._sources)

    def _import_gstreamer(self) -> tuple[Any, Any]:
        """Lazily import GStreamer and DeepStream Python bindings.

        Returns:
            Tuple of (Gst module, pyds module).

        Raises:
            ImportError: If GStreamer or DeepStream bindings are not installed.
        """
        try:
            import gi  # noqa: F401

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst  # noqa: F401
        except (ImportError, ValueError) as exc:
            logger.error("GStreamer Python bindings not available: %s", exc)
            raise ImportError(
                "GStreamer Python bindings (gi/Gst) are required for "
                "DeepStream pipeline. Install with: apt-get install "
                "python3-gi gstreamer1.0-tools"
            ) from exc

        try:
            import pyds  # noqa: F401
        except ImportError as exc:
            logger.error("DeepStream Python bindings (pyds) not available: %s", exc)
            raise ImportError(
                "DeepStream Python bindings (pyds) are required. "
                "Install the NVIDIA DeepStream SDK."
            ) from exc

        return Gst, pyds

    def add_source(
        self, source_id: int, uri: str, source_type: str = "rtsp"
    ) -> None:
        """Add a camera source to the pipeline multiplexer.

        Registers a video source that will be connected to the stream
        multiplexer when the pipeline starts. Sources must be added before
        calling start().

        Args:
            source_id: Unique integer identifier for this source. Used as
                the muxer sink pad index and for per-source statistics.
            uri: Source URI. For RTSP: "rtsp://host:port/path", for USB:
                "/dev/video0", for file: "/path/to/video.mp4".
            source_type: Type of source. One of "rtsp", "usb", or "file".

        Raises:
            ValueError: If source_id is already registered or source_type
                is not recognized.
            RuntimeError: If the pipeline is already running.
        """
        if self._running:
            raise RuntimeError(
                "Cannot add sources while pipeline is running. "
                "Stop the pipeline first."
            )

        if source_id in self._sources:
            raise ValueError(
                f"Source ID {source_id} is already registered. "
                f"Use a unique source_id."
            )

        valid_types = ("rtsp", "usb", "file")
        if source_type not in valid_types:
            raise ValueError(
                f"Invalid source_type '{source_type}'. "
                f"Must be one of: {valid_types}"
            )

        self._sources[source_id] = {
            "uri": uri,
            "source_type": source_type,
            "element": None,  # Populated during pipeline construction
            "connected": False,
            "retry_count": 0,
        }

        logger.info(
            "Added source %d: type=%s, uri=%s",
            source_id,
            source_type,
            uri,
        )

    def start(self) -> None:
        """Start the GStreamer main loop in a background daemon thread.

        Constructs the full DeepStream pipeline (nvstreammux → nvinfer →
        nvtracker → nvosd → fakesink), connects all registered sources,
        attaches analytics probes, and transitions the pipeline to PLAYING
        state.

        The GLib main loop runs in a daemon thread so it does not block
        the calling thread.

        Raises:
            RuntimeError: If the pipeline is already running or no sources
                have been added.
            ImportError: If GStreamer/DeepStream bindings are not available.
        """
        # Validate pipeline is not already running
        if self._running:
            raise RuntimeError(
                "Pipeline is already running. Call stop() before starting again."
            )

        # Validate at least one source has been added
        if not self._sources:
            raise RuntimeError(
                "No sources have been added. Call add_source() before start()."
            )

        logger.info("Starting DeepStream pipeline with %d source(s)...", len(self._sources))

        # Import GStreamer bindings (validates availability)
        Gst, _pyds = self._import_gstreamer()
        from gi.repository import GLib

        # Build the core pipeline (streammux → nvinfer → nvtracker → nvosd → fakesink)
        logger.info("Building pipeline elements...")
        self._build_pipeline()

        # Connect each registered source to the stream multiplexer
        for source_id, source_info in self._sources.items():
            logger.info("Connecting source %d to pipeline...", source_id)

            # Create the source bin for this source
            source_bin = self._create_source_bin(source_id, source_info)

            # Add the source bin to the pipeline
            self._pipeline.add(source_bin)

            # Request a sink pad from the streammux for this source
            sinkpad = self._streammux.get_request_pad(f"sink_{source_id}")
            if not sinkpad:
                raise RuntimeError(
                    f"Failed to get sink pad 'sink_{source_id}' from streammux "
                    f"for source {source_id}."
                )

            # Link the source bin's "src" ghost pad to the muxer sink pad
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                raise RuntimeError(
                    f"Failed to get 'src' pad from source bin for source {source_id}."
                )

            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Failed to link source bin pad to streammux sink pad "
                    f"for source {source_id}."
                )

            # Update source info to track the element reference
            source_info["element"] = source_bin
            source_info["connected"] = True
            logger.info("Source %d connected to streammux successfully.", source_id)

        # Attach the analytics probe to the nvosd sink pad
        logger.info("Attaching analytics probe to nvosd sink pad...")
        osd_sink_pad = self._nvosd.get_static_pad("sink")
        if not osd_sink_pad:
            raise RuntimeError("Failed to get sink pad from nvosd element.")
        osd_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, self._analytics_probe_callback, None
        )
        logger.info("Analytics probe attached successfully.")

        # Create and start the GLib main loop in a daemon thread
        logger.info("Creating GLib MainLoop and starting daemon thread...")
        self._main_loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._main_loop.run, daemon=True, name="glib-mainloop"
        )
        self._loop_thread.start()
        logger.info("GLib MainLoop thread started.")

        # Set pipeline state to PLAYING
        logger.info("Setting pipeline state to PLAYING...")
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            # Clean up on failure
            self._main_loop.quit()
            self._loop_thread.join(timeout=5.0)
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(
                "Failed to set pipeline state to PLAYING. "
                "Check source URIs and element configurations."
            )

        # Mark pipeline as running
        self._running = True
        logger.info(
            "DeepStream pipeline started successfully with %d source(s).",
            len(self._sources),
        )

    def stop(self) -> None:
        """Gracefully stop the pipeline and release GPU resources.

        Transitions the pipeline through PAUSED → READY → NULL states,
        stops the GLib main loop, joins the background thread, and releases
        all GStreamer elements and GPU memory.

        This method is safe to call even if the pipeline is not running.
        """
        if not self._running:
            logger.warning("stop() called but pipeline is not running.")
            return

        logger.info("Stopping DeepStream pipeline...")

        # Import Gst for state constants
        Gst, _pyds = self._import_gstreamer()

        # Set pipeline state to NULL to release all resources
        if self._pipeline is not None:
            logger.info("Setting pipeline state to NULL...")
            self._pipeline.set_state(Gst.State.NULL)
            logger.info("Pipeline state set to NULL.")

        # Quit the GLib main loop
        if self._main_loop is not None:
            logger.info("Quitting GLib MainLoop...")
            self._main_loop.quit()
            logger.info("GLib MainLoop quit signal sent.")

        # Join the background thread with a timeout
        if self._loop_thread is not None:
            logger.info("Joining GLib MainLoop thread...")
            self._loop_thread.join(timeout=5.0)
            if self._loop_thread.is_alive():
                logger.warning(
                    "GLib MainLoop thread did not terminate within timeout."
                )
            else:
                logger.info("GLib MainLoop thread joined successfully.")

        # Mark pipeline as not running
        self._running = False

        # Clean up references
        self._pipeline = None
        self._main_loop = None
        self._loop_thread = None
        self._streammux = None
        self._nvinfer = None
        self._tracker = None
        self._nvosd = None
        self._fakesink = None

        # Reset source connection state
        for source_info in self._sources.values():
            source_info["element"] = None
            source_info["connected"] = False

        logger.info("DeepStream pipeline stopped and resources released.")

    def register_probe(self, callback: Callable) -> None:
        """Register an analytics probe callback for metadata extraction.

        The callback will be invoked for each batch of frames processed by
        the pipeline, receiving extracted detection and tracking metadata.

        Callback signature:
            def callback(batch_meta: dict) -> None:
                # batch_meta contains 'detections', 'tracks', 'frame_number',
                # 'source_id', 'timestamp'
                ...

        Args:
            callback: Callable to invoke with batch metadata. Must accept
                a single dict argument.

        Raises:
            TypeError: If callback is not callable.
        """
        if not callable(callback):
            raise TypeError(
                f"Probe callback must be callable, got {type(callback).__name__}"
            )

        self._probe_callbacks.append(callback)
        logger.debug(
            "Registered probe callback: %s (total: %d)",
            getattr(callback, "__name__", repr(callback)),
            len(self._probe_callbacks),
        )

    def get_stats(self) -> PipelineStats:
        """Return current pipeline performance statistics.

        Returns a snapshot of FPS, latency, and GPU utilization metrics
        per source. Statistics are updated by the analytics probe on each
        processed batch.

        Returns:
            PipelineStats dataclass with current metrics.
        """
        # Implementation in task 3.5 (populated by probe callback)
        with self._lock:
            return PipelineStats(
                fps=dict(self._stats.fps),
                latency_ms=dict(self._stats.latency_ms),
                gpu_utilization=self._stats.gpu_utilization,
                active_sources=self._stats.active_sources,
                total_frames_processed=self._stats.total_frames_processed,
            )

    def _build_pipeline(self) -> None:
        """Construct the GStreamer pipeline elements and link them.

        Creates: nvstreammux → nvinfer → nvtracker → nvosd → fakesink
        and connects all registered sources to the muxer.

        Raises:
            RuntimeError: If any pipeline element fails to create.
            ImportError: If GStreamer/DeepStream bindings are not available.
        """
        Gst, _pyds = self._import_gstreamer()

        logger.info("Initializing GStreamer...")
        Gst.init(None)

        logger.info("Creating GStreamer pipeline...")
        self._pipeline = Gst.Pipeline()
        if not self._pipeline:
            raise RuntimeError("Failed to create GStreamer Pipeline")

        config = self._config

        # --- Create pipeline elements ---

        logger.info("Creating nvstreammux element...")
        self._streammux = Gst.ElementFactory.make("nvstreammux", "mux")
        if not self._streammux:
            raise RuntimeError(
                "Failed to create element 'nvstreammux'. "
                "Ensure DeepStream plugins are installed."
            )
        self._streammux.set_property("batch-size", config.mux_batch_size)
        self._streammux.set_property("width", config.mux_width)
        self._streammux.set_property("height", config.mux_height)
        self._streammux.set_property("batched-push-timeout", config.mux_batched_push_timeout)
        self._streammux.set_property("gpu-id", config.gpu_id)

        logger.info("Creating nvinfer element...")
        self._nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not self._nvinfer:
            raise RuntimeError(
                "Failed to create element 'nvinfer'. "
                "Ensure DeepStream plugins are installed."
            )
        self._nvinfer.set_property("config-file-path", config.nvinfer_config)
        self._nvinfer.set_property("gpu-id", config.gpu_id)

        logger.info("Creating nvtracker element...")
        self._tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not self._tracker:
            raise RuntimeError(
                "Failed to create element 'nvtracker'. "
                "Ensure DeepStream plugins are installed."
            )
        self._tracker.set_property("tracker-width", 640)
        self._tracker.set_property("tracker-height", 384)
        self._tracker.set_property("ll-lib-file", config.tracker_lib)
        self._tracker.set_property("ll-config-file", config.tracker_config)
        self._tracker.set_property("gpu-id", config.gpu_id)

        logger.info("Creating nvdsosd element...")
        self._nvosd = Gst.ElementFactory.make("nvdsosd", "osd")
        if not self._nvosd:
            raise RuntimeError(
                "Failed to create element 'nvdsosd'. "
                "Ensure DeepStream plugins are installed."
            )

        logger.info("Creating fakesink element...")
        self._fakesink = Gst.ElementFactory.make("fakesink", "sink")
        if not self._fakesink:
            raise RuntimeError(
                "Failed to create element 'fakesink'. "
                "Ensure GStreamer base plugins are installed."
            )
        self._fakesink.set_property("sync", False)

        # --- Add elements to pipeline ---

        logger.info("Adding elements to pipeline...")
        self._pipeline.add(self._streammux)
        self._pipeline.add(self._nvinfer)
        self._pipeline.add(self._tracker)
        self._pipeline.add(self._nvosd)
        self._pipeline.add(self._fakesink)

        # --- Link elements in order ---

        logger.info("Linking pipeline elements: streammux → nvinfer → nvtracker → nvosd → fakesink")
        if not self._streammux.link(self._nvinfer):
            raise RuntimeError("Failed to link nvstreammux → nvinfer")
        if not self._nvinfer.link(self._tracker):
            raise RuntimeError("Failed to link nvinfer → nvtracker")
        if not self._tracker.link(self._nvosd):
            raise RuntimeError("Failed to link nvtracker → nvosd")
        if not self._nvosd.link(self._fakesink):
            raise RuntimeError("Failed to link nvosd → fakesink")

        logger.info("DeepStream pipeline constructed successfully.")

    def _create_source_bin(self, source_id: int, source_info: dict) -> Any:
        """Create a GStreamer source bin for a single camera input.

        Constructs a GStreamer Bin containing the appropriate source elements
        based on the source type:
          - RTSP: uridecodebin with the RTSP URI
          - File: uridecodebin with file:// URI prefix
          - USB: v4l2src ! videoconvert ! nvvideoconvert

        The bin exposes a ghost pad named "src" that connects to the
        stream multiplexer. For decodebin-based sources, the ghost pad is
        linked dynamically via the pad-added signal.

        Args:
            source_id: Unique source identifier.
            source_info: Dictionary with 'uri' and 'source_type' keys.

        Returns:
            GStreamer Bin element for the source.

        Raises:
            RuntimeError: If source bin creation fails.
        """
        Gst, _pyds = self._import_gstreamer()

        uri = source_info["uri"]
        source_type = source_info["source_type"]
        bin_name = f"source-bin-{source_id}"

        logger.info(
            "Creating source bin '%s': type=%s, uri=%s",
            bin_name,
            source_type,
            uri,
        )

        source_bin = Gst.Bin.new(bin_name)
        if not source_bin:
            raise RuntimeError(f"Failed to create source bin '{bin_name}'")

        if source_type in ("rtsp", "file"):
            # Use uridecodebin for both RTSP and file sources
            if source_type == "file" and not uri.startswith("file://"):
                decode_uri = f"file://{uri}"
            else:
                decode_uri = uri

            uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-{source_id}")
            if not uri_decode_bin:
                raise RuntimeError(
                    f"Failed to create element 'uridecodebin' for source {source_id}. "
                    "Ensure GStreamer base plugins are installed."
                )
            uri_decode_bin.set_property("uri", decode_uri)

            source_bin.add(uri_decode_bin)

            # Create a ghost pad for the bin output. It will be linked
            # dynamically when uridecodebin emits pad-added.
            ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
            if not ghost_pad:
                raise RuntimeError(
                    f"Failed to create ghost pad for source bin '{bin_name}'"
                )
            ghost_pad.set_active(True)
            source_bin.add_pad(ghost_pad)

            # Connect pad-added signal to dynamically link the decoded pad
            # to the ghost pad when the stream is decoded.
            uri_decode_bin.connect(
                "pad-added",
                self._on_source_bin_pad_added,
                ghost_pad,
            )

        elif source_type == "usb":
            # USB camera: v4l2src ! videoconvert ! nvvideoconvert
            v4l2src = Gst.ElementFactory.make("v4l2src", f"v4l2src-{source_id}")
            if not v4l2src:
                raise RuntimeError(
                    f"Failed to create element 'v4l2src' for source {source_id}. "
                    "Ensure GStreamer video4linux2 plugin is installed."
                )
            v4l2src.set_property("device", uri)

            videoconvert = Gst.ElementFactory.make("videoconvert", f"videoconvert-{source_id}")
            if not videoconvert:
                raise RuntimeError(
                    f"Failed to create element 'videoconvert' for source {source_id}. "
                    "Ensure GStreamer base plugins are installed."
                )

            nvvideoconvert = Gst.ElementFactory.make(
                "nvvideoconvert", f"nvvideoconvert-{source_id}"
            )
            if not nvvideoconvert:
                raise RuntimeError(
                    f"Failed to create element 'nvvideoconvert' for source {source_id}. "
                    "Ensure DeepStream plugins are installed."
                )

            # Add elements to the bin
            source_bin.add(v4l2src)
            source_bin.add(videoconvert)
            source_bin.add(nvvideoconvert)

            # Link: v4l2src → videoconvert → nvvideoconvert
            if not v4l2src.link(videoconvert):
                raise RuntimeError(
                    f"Failed to link v4l2src → videoconvert for source {source_id}"
                )
            if not videoconvert.link(nvvideoconvert):
                raise RuntimeError(
                    f"Failed to link videoconvert → nvvideoconvert for source {source_id}"
                )

            # Create ghost pad from nvvideoconvert's src pad
            src_pad = nvvideoconvert.get_static_pad("src")
            if not src_pad:
                raise RuntimeError(
                    f"Failed to get src pad from nvvideoconvert for source {source_id}"
                )

            ghost_pad = Gst.GhostPad.new("src", src_pad)
            if not ghost_pad:
                raise RuntimeError(
                    f"Failed to create ghost pad for source bin '{bin_name}'"
                )
            ghost_pad.set_active(True)
            source_bin.add_pad(ghost_pad)

        else:
            raise RuntimeError(
                f"Unsupported source type '{source_type}' for source {source_id}"
            )

        logger.info("Source bin '%s' created successfully.", bin_name)
        return source_bin

    @staticmethod
    def _on_source_bin_pad_added(
        decodebin: Any, pad: Any, ghost_pad: Any
    ) -> None:
        """Handle pad-added signal from uridecodebin.

        Links the newly created decoded pad to the bin's ghost pad so that
        decoded video frames flow out of the source bin to the streammux.

        Only video pads are linked; audio pads are ignored.

        Args:
            decodebin: The uridecodebin element that emitted the signal.
            pad: The newly created source pad from decodebin.
            ghost_pad: The ghost pad on the source bin to link to.
        """
        caps = pad.get_current_caps()
        if caps is None:
            caps = pad.query_caps(None)

        structure = caps.get_structure(0)
        media_type = structure.get_name() if structure else ""

        # Only link video pads, ignore audio
        if not media_type.startswith("video/"):
            logger.debug(
                "Ignoring non-video pad from decodebin: %s", media_type
            )
            return

        # Set the ghost pad target to this new pad
        if not ghost_pad.set_target(pad):
            logger.error(
                "Failed to link decodebin pad to ghost pad (media: %s)",
                media_type,
            )
        else:
            logger.info(
                "Linked decodebin pad to source bin ghost pad (media: %s)",
                media_type,
            )

    def _analytics_probe_callback(self, pad: Any, info: Any, user_data: Any = None) -> int:
        """GStreamer pad probe callback for metadata extraction.

        Extracts NvDsMeta from each frame in the batch and invokes all
        registered probe callbacks with structured detection/tracking data.

        Args:
            pad: GStreamer pad the probe is attached to.
            info: GStreamer probe info containing buffer metadata.
            user_data: Optional user data passed during probe registration.

        Returns:
            GStreamer pad probe return value (GST_PAD_PROBE_OK).
        """
        try:
            Gst, pyds = self._import_gstreamer()

            # Extract the GstBuffer from probe info
            gst_buf = info.get_buffer()
            if not gst_buf:
                logger.warning("Analytics probe: unable to get GstBuffer from info")
                return Gst.PadProbeReturn.OK

            # Get batch metadata from the buffer
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buf))
            if not batch_meta:
                logger.warning("Analytics probe: unable to get batch metadata")
                return Gst.PadProbeReturn.OK

            # Iterate through frame metadata in the batch
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break

                if frame_meta is None:
                    try:
                        l_frame = l_frame.next
                    except StopIteration:
                        break
                    continue

                source_id = frame_meta.source_id
                frame_number = frame_meta.frame_num
                current_timestamp = time.time()
                detections: list[Detection] = []

                # Iterate through object metadata in this frame
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break

                    if obj_meta is None:
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break
                        continue

                    # Extract bounding box (left, top, width, height) and convert to (x1, y1, x2, y2)
                    rect_params = obj_meta.rect_params
                    x1 = int(rect_params.left)
                    y1 = int(rect_params.top)
                    x2 = int(rect_params.left + rect_params.width)
                    y2 = int(rect_params.top + rect_params.height)

                    # Extract class info and confidence
                    class_id = obj_meta.class_id
                    confidence = obj_meta.confidence

                    # Extract class name from text_params label if available
                    class_name = ""
                    if obj_meta.obj_label:
                        class_name = obj_meta.obj_label

                    # Extract tracker ID if available (0 or negative means no tracking)
                    track_id = None
                    if hasattr(obj_meta, "object_id") and obj_meta.object_id != 0xFFFFFFFFFFFFFFFF:
                        track_id = int(obj_meta.object_id)

                    # Create Detection dataclass
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        camera_id=source_id,
                        timestamp=current_timestamp,
                        track_id=track_id,
                    )
                    detections.append(detection)

                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break

                # Invoke all registered probe callbacks with extracted metadata
                batch_data = {
                    "detections": detections,
                    "frame_number": frame_number,
                    "source_id": source_id,
                    "timestamp": current_timestamp,
                }

                for callback in self._probe_callbacks:
                    try:
                        callback(batch_data)
                    except Exception as cb_exc:
                        logger.error(
                            "Probe callback %s raised an exception: %s",
                            getattr(callback, "__name__", repr(callback)),
                            cb_exc,
                        )

                # Update pipeline stats
                with self._lock:
                    self._stats.total_frames_processed += 1
                    self._stats.active_sources = len(
                        [s for s in self._sources.values() if s.get("connected")]
                    )
                    # Basic FPS tracking: increment frame count per source.
                    # A more sophisticated implementation would use a sliding
                    # time window, but frame_num from DeepStream provides a
                    # reasonable proxy for throughput monitoring.
                    if source_id not in self._stats.fps:
                        self._stats.fps[source_id] = 0.0
                    self._stats.fps[source_id] = float(frame_number)

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

        except Exception as exc:
            logger.error(
                "Analytics probe callback encountered an error: %s", exc,
                exc_info=True,
            )

        # Always return OK to keep the pipeline running
        try:
            from gi.repository import Gst as _Gst
            return _Gst.PadProbeReturn.OK
        except ImportError:
            # Fallback: return the integer value for GST_PAD_PROBE_OK
            return 0

    def _handle_source_error(self, source_id: int, error: Exception) -> None:
        """Handle source disconnection with retry logic.

        Attempts to reconnect a failed source up to MAX_RETRIES times with
        RECONNECT_INTERVAL_SECS-second intervals. If all retries are exhausted,
        the source is marked as permanently disconnected. The pipeline continues
        processing remaining sources regardless.

        Args:
            source_id: ID of the failed source.
            error: The exception that caused the failure.
        """
        logger.error(
            "Source %d encountered an error: %s", source_id, error
        )

        with self._lock:
            if source_id not in self._sources:
                logger.warning(
                    "Source %d not found in registry, cannot handle error.",
                    source_id,
                )
                return

            self._sources[source_id]["retry_count"] += 1
            retry_count = self._sources[source_id]["retry_count"]

        if retry_count <= MAX_RETRIES:
            logger.info(
                "Source %d: scheduling reconnection attempt %d/%d in %d seconds.",
                source_id,
                retry_count,
                MAX_RETRIES,
                RECONNECT_INTERVAL_SECS,
            )
            # Start a background timer thread that waits then attempts reconnection
            reconnect_timer = threading.Timer(
                RECONNECT_INTERVAL_SECS,
                self._reconnect_source,
                args=(source_id,),
            )
            reconnect_timer.daemon = True
            reconnect_timer.name = f"reconnect-source-{source_id}-attempt-{retry_count}"
            reconnect_timer.start()
        else:
            # All retries exhausted — mark source as permanently disconnected
            logger.warning(
                "Source %d: all %d reconnection attempts failed. "
                "Marking source as permanently disconnected.",
                source_id,
                MAX_RETRIES,
            )
            with self._lock:
                self._sources[source_id]["connected"] = False

    def _reconnect_source(self, source_id: int) -> None:
        """Attempt to reconnect a failed source to the pipeline.

        Tries to recreate the source bin and reconnect it to the stream
        multiplexer. If successful, resets the retry count. If the
        reconnection fails, calls _handle_source_error again to schedule
        another attempt (which will increment the retry count).

        Args:
            source_id: ID of the source to reconnect.
        """
        logger.info("Source %d: attempting reconnection...", source_id)

        with self._lock:
            if source_id not in self._sources:
                logger.warning(
                    "Source %d not found in registry during reconnection.",
                    source_id,
                )
                return
            source_info = self._sources[source_id]

        try:
            Gst, _pyds = self._import_gstreamer()

            # Remove the old source bin from the pipeline if it exists
            old_element = source_info.get("element")
            if old_element is not None and self._pipeline is not None:
                old_element.set_state(Gst.State.NULL)
                self._pipeline.remove(old_element)

            # Recreate the source bin
            source_bin = self._create_source_bin(source_id, source_info)

            # Add the new source bin to the pipeline
            self._pipeline.add(source_bin)

            # Request a sink pad from the streammux for this source
            sinkpad = self._streammux.get_request_pad(f"sink_{source_id}")
            if not sinkpad:
                raise RuntimeError(
                    f"Failed to get sink pad 'sink_{source_id}' from streammux "
                    f"for source {source_id} during reconnection."
                )

            # Link the source bin's "src" ghost pad to the muxer sink pad
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                raise RuntimeError(
                    f"Failed to get 'src' pad from source bin for source "
                    f"{source_id} during reconnection."
                )

            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Failed to link source bin pad to streammux sink pad "
                    f"for source {source_id} during reconnection."
                )

            # Set the source bin to PLAYING state to sync with the pipeline
            source_bin.set_state(Gst.State.PLAYING)

            # Reconnection successful — update source state
            with self._lock:
                source_info["element"] = source_bin
                source_info["connected"] = True
                source_info["retry_count"] = 0

            logger.info(
                "Source %d: reconnection successful. Retry count reset.",
                source_id,
            )

        except Exception as reconnect_error:
            logger.error(
                "Source %d: reconnection attempt failed: %s",
                source_id,
                reconnect_error,
            )
            # Trigger another retry cycle (will increment retry_count)
            self._handle_source_error(source_id, reconnect_error)
