"""Triton Inference Server gRPC client for NVIDIA Metropolis integration.

Provides a high-level interface to NVIDIA Triton Inference Server for batched
model inference via gRPC. Supports health checking, model readiness polling,
and metadata retrieval for the configured model ensemble.

The tritonclient.grpc dependency is lazily imported to allow the module to be
loaded in environments where the Triton client library is not installed.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from .orchestrator import Detection

logger = logging.getLogger(__name__)


class TritonClient:
    """Client for NVIDIA Triton Inference Server via gRPC.

    Manages a gRPC connection to a Triton server instance and provides
    methods for batched inference, health checking, and model metadata
    retrieval. The tritonclient.grpc library is lazily imported so this
    module can be safely imported even when the library is not installed.

    Attributes:
        server_url: The gRPC endpoint of the Triton server (host:port).
        model_name: Name of the model/ensemble to use for inference.
    """

    def __init__(
        self,
        server_url: str = "localhost:8001",
        model_name: str = "yolov8_ensemble",
    ) -> None:
        """Initialize the Triton gRPC client.

        Connects to the Triton Inference Server at the specified URL and
        prepares the client for inference requests against the named model.

        Args:
            server_url: Triton server gRPC endpoint (host:port).
            model_name: Name of the model or ensemble in the Triton model
                repository to target for inference requests.
        """
        self.server_url = server_url
        self.model_name = model_name
        self._connected: bool = False
        self._model_ready: bool = False
        self._client = None

        # Fallback state for local TensorRT inference
        self._fallback_enabled: bool = False
        self._using_fallback: bool = False
        self._fallback_engine_path: Optional[str] = None

        # Lazy import of tritonclient.grpc to handle environments
        # where the library is not installed.
        try:
            import tritonclient.grpc as grpcclient

            self._grpcclient = grpcclient
            self._client = grpcclient.InferenceServerClient(
                url=self.server_url,
                verbose=False,
            )
            self._connected = True
            logger.info(
                "Triton gRPC client connected to %s for model '%s'",
                self.server_url,
                self.model_name,
            )
        except ImportError:
            logger.warning(
                "tritonclient.grpc is not installed. Triton inference will "
                "not be available. Install with: pip install tritonclient[grpc]"
            )
            self._grpcclient = None
        except Exception as exc:
            logger.error(
                "Failed to connect to Triton server at %s: %s",
                self.server_url,
                exc,
            )
            self._grpcclient = None

    def infer(
        self,
        frames: list[np.ndarray],
        batch_size: Optional[int] = None,
    ) -> list[Detection]:
        """Send a batch of frames for inference on the Triton server.

        Preprocesses the input frames, assembles them into a batch, sends
        the inference request via gRPC, and parses the response into a list
        of Detection objects. If Triton is unavailable and a fallback engine
        is configured, automatically switches to local TensorRT inference.

        Args:
            frames: List of BGR numpy arrays (H, W, 3) representing video
                frames to run inference on.
            batch_size: Optional override for the batch size. If None, all
                frames are sent in a single batch.

        Returns:
            List of Detection objects extracted from the model output.

        Raises:
            RuntimeError: If the Triton client is not connected and no
                fallback engine is configured.
        """
        # If already using fallback, go directly to local inference
        if self._using_fallback:
            return self._infer_local_tensorrt(frames)

        # Check basic connectivity prerequisites
        if not self._connected or self._client is None or self._grpcclient is None:
            if self._fallback_enabled:
                logger.warning(
                    "Triton client is not connected. Switching to local "
                    "TensorRT fallback inference."
                )
                self._using_fallback = True
                return self._infer_local_tensorrt(frames)
            raise RuntimeError(
                "Triton client is not connected. Cannot perform inference."
            )

        all_detections: list[Detection] = []

        for frame in frames:
            try:
                detections = self._infer_single_frame(frame)
                all_detections.extend(detections)
            except Exception as exc:
                logger.error(
                    "Inference failed for frame (shape=%s): %s",
                    frame.shape if hasattr(frame, "shape") else "unknown",
                    exc,
                )
                # If fallback is enabled, switch to local TensorRT
                if self._fallback_enabled:
                    logger.warning(
                        "gRPC error detected. Switching to local TensorRT "
                        "fallback inference: %s",
                        exc,
                    )
                    self._using_fallback = True
                    return self._infer_local_tensorrt(frames)
                # Re-raise gRPC-specific errors as RuntimeError
                raise RuntimeError(
                    f"Triton inference failed: {exc}"
                ) from exc

        logger.debug(
            "Inference complete: %d frames processed, %d detections returned",
            len(frames),
            len(all_detections),
        )
        return all_detections

    def _infer_single_frame(self, frame: np.ndarray) -> list[Detection]:
        """Send a single frame to Triton and parse the response.

        The ensemble model accepts raw uint8 images in HWC format and handles
        preprocessing internally.

        Args:
            frame: A single BGR numpy array (H, W, 3) in uint8 format.

        Returns:
            List of Detection objects parsed from the model output.
        """
        grpcclient = self._grpcclient

        # Ensure frame is uint8 HWC format as expected by the ensemble
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Create InferInput tensor for the raw image
        # Shape is [H, W, C] for the ensemble's raw_image input
        shape = list(frame.shape)
        input_tensor = grpcclient.InferInput("raw_image", shape, "UINT8")
        input_tensor.set_data_from_numpy(frame)

        # Send inference request
        result = self._client.infer(
            model_name=self.model_name,
            inputs=[input_tensor],
        )

        # Parse the response
        # Output shape is [N, 6] where each row is [x1, y1, x2, y2, confidence, class_id]
        output = result.as_numpy("detections")

        detections = self._parse_detections(output)
        return detections

    def _parse_detections(
        self,
        output: np.ndarray,
        camera_id: int = 0,
    ) -> list[Detection]:
        """Parse the raw detection output tensor into Detection objects.

        Args:
            output: Numpy array of shape [N, 6] where each row contains
                [x1, y1, x2, y2, confidence, class_id].
            camera_id: Camera identifier to assign to detections.

        Returns:
            List of Detection objects.
        """
        detections: list[Detection] = []
        timestamp = time.time()

        if output is None or output.size == 0:
            return detections

        # Handle case where output might be 1D (single detection)
        if output.ndim == 1:
            output = output.reshape(1, -1)

        for row in output:
            if len(row) < 6:
                logger.warning("Skipping malformed detection row: %s", row)
                continue

            x1, y1, x2, y2, confidence, class_id = row[:6]

            detection = Detection(
                class_id=int(class_id),
                class_name=str(int(class_id)),
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                camera_id=camera_id,
                timestamp=timestamp,
            )
            detections.append(detection)

        return detections

    @property
    def is_using_fallback(self) -> bool:
        """Check whether the client is currently using local TensorRT fallback.

        Returns:
            True if inference is being routed to the local TensorRT engine
            instead of the Triton server.
        """
        return self._using_fallback

    def set_fallback_engine(self, engine_path: str) -> None:
        """Configure the local TensorRT engine path for fallback inference.

        When set, the client will automatically switch to local TensorRT
        inference if the Triton server becomes unavailable, and auto-recover
        when Triton comes back online.

        Args:
            engine_path: Path to a TensorRT .engine file to use for local
                fallback inference.

        Raises:
            FileNotFoundError: If the engine_path does not exist.
        """
        import os

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(
                f"Fallback TensorRT engine not found: {engine_path}"
            )

        self._fallback_engine_path = engine_path
        self._fallback_enabled = True
        logger.info(
            "Fallback TensorRT engine configured: %s", engine_path
        )

    def _infer_local_tensorrt(self, frames: list[np.ndarray]) -> list[Detection]:
        """Run inference locally using the configured TensorRT engine.

        Falls back to local TensorRT inference when Triton is unavailable.
        Processes each frame through the engine and returns Detection objects.

        Args:
            frames: List of BGR numpy arrays (H, W, 3) representing video
                frames to run inference on.

        Returns:
            List of Detection objects extracted from the local engine output.

        Raises:
            RuntimeError: If no fallback engine is configured.
        """
        if not self._fallback_enabled or self._fallback_engine_path is None:
            raise RuntimeError(
                "No fallback TensorRT engine configured. Call "
                "set_fallback_engine() to configure local inference."
            )

        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError(
                "tensorrt is required for local fallback inference. "
                "Install with: pip install tensorrt"
            )

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "pycuda is required for local TensorRT inference. "
                "Install with: pip install pycuda"
            )

        all_detections: list[Detection] = []

        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(self._fallback_engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        for frame in frames:
            try:
                detections = self._run_local_engine(
                    frame, engine, context, cuda
                )
                all_detections.extend(detections)
            except Exception as exc:
                logger.error(
                    "Local TensorRT inference failed for frame: %s", exc
                )

        logger.debug(
            "Local TensorRT fallback: %d frames processed, %d detections",
            len(frames),
            len(all_detections),
        )
        return all_detections

    def _run_local_engine(
        self,
        frame: np.ndarray,
        engine,
        context,
        cuda,
    ) -> list[Detection]:
        """Run a single frame through the local TensorRT engine.

        Preprocesses the frame (resize, normalize, HWC→CHW), runs inference,
        and parses the output into Detection objects.

        Args:
            frame: A single BGR numpy array (H, W, 3).
            engine: Deserialized TensorRT engine.
            context: TensorRT execution context.
            cuda: pycuda.driver module.

        Returns:
            List of Detection objects for this frame.
        """
        # Preprocess: resize to 640x640, normalize, HWC→CHW
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Resize using numpy/basic interpolation
        from PIL import Image

        img = Image.fromarray(frame)
        img_resized = img.resize((640, 640), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
        img_array = np.ascontiguousarray(img_array)

        # Set input shape for dynamic batch
        input_name = engine.get_tensor_name(0)
        context.set_input_shape(input_name, (1, 3, 640, 640))

        # Allocate buffers
        input_size = int(np.prod((1, 3, 640, 640))) * 4  # float32
        d_input = cuda.mem_alloc(input_size)

        output_name = engine.get_tensor_name(1)
        output_shape = context.get_tensor_shape(output_name)
        output_size = int(np.prod(output_shape)) * 4
        d_output = cuda.mem_alloc(output_size)

        # Set tensor addresses
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))

        # Copy input to device and run inference
        cuda.memcpy_htod(d_input, img_array)
        stream = cuda.Stream()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

        # Copy output from device
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)

        # Parse YOLOv8 output: (1, 84, 8400) → detections
        detections = self._parse_local_output(output)
        return detections

    def _parse_local_output(
        self,
        raw_output: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> list[Detection]:
        """Parse raw TensorRT output into Detection objects.

        Handles YOLOv8 output format: (1, num_features, num_predictions)
        where num_features = 4 (bbox) + num_classes.

        Args:
            raw_output: Raw output tensor from TensorRT engine.
            conf_threshold: Minimum confidence threshold for detections.

        Returns:
            List of Detection objects.
        """
        timestamp = time.time()

        # Handle batch dimension
        if raw_output.ndim == 3:
            output = raw_output[0]
        else:
            output = raw_output

        # Transpose if needed: (features, predictions) → (predictions, features)
        if output.shape[0] < output.shape[1]:
            output = output.T

        num_predictions = output.shape[0]

        # Extract boxes (center format) and class scores
        boxes_cxcywh = output[:, :4]
        class_scores = output[:, 4:]

        # Get max class score and class id
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = max_scores > conf_threshold
        boxes_cxcywh = boxes_cxcywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # Convert center format to xyxy
        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Simple NMS
        keep = self._simple_nms(boxes_xyxy, max_scores, iou_threshold=0.45)

        detections: list[Detection] = []
        for idx in keep:
            detection = Detection(
                class_id=int(class_ids[idx]),
                class_name=str(int(class_ids[idx])),
                confidence=float(max_scores[idx]),
                bbox=(
                    int(boxes_xyxy[idx, 0]),
                    int(boxes_xyxy[idx, 1]),
                    int(boxes_xyxy[idx, 2]),
                    int(boxes_xyxy[idx, 3]),
                ),
                camera_id=0,
                timestamp=timestamp,
            )
            detections.append(detection)

        return detections

    @staticmethod
    def _simple_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Apply Non-Maximum Suppression to filter overlapping boxes.

        Args:
            boxes: Array of shape (N, 4) with xyxy coordinates.
            scores: Array of shape (N,) with confidence scores.
            iou_threshold: IoU threshold above which boxes are suppressed.

        Returns:
            List of indices of boxes to keep.
        """
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def is_model_ready(self) -> bool:
        """Check if the configured model is loaded and ready for inference.

        Queries the Triton server to determine whether the model specified
        during initialization is in a ready state and can accept inference
        requests. Updates internal _model_ready state.

        Returns:
            True if the model is loaded and ready, False otherwise.
        """
        if self._client is None:
            logger.warning(
                "Model readiness check failed: Triton client is not initialized"
            )
            self._model_ready = False
            return False

        try:
            ready = self._client.is_model_ready(model_name=self.model_name)
            self._model_ready = ready
            if ready:
                logger.debug(
                    "Model '%s' is ready on Triton server", self.model_name
                )
            else:
                logger.warning(
                    "Model '%s' is not ready on Triton server", self.model_name
                )
            return ready
        except Exception as exc:
            logger.warning(
                "Model readiness check failed for '%s': %s",
                self.model_name,
                exc,
            )
            self._model_ready = False
            return False

    def get_model_metadata(self) -> dict:
        """Retrieve metadata for the configured model from Triton.

        Returns information about the model including input/output tensor
        shapes, data types, and supported batch sizes.

        Returns:
            Dictionary containing model metadata with keys:
                - name: Model name
                - versions: Available model versions
                - inputs: List of input tensor descriptions (name, shape, dtype)
                - outputs: List of output tensor descriptions (name, shape, dtype)

        Raises:
            RuntimeError: If the Triton client is not connected or the
                tritonclient library is not available.
        """
        if not self._connected or self._client is None:
            raise RuntimeError(
                "Triton client is not connected. Cannot retrieve model metadata."
            )

        try:
            metadata = self._client.get_model_metadata(
                model_name=self.model_name,
            )
            result = {
                "name": metadata.name,
                "versions": list(metadata.versions),
                "inputs": [
                    {
                        "name": inp.name,
                        "shape": list(inp.shape),
                        "datatype": inp.datatype,
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "shape": list(out.shape),
                        "datatype": out.datatype,
                    }
                    for out in metadata.outputs
                ],
            }
            logger.debug("Retrieved metadata for model '%s': %s", self.model_name, result)
            return result
        except Exception as exc:
            logger.error(
                "Failed to retrieve metadata for model '%s': %s",
                self.model_name,
                exc,
            )
            raise RuntimeError(
                f"Failed to retrieve model metadata: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """Check the health status of the Triton Inference Server.

        Calls the Triton server's is_server_live() endpoint to verify
        that the server is responsive. Updates internal _connected state.

        Returns:
            True if the server is healthy and responsive, False otherwise.
        """
        if self._client is None:
            logger.warning("Health check failed: Triton client is not initialized")
            self._connected = False
            return False

        try:
            is_live = self._client.is_server_live()
            self._connected = is_live
            if is_live:
                logger.debug("Triton server at %s is live", self.server_url)
            else:
                logger.warning("Triton server at %s is not live", self.server_url)
            return is_live
        except Exception as exc:
            logger.warning(
                "Health check failed for Triton server at %s: %s",
                self.server_url,
                exc,
            )
            self._connected = False
            return False

    def start_health_polling(self, interval: float = 5.0) -> None:
        """Start a background thread that periodically checks server health.

        Creates a daemon thread that calls health_check() and is_model_ready()
        at the specified interval, updating internal state (_connected,
        _model_ready) based on results. The polling thread is optional and
        not started automatically during __init__.

        Args:
            interval: Time in seconds between health check polls.
                Defaults to 5.0 seconds per Requirement 3.4.
        """
        if hasattr(self, "_polling_stop_event") and not self._polling_stop_event.is_set():
            logger.warning("Health polling is already running")
            return

        self._polling_interval = interval
        self._polling_stop_event = threading.Event()
        self._polling_thread = threading.Thread(
            target=self._polling_loop,
            name="triton-health-poller",
            daemon=True,
        )
        self._polling_thread.start()
        logger.info(
            "Started health polling thread (interval=%.1fs) for Triton at %s",
            interval,
            self.server_url,
        )

    def stop_health_polling(self) -> None:
        """Stop the background health polling thread.

        Signals the polling thread to stop and waits for it to finish.
        Safe to call even if polling was never started.
        """
        if not hasattr(self, "_polling_stop_event"):
            return

        self._polling_stop_event.set()

        if hasattr(self, "_polling_thread") and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=self._polling_interval + 1.0)
            if self._polling_thread.is_alive():
                logger.warning("Health polling thread did not stop cleanly")
            else:
                logger.info("Health polling thread stopped")

    def _polling_loop(self) -> None:
        """Internal polling loop that runs in the background thread.

        Periodically calls health_check() and is_model_ready(), updating
        internal state. When Triton becomes available again after a fallback,
        automatically switches back to Triton inference. Uses
        threading.Event.wait() for clean shutdown instead of time.sleep().
        """
        logger.debug("Health polling loop started")
        while not self._polling_stop_event.is_set():
            try:
                server_live = self.health_check()
                if server_live:
                    self.is_model_ready()
                    # Auto-recover from fallback when Triton is back
                    if self._using_fallback:
                        logger.info(
                            "Triton server is available again. Switching "
                            "back from local TensorRT fallback."
                        )
                        self._using_fallback = False
            except Exception as exc:
                logger.error("Unexpected error in health polling loop: %s", exc)
                self._connected = False
                self._model_ready = False

            # Wait for the interval or until stop is signaled
            self._polling_stop_event.wait(timeout=self._polling_interval)

        logger.debug("Health polling loop exiting")

    def close(self) -> None:
        """Close the Triton client and release resources.

        Stops the health polling thread if running and cleans up the
        gRPC client connection.
        """
        self.stop_health_polling()
        self._connected = False
        self._model_ready = False
        self._client = None
        logger.info("Triton client closed")
