"""
Triton Python Backend - YOLOv8 Postprocessing Model

Performs postprocessing on raw YOLOv8 inference output:
1. Transpose raw output from [84, 8400] to [8400, 84]
2. Extract bounding boxes (cx, cy, w, h) and class scores (80 classes)
3. Filter predictions by confidence threshold
4. Convert bounding boxes from center format to corner format (x1, y1, x2, y2)
5. Apply Non-Maximum Suppression (NMS)
6. Format output as [N, 6] where each row is [x1, y1, x2, y2, confidence, class_id]

Input:  "raw_output" - float32 tensor of shape [84, 8400]
Output: "detections" - float32 tensor of shape [N, 6]
"""

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    # Allow importing outside Triton for testing purposes
    pb_utils = None

# Default thresholds
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_NMS_IOU_THRESHOLD = 0.45


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """Apply Non-Maximum Suppression to filter overlapping bounding boxes.

    Uses a greedy algorithm that iteratively selects the highest-scoring box
    and removes all boxes with IoU above the threshold.

    Args:
        boxes: Array of shape (N, 4) with corner-format coordinates [x1, y1, x2, y2].
        scores: Array of shape (N,) with confidence scores.
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.

    Returns:
        List of integer indices of boxes to keep.
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

        # Compute IoU of the picked box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess_yolov8_output(
    raw_output: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
) -> np.ndarray:
    """Parse raw YOLOv8 output tensor and produce formatted detections.

    Takes the raw model output of shape [84, 8400] (4 bbox coords + 80 class
    scores for 8400 anchor predictions) and produces a detection array of
    shape [N, 6] where each row is [x1, y1, x2, y2, confidence, class_id].

    Args:
        raw_output: Raw output tensor of shape [84, 8400].
        conf_threshold: Minimum confidence score to keep a detection.
        nms_iou_threshold: IoU threshold for Non-Maximum Suppression.

    Returns:
        Numpy array of shape [N, 6] with dtype float32. Returns shape [0, 6]
        if no detections pass the confidence threshold.
    """
    # Step 1: Transpose from [84, 8400] to [8400, 84] for easier processing
    output = raw_output.T  # Shape: [8400, 84]

    # Step 2: Extract bounding boxes and class scores
    # First 4 values: cx, cy, w, h (center format)
    boxes_cxcywh = output[:, :4]
    # Remaining 80 values: class scores
    class_scores = output[:, 4:]

    # Step 3: Get max class score and class ID for each prediction
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Step 4: Filter by confidence threshold
    mask = max_scores > conf_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    # Handle edge case: no detections above threshold
    if len(boxes_cxcywh) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Step 5: Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

    # Step 6: Apply Non-Maximum Suppression
    keep_indices = _nms(boxes_xyxy, max_scores, nms_iou_threshold)

    if len(keep_indices) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Step 7: Format output as [N, 6]: [x1, y1, x2, y2, confidence, class_id]
    kept_boxes = boxes_xyxy[keep_indices]
    kept_scores = max_scores[keep_indices]
    kept_class_ids = class_ids[keep_indices]

    detections = np.zeros((len(keep_indices), 6), dtype=np.float32)
    detections[:, 0:4] = kept_boxes
    detections[:, 4] = kept_scores
    detections[:, 5] = kept_class_ids.astype(np.float32)

    return detections


class TritonPythonModel:
    """Triton Python backend model for YOLOv8 postprocessing (NMS + formatting)."""

    def initialize(self, args):
        """Called once when the model is loaded by Triton.

        Initializes configurable thresholds for confidence filtering and NMS.

        Args:
            args: Dictionary containing model configuration and instance info.
        """
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.nms_iou_threshold = DEFAULT_NMS_IOU_THRESHOLD
        self.logger = pb_utils.Logger if pb_utils else None

        # Attempt to read thresholds from model config parameters
        if pb_utils and "model_config" in args:
            import json
            model_config = json.loads(args["model_config"])
            parameters = model_config.get("parameters", {})

            if "conf_threshold" in parameters:
                self.conf_threshold = float(
                    parameters["conf_threshold"]["string_value"]
                )
            if "nms_iou_threshold" in parameters:
                self.nms_iou_threshold = float(
                    parameters["nms_iou_threshold"]["string_value"]
                )

        if self.logger:
            self.logger.log(
                f"Postprocessing model initialized: "
                f"conf_threshold={self.conf_threshold}, "
                f"nms_iou_threshold={self.nms_iou_threshold}"
            )

    def execute(self, requests):
        """Called for each batch of inference requests.

        Each request contains the raw YOLOv8 output tensor of shape [84, 8400].
        The model applies confidence filtering, NMS, and formats the output as
        a float32 tensor of shape [N, 6].

        Args:
            requests: List of pb_utils.InferenceRequest objects.

        Returns:
            List of pb_utils.InferenceResponse objects, one per request.
        """
        responses = []

        for request in requests:
            try:
                # Get the input tensor
                input_tensor = pb_utils.get_input_tensor_by_name(
                    request, "raw_output"
                )
                if input_tensor is None:
                    error = pb_utils.TritonError(
                        "Input tensor 'raw_output' not found in request"
                    )
                    response = pb_utils.InferenceResponse(
                        output_tensors=[], error=error
                    )
                    responses.append(response)
                    continue

                # Convert to numpy array - shape [84, 8400]
                raw_output = input_tensor.as_numpy()

                # Validate input shape
                if raw_output.ndim != 2 or raw_output.shape[0] != 84 or raw_output.shape[1] != 8400:
                    error = pb_utils.TritonError(
                        f"Expected input shape [84, 8400], got {raw_output.shape}"
                    )
                    response = pb_utils.InferenceResponse(
                        output_tensors=[], error=error
                    )
                    responses.append(response)
                    continue

                # Run postprocessing
                detections = postprocess_yolov8_output(
                    raw_output,
                    conf_threshold=self.conf_threshold,
                    nms_iou_threshold=self.nms_iou_threshold,
                )

                # Ensure contiguous memory layout
                detections = np.ascontiguousarray(detections)

                # Create output tensor
                output_tensor = pb_utils.Tensor("detections", detections)

                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(response)

            except Exception as e:
                # Log the error and return an error response
                if self.logger:
                    self.logger.log(
                        f"ERROR: Postprocessing failed: {str(e)}"
                    )
                error = pb_utils.TritonError(
                    f"Postprocessing failed: {str(e)}"
                )
                response = pb_utils.InferenceResponse(
                    output_tensors=[], error=error
                )
                responses.append(response)

        return responses

    def finalize(self):
        """Called once when the model is unloaded by Triton."""
        if self.logger:
            self.logger.log("Postprocessing model finalized")
