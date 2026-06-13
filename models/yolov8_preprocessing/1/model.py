"""
Triton Python Backend - YOLOv8 Preprocessing Model

Performs image preprocessing for YOLOv8 inference:
1. Resize to 640x640 using bilinear interpolation
2. Normalize pixel values to [0, 1] (divide by 255.0)
3. Convert HWC to CHW format (transpose)
4. Convert to float32

Input:  "raw_image" - uint8 tensor of shape [H, W, 3] (HWC format)
Output: "processed" - float32 tensor of shape [3, 640, 640] (CHW, normalized)
"""

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    # Allow importing outside Triton for testing purposes
    pb_utils = None

# Target dimensions for YOLOv8 input
TARGET_WIDTH = 640
TARGET_HEIGHT = 640


def _resize_bilinear(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize an image using bilinear interpolation (pure numpy).

    Args:
        image: Input image array of shape [H, W, C] with dtype uint8 or float.
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized image of shape [target_h, target_w, C] with same dtype as input.
    """
    src_h, src_w = image.shape[:2]

    # If already the target size, return as-is
    if src_h == target_h and src_w == target_w:
        return image

    # Try to use cv2 for faster resize if available
    try:
        import cv2
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        pass

    # Pure numpy bilinear interpolation fallback
    # Create coordinate grids for the target image
    row_scale = src_h / target_h
    col_scale = src_w / target_w

    # Target pixel centers mapped back to source coordinates
    target_rows = np.arange(target_h, dtype=np.float32)
    target_cols = np.arange(target_w, dtype=np.float32)

    src_rows = (target_rows + 0.5) * row_scale - 0.5
    src_cols = (target_cols + 0.5) * col_scale - 0.5

    # Clip to valid source range
    src_rows = np.clip(src_rows, 0, src_h - 1)
    src_cols = np.clip(src_cols, 0, src_w - 1)

    # Integer and fractional parts
    row_floor = np.floor(src_rows).astype(np.int32)
    col_floor = np.floor(src_cols).astype(np.int32)
    row_ceil = np.minimum(row_floor + 1, src_h - 1)
    col_ceil = np.minimum(col_floor + 1, src_w - 1)

    row_frac = (src_rows - row_floor).reshape(-1, 1, 1)  # [target_h, 1, 1]
    col_frac = (src_cols - col_floor).reshape(1, -1, 1)  # [1, target_w, 1]

    # Gather the four corner pixels
    # image shape: [src_h, src_w, C]
    top_left = image[row_floor][:, col_floor]        # [target_h, target_w, C]
    top_right = image[row_floor][:, col_ceil]        # [target_h, target_w, C]
    bottom_left = image[row_ceil][:, col_floor]      # [target_h, target_w, C]
    bottom_right = image[row_ceil][:, col_ceil]      # [target_h, target_w, C]

    # Convert to float for interpolation
    top_left = top_left.astype(np.float32)
    top_right = top_right.astype(np.float32)
    bottom_left = bottom_left.astype(np.float32)
    bottom_right = bottom_right.astype(np.float32)

    # Bilinear interpolation
    top = top_left * (1 - col_frac) + top_right * col_frac
    bottom = bottom_left * (1 - col_frac) + bottom_right * col_frac
    result = top * (1 - row_frac) + bottom * row_frac

    if image.dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    return result


class TritonPythonModel:
    """Triton Python backend model for YOLOv8 image preprocessing."""

    def initialize(self, args):
        """
        Called once when the model is loaded by Triton.

        Args:
            args: Dictionary containing model configuration and instance info.
        """
        self.target_h = TARGET_HEIGHT
        self.target_w = TARGET_WIDTH
        self.logger = pb_utils.Logger if pb_utils else None

        # Check if cv2 is available for optimized resize
        self._use_cv2 = False
        try:
            import cv2
            self._use_cv2 = True
            if self.logger:
                self.logger.log("INFO: OpenCV available, using cv2.resize for preprocessing")
        except ImportError:
            if self.logger:
                self.logger.log("INFO: OpenCV not available, using numpy bilinear interpolation")

        if self.logger:
            self.logger.log(
                f"Preprocessing model initialized: "
                f"target_size=({self.target_h}, {self.target_w}), "
                f"cv2_available={self._use_cv2}"
            )

    def execute(self, requests):
        """
        Called for each batch of inference requests.

        Each request contains a single image as a uint8 tensor of shape [H, W, 3].
        The model resizes, normalizes, and transposes the image to produce a
        float32 tensor of shape [3, 640, 640].

        Args:
            requests: List of pb_utils.InferenceRequest objects.

        Returns:
            List of pb_utils.InferenceResponse objects, one per request.
        """
        responses = []

        for request in requests:
            try:
                # Get the input tensor
                input_tensor = pb_utils.get_input_tensor_by_name(request, "raw_image")
                if input_tensor is None:
                    error = pb_utils.TritonError(
                        "Input tensor 'raw_image' not found in request"
                    )
                    response = pb_utils.InferenceResponse(
                        output_tensors=[], error=error
                    )
                    responses.append(response)
                    continue

                # Convert to numpy array - shape [H, W, 3], dtype uint8
                image = input_tensor.as_numpy()

                # Validate input shape
                if image.ndim != 3 or image.shape[2] != 3:
                    error = pb_utils.TritonError(
                        f"Expected input shape [H, W, 3], got {image.shape}"
                    )
                    response = pb_utils.InferenceResponse(
                        output_tensors=[], error=error
                    )
                    responses.append(response)
                    continue

                # Step 1: Resize to 640x640 using bilinear interpolation
                resized = _resize_bilinear(image, self.target_h, self.target_w)

                # Step 2: Normalize pixel values to [0, 1] and convert to float32
                normalized = resized.astype(np.float32) / 255.0

                # Step 3: Convert HWC to CHW format
                # Input shape: [640, 640, 3] -> Output shape: [3, 640, 640]
                chw = np.transpose(normalized, (2, 0, 1))

                # Ensure contiguous memory layout for efficient transfer
                chw = np.ascontiguousarray(chw)

                # Create output tensor
                output_tensor = pb_utils.Tensor("processed", chw)

                # Create response
                response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(response)

            except Exception as e:
                # Log the error and return an error response
                if self.logger:
                    self.logger.log(f"ERROR: Preprocessing failed: {str(e)}")
                error = pb_utils.TritonError(
                    f"Preprocessing failed: {str(e)}"
                )
                response = pb_utils.InferenceResponse(
                    output_tensors=[], error=error
                )
                responses.append(response)

        return responses

    def finalize(self):
        """Called once when the model is unloaded by Triton."""
        if self.logger:
            self.logger.log("Preprocessing model finalized")
