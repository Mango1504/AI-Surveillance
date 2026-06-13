/**
 * @file bindings.cpp
 * @brief pybind11 bindings exposing C++/CUDA extensions to Python.
 *
 * Provides Python-accessible functions with numpy array interoperability:
 * - cuda_preprocess: Batch frame preprocessing on GPU
 * - batched_nms: GPU-accelerated Non-Maximum Suppression
 * - compute_risk_score: Exponential-recency-weighted risk scoring
 *
 * All functions accept and return numpy arrays directly via pybind11's
 * numpy integration. Input validation and error handling convert C++
 * exceptions to Python exceptions automatically.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "metropolis_cpp.h"

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;

/**
 * @brief Python wrapper for CUDA batch frame preprocessing.
 *
 * Accepts a list of numpy arrays (each HxWx3 uint8 BGR frames) and returns
 * a single numpy array of shape (N, 3, target_h, target_w) in float32 NCHW format.
 *
 * @param frames      List of numpy arrays, each shape (H, W, 3) dtype uint8
 * @param target_size Tuple of (height, width) for output dimensions
 * @param normalize   Whether to normalize pixel values to [0, 1]
 * @return numpy array of shape (N, 3, target_h, target_w) dtype float32
 */
py::array_t<float> py_cuda_preprocess(
    const std::vector<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>& frames,
    std::tuple<int, int> target_size,
    bool normalize
) {
    const int batch_size = static_cast<int>(frames.size());

    if (batch_size == 0) {
        // Return empty array with correct shape (0, 3, target_h, target_w)
        auto [target_h, target_w] = target_size;
        return py::array_t<float>({0, 3, target_h, target_w});
    }

    // Extract target dimensions
    auto [target_h, target_w] = target_size;

    if (target_h <= 0 || target_w <= 0) {
        throw std::invalid_argument("target_size dimensions must be positive");
    }

    // Validate and extract frame data pointers, heights, and widths
    std::vector<uint8_t*> frame_ptrs(batch_size);
    std::vector<int> frame_heights(batch_size);
    std::vector<int> frame_widths(batch_size);

    for (int i = 0; i < batch_size; i++) {
        py::buffer_info buf = frames[i].request();

        // Validate dimensions: must be (H, W, 3)
        if (buf.ndim != 3) {
            throw std::invalid_argument(
                "Frame " + std::to_string(i) + " must be 3-dimensional (H, W, 3), "
                "got " + std::to_string(buf.ndim) + " dimensions");
        }

        if (buf.shape[2] != 3) {
            throw std::invalid_argument(
                "Frame " + std::to_string(i) + " must have 3 channels (BGR), "
                "got " + std::to_string(buf.shape[2]) + " channels");
        }

        frame_heights[i] = static_cast<int>(buf.shape[0]);
        frame_widths[i] = static_cast<int>(buf.shape[1]);
        frame_ptrs[i] = static_cast<uint8_t*>(buf.ptr);
    }

    // Call the C++ CUDA implementation
    std::vector<float> result = metropolis::cuda_preprocess(
        frame_ptrs, frame_heights, frame_widths,
        target_h, target_w, normalize
    );

    // Create output numpy array with shape (N, 3, target_h, target_w)
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(batch_size),
        3,
        static_cast<py::ssize_t>(target_h),
        static_cast<py::ssize_t>(target_w)
    };

    // Allocate output array and copy data
    py::array_t<float> output(shape);
    auto output_buf = output.request();
    float* output_ptr = static_cast<float*>(output_buf.ptr);

    std::memcpy(output_ptr, result.data(), result.size() * sizeof(float));

    return output;
}

/**
 * @brief Python wrapper for GPU-accelerated batched NMS.
 *
 * Accepts numpy arrays for boxes, scores, and classes, and returns
 * a numpy array of kept indices after non-maximum suppression.
 *
 * @param boxes          numpy array of shape (N, 4) dtype float32 (x1, y1, x2, y2)
 * @param scores         numpy array of shape (N,) dtype float32
 * @param classes        numpy array of shape (N,) dtype int32
 * @param iou_threshold  IoU threshold for suppression (default: 0.45)
 * @param score_threshold Minimum score to keep (default: 0.25)
 * @return numpy array of kept indices, dtype int32
 */
py::array_t<int32_t> py_batched_nms(
    py::array_t<float, py::array::c_style | py::array::forcecast> boxes,
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> classes,
    float iou_threshold,
    float score_threshold
) {
    // Get buffer info for validation
    py::buffer_info boxes_buf = boxes.request();
    py::buffer_info scores_buf = scores.request();
    py::buffer_info classes_buf = classes.request();

    // Validate boxes shape: (N, 4)
    if (boxes_buf.ndim != 2) {
        throw std::invalid_argument(
            "boxes must be 2-dimensional (N, 4), got " +
            std::to_string(boxes_buf.ndim) + " dimensions");
    }
    if (boxes_buf.shape[1] != 4) {
        throw std::invalid_argument(
            "boxes must have 4 columns (x1, y1, x2, y2), got " +
            std::to_string(boxes_buf.shape[1]) + " columns");
    }

    const int num_boxes = static_cast<int>(boxes_buf.shape[0]);

    // Validate scores shape: (N,)
    if (scores_buf.ndim != 1) {
        throw std::invalid_argument(
            "scores must be 1-dimensional (N,), got " +
            std::to_string(scores_buf.ndim) + " dimensions");
    }
    if (scores_buf.shape[0] != num_boxes) {
        throw std::invalid_argument(
            "scores length (" + std::to_string(scores_buf.shape[0]) +
            ") must match number of boxes (" + std::to_string(num_boxes) + ")");
    }

    // Validate classes shape: (N,)
    if (classes_buf.ndim != 1) {
        throw std::invalid_argument(
            "classes must be 1-dimensional (N,), got " +
            std::to_string(classes_buf.ndim) + " dimensions");
    }
    if (classes_buf.shape[0] != num_boxes) {
        throw std::invalid_argument(
            "classes length (" + std::to_string(classes_buf.shape[0]) +
            ") must match number of boxes (" + std::to_string(num_boxes) + ")");
    }

    // Validate threshold ranges
    if (iou_threshold < 0.0f || iou_threshold > 1.0f) {
        throw std::invalid_argument(
            "iou_threshold must be in [0, 1], got " + std::to_string(iou_threshold));
    }
    if (score_threshold < 0.0f || score_threshold > 1.0f) {
        throw std::invalid_argument(
            "score_threshold must be in [0, 1], got " + std::to_string(score_threshold));
    }

    // Handle empty input
    if (num_boxes == 0) {
        return py::array_t<int32_t>(0);
    }

    // Extract data pointers
    const float* boxes_ptr = static_cast<const float*>(boxes_buf.ptr);
    const float* scores_ptr = static_cast<const float*>(scores_buf.ptr);
    const int32_t* classes_ptr = static_cast<const int32_t*>(classes_buf.ptr);

    // Call the C++ CUDA implementation
    std::vector<int> kept_indices = metropolis::batched_nms(
        boxes_ptr, scores_ptr, classes_ptr,
        num_boxes, iou_threshold, score_threshold
    );

    // Create output numpy array
    const py::ssize_t num_kept = static_cast<py::ssize_t>(kept_indices.size());
    py::array_t<int32_t> output(num_kept);

    if (num_kept > 0) {
        auto output_buf = output.request();
        int32_t* output_ptr = static_cast<int32_t*>(output_buf.ptr);

        for (py::ssize_t i = 0; i < num_kept; i++) {
            output_ptr[i] = static_cast<int32_t>(kept_indices[i]);
        }
    }

    return output;
}

/**
 * @brief Python wrapper for exponential-recency-weighted risk score computation.
 *
 * Supports two input formats:
 *   1. List of (timestamp, weight) tuples
 *   2. Two separate numpy arrays: timestamps and weights
 *
 * This overload accepts two numpy arrays for timestamps and weights.
 *
 * @param timestamps   numpy array of shape (N,) dtype float64 (Unix epoch seconds)
 * @param weights      numpy array of shape (N,) dtype float64 (event severities)
 * @param window_secs  Time window in seconds for decay calculation
 * @param current_time Current timestamp (Unix epoch seconds)
 * @return Risk score in range [0.0, 1.0]
 */
float py_compute_risk_score_arrays(
    py::array_t<double, py::array::c_style | py::array::forcecast> timestamps,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    double window_secs,
    double current_time
) {
    py::buffer_info ts_buf = timestamps.request();
    py::buffer_info wt_buf = weights.request();

    // Validate dimensions
    if (ts_buf.ndim != 1) {
        throw std::invalid_argument(
            "timestamps must be 1-dimensional, got " +
            std::to_string(ts_buf.ndim) + " dimensions");
    }
    if (wt_buf.ndim != 1) {
        throw std::invalid_argument(
            "weights must be 1-dimensional, got " +
            std::to_string(wt_buf.ndim) + " dimensions");
    }

    const int num_events = static_cast<int>(ts_buf.shape[0]);

    if (wt_buf.shape[0] != num_events) {
        throw std::invalid_argument(
            "timestamps length (" + std::to_string(ts_buf.shape[0]) +
            ") must match weights length (" + std::to_string(wt_buf.shape[0]) + ")");
    }

    // Handle empty input
    if (num_events == 0) {
        return 0.0f;
    }

    // Extract data pointers
    const double* ts_ptr = static_cast<const double*>(ts_buf.ptr);
    const double* wt_ptr = static_cast<const double*>(wt_buf.ptr);

    // Call the C++ implementation
    return metropolis::compute_risk_score(
        ts_ptr, wt_ptr, num_events, window_secs, current_time
    );
}

/**
 * @brief Python wrapper for risk score computation accepting list of tuples.
 *
 * Accepts events as a list of (timestamp, weight) tuples and converts
 * them to the array format expected by the C++ implementation.
 *
 * @param events       List of (timestamp, weight) tuples
 * @param window_secs  Time window in seconds for decay calculation
 * @param current_time Current timestamp (Unix epoch seconds)
 * @return Risk score in range [0.0, 1.0]
 */
float py_compute_risk_score_tuples(
    const std::vector<std::tuple<double, double>>& events,
    double window_secs,
    double current_time
) {
    const int num_events = static_cast<int>(events.size());

    if (num_events == 0) {
        return 0.0f;
    }

    // Convert tuples to separate arrays
    std::vector<double> timestamps(num_events);
    std::vector<double> weights(num_events);

    for (int i = 0; i < num_events; i++) {
        timestamps[i] = std::get<0>(events[i]);
        weights[i] = std::get<1>(events[i]);
    }

    // Call the C++ implementation
    return metropolis::compute_risk_score(
        timestamps.data(), weights.data(), num_events, window_secs, current_time
    );
}

/**
 * @brief Define the pybind11 module "metropolis_cpp".
 *
 * Exposes all C++ extension functions to Python with proper docstrings,
 * default argument values, and numpy array interoperability.
 */
PYBIND11_MODULE(metropolis_cpp, m) {
    m.doc() = R"pbdoc(
        NVIDIA Metropolis C++/CUDA Extensions
        ======================================

        High-performance C++ and CUDA implementations for the AI Surveillance
        system's compute-intensive operations. All functions accept and return
        numpy arrays directly.

        Functions:
            cuda_preprocess: Batch frame preprocessing on GPU (resize, BGR->RGB, HWC->CHW, normalize)
            batched_nms: GPU-accelerated batched Non-Maximum Suppression
            compute_risk_score: Exponential-recency-weighted risk score computation
    )pbdoc";

    // --- cuda_preprocess ---
    m.def("cuda_preprocess", &py_cuda_preprocess,
        py::arg("frames"),
        py::arg("target_size") = std::make_tuple(640, 640),
        py::arg("normalize") = true,
        R"pbdoc(
            Batch preprocess frames on GPU using CUDA.

            Performs resize (bilinear interpolation), BGR to RGB conversion,
            HWC to CHW layout transformation, and optional normalization to [0, 1].
            All operations run entirely in GPU memory for maximum throughput.

            Args:
                frames: List of numpy arrays, each of shape (H, W, 3) with dtype uint8.
                    Frames can have different heights and widths.
                target_size: Tuple of (height, width) for the output dimensions.
                    Default is (640, 640).
                normalize: Whether to normalize pixel values to [0.0, 1.0] range.
                    Default is True.

            Returns:
                numpy.ndarray: Preprocessed tensor of shape (N, 3, target_h, target_w)
                with dtype float32 in NCHW format.

            Raises:
                ValueError: If any frame is not 3-dimensional or doesn't have 3 channels.
                RuntimeError: If CUDA operations fail (e.g., out of GPU memory).

            Example:
                >>> import numpy as np
                >>> import metropolis_cpp
                >>> frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
                >>> result = metropolis_cpp.cuda_preprocess(frames, target_size=(640, 640))
                >>> result.shape
                (1, 3, 640, 640)
        )pbdoc"
    );

    // --- batched_nms ---
    m.def("batched_nms", &py_batched_nms,
        py::arg("boxes"),
        py::arg("scores"),
        py::arg("classes"),
        py::arg("iou_threshold") = 0.45f,
        py::arg("score_threshold") = 0.25f,
        R"pbdoc(
            GPU-accelerated batched Non-Maximum Suppression.

            Performs class-aware NMS where only boxes of the same class can
            suppress each other. Produces identical results to the Python
            reference implementation (torchvision.ops.batched_nms).

            Args:
                boxes: numpy.ndarray of shape (N, 4) with dtype float32.
                    Bounding boxes in (x1, y1, x2, y2) format.
                scores: numpy.ndarray of shape (N,) with dtype float32.
                    Confidence scores for each box.
                classes: numpy.ndarray of shape (N,) with dtype int32.
                    Class index for each box (NMS is applied per-class).
                iou_threshold: IoU threshold for suppression. Boxes with IoU
                    above this value with a higher-scored box are suppressed.
                    Default is 0.45.
                score_threshold: Minimum confidence score to keep a box.
                    Boxes below this threshold are discarded before NMS.
                    Default is 0.25.

            Returns:
                numpy.ndarray: Indices of kept boxes (dtype int32), sorted by
                descending confidence score.

            Raises:
                ValueError: If input shapes are inconsistent or thresholds are
                    out of [0, 1] range.
                RuntimeError: If CUDA operations fail.

            Example:
                >>> import numpy as np
                >>> import metropolis_cpp
                >>> boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52]], dtype=np.float32)
                >>> scores = np.array([0.9, 0.8], dtype=np.float32)
                >>> classes = np.array([0, 0], dtype=np.int32)
                >>> kept = metropolis_cpp.batched_nms(boxes, scores, classes)
                >>> kept
                array([0], dtype=int32)
        )pbdoc"
    );

    // --- compute_risk_score (numpy array version) ---
    m.def("compute_risk_score", &py_compute_risk_score_arrays,
        py::arg("timestamps"),
        py::arg("weights"),
        py::arg("window_secs"),
        py::arg("current_time"),
        R"pbdoc(
            Compute exponential-recency-weighted risk score from numpy arrays.

            Calculates a risk score in [0, 1] based on weighted events within
            a sliding time window. Events closer to current_time contribute more
            due to exponential decay (tau = window_secs / 3.0).

            Args:
                timestamps: numpy.ndarray of shape (N,) with dtype float64.
                    Event timestamps in Unix epoch seconds.
                weights: numpy.ndarray of shape (N,) with dtype float64.
                    Event weights/severities (higher = more risky).
                window_secs: Time window in seconds. Events outside
                    [current_time - window_secs, current_time] are excluded.
                current_time: Current timestamp in Unix epoch seconds for
                    decay calculation.

            Returns:
                float: Risk score in range [0.0, 1.0].

            Raises:
                ValueError: If timestamps and weights have different lengths.

            Example:
                >>> import numpy as np
                >>> import metropolis_cpp
                >>> timestamps = np.array([100.0, 105.0, 110.0])
                >>> weights = np.array([1.0, 2.0, 3.0])
                >>> score = metropolis_cpp.compute_risk_score(timestamps, weights, 60.0, 110.0)
                >>> 0.0 <= score <= 1.0
                True
        )pbdoc"
    );

    // --- compute_risk_score (tuple list version) ---
    m.def("compute_risk_score", &py_compute_risk_score_tuples,
        py::arg("events"),
        py::arg("window_secs"),
        py::arg("current_time"),
        R"pbdoc(
            Compute exponential-recency-weighted risk score from event tuples.

            Alternative interface that accepts events as a list of
            (timestamp, weight) tuples instead of separate arrays.

            Args:
                events: List of (timestamp, weight) tuples where timestamp
                    is Unix epoch seconds (float) and weight is the event
                    severity (float).
                window_secs: Time window in seconds. Events outside
                    [current_time - window_secs, current_time] are excluded.
                current_time: Current timestamp in Unix epoch seconds for
                    decay calculation.

            Returns:
                float: Risk score in range [0.0, 1.0].

            Example:
                >>> import metropolis_cpp
                >>> events = [(100.0, 1.0), (105.0, 2.0), (110.0, 3.0)]
                >>> score = metropolis_cpp.compute_risk_score(events, 60.0, 110.0)
                >>> 0.0 <= score <= 1.0
                True
        )pbdoc"
    );

    // Module version
    m.attr("__version__") = "0.1.0";
}
