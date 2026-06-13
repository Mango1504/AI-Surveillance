/**
 * @file metropolis_cpp.h
 * @brief Public header for metropolis_cpp C++/CUDA extensions.
 *
 * Declares the public interface for all C++ extension functions
 * used by the NVIDIA Metropolis integration.
 */

#ifndef METROPOLIS_CPP_H
#define METROPOLIS_CPP_H

#include <vector>
#include <tuple>
#include <cstdint>

namespace metropolis {

/**
 * @brief Batch preprocess frames on GPU.
 *
 * Performs resize, BGR→RGB conversion, HWC→CHW layout transformation,
 * and normalization to [0, 1] range entirely in GPU memory.
 *
 * @param frames Vector of raw frame data (BGR, HWC, uint8)
 * @param frame_heights Heights of input frames
 * @param frame_widths Widths of input frames
 * @param target_h Target height after resize
 * @param target_w Target width after resize
 * @param normalize Whether to normalize pixel values to [0, 1]
 * @return Preprocessed tensor data in NCHW float32 format
 */
std::vector<float> cuda_preprocess(
    const std::vector<uint8_t*>& frames,
    const std::vector<int>& frame_heights,
    const std::vector<int>& frame_widths,
    int target_h = 640,
    int target_w = 640,
    bool normalize = true
);

/**
 * @brief GPU-accelerated batched Non-Maximum Suppression.
 *
 * Produces identical results to torchvision.ops.nms reference.
 *
 * @param boxes Bounding boxes in (x1, y1, x2, y2) format, shape (N, 4)
 * @param scores Confidence scores, shape (N,)
 * @param classes Class indices, shape (N,)
 * @param iou_threshold IoU threshold for suppression (default: 0.45)
 * @param score_threshold Minimum score to keep (default: 0.25)
 * @return Indices of kept boxes after NMS
 */
std::vector<int> batched_nms(
    const float* boxes,
    const float* scores,
    const int32_t* classes,
    int num_boxes,
    float iou_threshold = 0.45f,
    float score_threshold = 0.25f
);

/**
 * @brief Compute exponential-recency-weighted risk score.
 *
 * Calculates a risk score in [0, 1] based on weighted events within
 * a sliding time window, with exponential decay favoring recent events.
 *
 * @param timestamps Event timestamps (Unix epoch seconds)
 * @param weights Event weights/severities
 * @param num_events Number of events
 * @param window_secs Time window in seconds
 * @param current_time Current timestamp for decay calculation
 * @return Risk score in range [0.0, 1.0]
 */
float compute_risk_score(
    const double* timestamps,
    const double* weights,
    int num_events,
    double window_secs,
    double current_time
);

}  // namespace metropolis

#endif  // METROPOLIS_CPP_H
