/**
 * @file preprocess.cu
 * @brief CUDA batch frame preprocessing kernels.
 *
 * Implements GPU-accelerated batch preprocessing for video frames:
 * - Resize to target dimensions (bilinear interpolation)
 * - BGR to RGB color space conversion
 * - HWC to CHW tensor layout conversion
 * - Normalization to [0, 1] float range
 *
 * All operations run entirely in GPU memory for zero-copy performance.
 * The kernel processes multiple frames in a single launch using the
 * z-dimension of the CUDA grid for batch indexing.
 *
 * Output format: NCHW float32 (batch × 3 channels × target_h × target_w)
 */

#include "metropolis_cpp.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <stdexcept>

// Block dimensions for 2D spatial processing
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Maximum supported batch size (for static device arrays)
#define MAX_BATCH_SIZE 64

/**
 * @brief Check CUDA errors and throw on failure.
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            char msg[256];                                                     \
            snprintf(msg, sizeof(msg),                                         \
                     "CUDA error at %s:%d: %s",                                \
                     __FILE__, __LINE__, cudaGetErrorString(err));              \
            throw std::runtime_error(msg);                                     \
        }                                                                      \
    } while (0)

/**
 * @brief Device-side frame descriptor for batch processing.
 *
 * Each frame in the batch has its own source pointer, height, and width.
 * This allows variable-size input frames in a single kernel launch.
 */
struct FrameDesc {
    const uint8_t* data;  ///< Pointer to frame data in device memory (BGR, HWC)
    int height;           ///< Source frame height
    int width;            ///< Source frame width
};

/**
 * @brief CUDA kernel for batch frame preprocessing.
 *
 * For each output pixel (x, y) in frame `batch_idx`:
 *   1. Compute source coordinates using bilinear interpolation weights
 *   2. Sample 4 neighboring pixels from the source frame
 *   3. Interpolate to get the BGR value at the fractional source position
 *   4. Swap B and R channels (BGR → RGB)
 *   5. Optionally normalize to [0, 1] by dividing by 255.0
 *   6. Write to output in CHW layout (channel-first)
 *
 * Grid layout:
 *   - blockDim: (BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)
 *   - gridDim:  (ceil(target_w / BLOCK_SIZE_X),
 *                ceil(target_h / BLOCK_SIZE_Y),
 *                batch_size)
 *
 * @param frames     Array of frame descriptors (device pointers + dimensions)
 * @param output     Output buffer in NCHW float32 format
 * @param target_h   Target output height
 * @param target_w   Target output width
 * @param normalize  Whether to normalize pixel values to [0, 1]
 */
__global__ void preprocess_kernel(
    const FrameDesc* __restrict__ frames,
    float* __restrict__ output,
    int target_h,
    int target_w,
    bool normalize
) {
    // Determine output pixel coordinates and batch index
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;

    // Bounds check
    if (out_x >= target_w || out_y >= target_h) {
        return;
    }

    // Get source frame descriptor
    const FrameDesc& frame = frames[batch_idx];
    const int src_h = frame.height;
    const int src_w = frame.width;
    const uint8_t* src_data = frame.data;

    // Compute scale factors for resize (source / target)
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(target_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(target_h);

    // Map output pixel to source coordinates (center-aligned)
    const float src_x = (static_cast<float>(out_x) + 0.5f) * scale_x - 0.5f;
    const float src_y = (static_cast<float>(out_y) + 0.5f) * scale_y - 0.5f;

    // Compute integer coordinates for bilinear interpolation
    const int x0 = static_cast<int>(floorf(src_x));
    const int y0 = static_cast<int>(floorf(src_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    // Compute fractional weights
    const float wx = src_x - static_cast<float>(x0);
    const float wy = src_y - static_cast<float>(y0);

    // Clamp coordinates to valid range
    const int x0_clamped = max(0, min(x0, src_w - 1));
    const int y0_clamped = max(0, min(y0, src_h - 1));
    const int x1_clamped = max(0, min(x1, src_w - 1));
    const int y1_clamped = max(0, min(y1, src_h - 1));

    // Source stride (3 channels per pixel, HWC layout)
    const int src_stride = src_w * 3;

    // Perform bilinear interpolation for each BGR channel
    float bgr[3];
    for (int c = 0; c < 3; c++) {
        // Sample 4 neighboring pixels
        const float val00 = static_cast<float>(src_data[y0_clamped * src_stride + x0_clamped * 3 + c]);
        const float val01 = static_cast<float>(src_data[y0_clamped * src_stride + x1_clamped * 3 + c]);
        const float val10 = static_cast<float>(src_data[y1_clamped * src_stride + x0_clamped * 3 + c]);
        const float val11 = static_cast<float>(src_data[y1_clamped * src_stride + x1_clamped * 3 + c]);

        // Bilinear interpolation
        bgr[c] = (1.0f - wy) * ((1.0f - wx) * val00 + wx * val01)
                + wy          * ((1.0f - wx) * val10 + wx * val11);
    }

    // BGR → RGB conversion (swap channels 0 and 2)
    float rgb[3];
    rgb[0] = bgr[2];  // R = B channel from BGR
    rgb[1] = bgr[1];  // G = G channel
    rgb[2] = bgr[0];  // B = R channel from BGR

    // Normalize to [0, 1] if requested
    if (normalize) {
        rgb[0] /= 255.0f;
        rgb[1] /= 255.0f;
        rgb[2] /= 255.0f;
    }

    // Write to output in NCHW format
    // Layout: output[batch_idx][channel][out_y][out_x]
    // Linear index: batch_idx * (3 * H * W) + channel * (H * W) + out_y * W + out_x
    const int channel_stride = target_h * target_w;
    const int batch_stride = 3 * channel_stride;
    const int base_offset = batch_idx * batch_stride + out_y * target_w + out_x;

    output[base_offset + 0 * channel_stride] = rgb[0];  // R channel
    output[base_offset + 1 * channel_stride] = rgb[1];  // G channel
    output[base_offset + 2 * channel_stride] = rgb[2];  // B channel
}

namespace metropolis {

/**
 * @brief Batch preprocess frames on GPU.
 *
 * Host function that orchestrates the CUDA preprocessing pipeline:
 *   1. Allocates device memory for input frames and output tensor
 *   2. Copies frame data from host to device
 *   3. Builds frame descriptor array on device
 *   4. Launches the preprocessing kernel
 *   5. Copies results back to host
 *   6. Frees device memory
 *
 * @param frames        Vector of pointers to raw frame data (BGR, HWC, uint8)
 * @param frame_heights Heights of each input frame
 * @param frame_widths  Widths of each input frame
 * @param target_h      Target output height (default: 640)
 * @param target_w      Target output width (default: 640)
 * @param normalize     Whether to normalize to [0, 1] (default: true)
 * @return Preprocessed data as contiguous float vector in NCHW format
 *
 * @throws std::invalid_argument if input vectors have mismatched sizes
 * @throws std::invalid_argument if batch size exceeds MAX_BATCH_SIZE
 * @throws std::runtime_error on CUDA allocation or kernel launch failure
 */
std::vector<float> cuda_preprocess(
    const std::vector<uint8_t*>& frames,
    const std::vector<int>& frame_heights,
    const std::vector<int>& frame_widths,
    int target_h,
    int target_w,
    bool normalize
) {
    const int batch_size = static_cast<int>(frames.size());

    // Validate inputs
    if (batch_size == 0) {
        return {};
    }

    if (static_cast<int>(frame_heights.size()) != batch_size ||
        static_cast<int>(frame_widths.size()) != batch_size) {
        throw std::invalid_argument(
            "frames, frame_heights, and frame_widths must have the same size");
    }

    if (batch_size > MAX_BATCH_SIZE) {
        throw std::invalid_argument(
            "Batch size exceeds maximum supported (" +
            std::to_string(MAX_BATCH_SIZE) + ")");
    }

    if (target_h <= 0 || target_w <= 0) {
        throw std::invalid_argument("Target dimensions must be positive");
    }

    // Calculate output size: N × C × H × W
    const size_t output_size = static_cast<size_t>(batch_size) * 3 *
                               static_cast<size_t>(target_h) *
                               static_cast<size_t>(target_w);

    // Allocate device memory for output tensor
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    // Allocate device memory for each input frame and copy data
    std::vector<uint8_t*> d_frames(batch_size, nullptr);
    std::vector<FrameDesc> h_frame_descs(batch_size);

    for (int i = 0; i < batch_size; i++) {
        const size_t frame_bytes = static_cast<size_t>(frame_heights[i]) *
                                   static_cast<size_t>(frame_widths[i]) * 3;

        CUDA_CHECK(cudaMalloc(&d_frames[i], frame_bytes));
        CUDA_CHECK(cudaMemcpy(d_frames[i], frames[i], frame_bytes,
                              cudaMemcpyHostToDevice));

        h_frame_descs[i].data = d_frames[i];
        h_frame_descs[i].height = frame_heights[i];
        h_frame_descs[i].width = frame_widths[i];
    }

    // Allocate and copy frame descriptors to device
    FrameDesc* d_frame_descs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frame_descs, batch_size * sizeof(FrameDesc)));
    CUDA_CHECK(cudaMemcpy(d_frame_descs, h_frame_descs.data(),
                          batch_size * sizeof(FrameDesc),
                          cudaMemcpyHostToDevice));

    // Configure kernel launch dimensions
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    dim3 grid(
        (target_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (target_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size
    );

    // Launch preprocessing kernel
    preprocess_kernel<<<grid, block>>>(
        d_frame_descs, d_output, target_h, target_w, normalize
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure kernel completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<float> output(output_size);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_output);
    cudaFree(d_frame_descs);
    for (int i = 0; i < batch_size; i++) {
        cudaFree(d_frames[i]);
    }

    return output;
}

}  // namespace metropolis
