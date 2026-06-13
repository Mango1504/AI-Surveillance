/**
 * @file nms.cu
 * @brief CUDA-accelerated batched Non-Maximum Suppression (NMS).
 *
 * Implements GPU-accelerated NMS that produces identical results to the
 * Python reference implementation (torchvision.ops.nms). Supports:
 * - Batched NMS with per-class suppression
 * - Configurable IoU threshold
 * - Score threshold filtering
 * - Class-aware suppression (only suppress within the same class)
 *
 * Algorithm:
 *   1. Filter boxes by score_threshold on the host
 *   2. Sort remaining boxes by score (descending)
 *   3. Launch CUDA kernel to compute pairwise IoU in a 2D grid
 *   4. Each thread computes IoU between box i and box j, sets suppression bit
 *   5. Iterate through bitmask on host to collect kept indices
 *
 * The bitmask approach uses one bit per box pair, stored in uint64_t words.
 * For N boxes, the suppression mask is an N × ceil(N/64) matrix where
 * mask[i][j/64] bit (j%64) indicates box j suppresses box i.
 */

#include "metropolis_cpp.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>

// Number of threads per block dimension for the NMS IoU kernel
#define NMS_BLOCK_SIZE 64

// Maximum number of boxes after score filtering (for device memory limits)
#define MAX_NMS_BOXES 4096

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
 * @brief Compute Intersection over Union (IoU) between two boxes.
 *
 * Boxes are in xyxy format: (x1, y1, x2, y2).
 *
 * @param box_a Pointer to first box (4 floats: x1, y1, x2, y2)
 * @param box_b Pointer to second box (4 floats: x1, y1, x2, y2)
 * @return IoU value in [0, 1]
 */
__device__ __forceinline__ float compute_iou(
    const float* box_a,
    const float* box_b
) {
    // Intersection coordinates
    const float inter_x1 = fmaxf(box_a[0], box_b[0]);
    const float inter_y1 = fmaxf(box_a[1], box_b[1]);
    const float inter_x2 = fminf(box_a[2], box_b[2]);
    const float inter_y2 = fminf(box_a[3], box_b[3]);

    // Intersection area (zero if no overlap)
    const float inter_w = fmaxf(0.0f, inter_x2 - inter_x1);
    const float inter_h = fmaxf(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    // Union area
    const float area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    const float area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
    const float union_area = area_a + area_b - inter_area;

    // Avoid division by zero
    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

/**
 * @brief CUDA kernel for computing pairwise IoU and building suppression mask.
 *
 * Each block processes a tile of box pairs. Block (bx, by) computes IoU
 * between boxes in row-block bx and column-block by. Within each block,
 * thread tid computes IoU between box (bx * NMS_BLOCK_SIZE + tid) and
 * each box in the column block, building a bitmask of suppressions.
 *
 * For the NMS algorithm to work correctly:
 * - Only box j can suppress box i if j has higher score (j < i in sorted order)
 * - Two boxes must be of the same class to suppress each other
 *
 * Grid layout:
 *   - gridDim.x = ceil(num_boxes / NMS_BLOCK_SIZE)  (row blocks)
 *   - gridDim.y = ceil(num_boxes / NMS_BLOCK_SIZE)  (column blocks)
 *   - blockDim.x = NMS_BLOCK_SIZE
 *
 * Output mask layout:
 *   mask[i * num_col_blocks + col_block] contains bits for boxes
 *   [col_block * 64 .. col_block * 64 + 63] that suppress box i.
 *
 * @param boxes       Sorted boxes in xyxy format, shape (N, 4)
 * @param classes     Class indices for each sorted box, shape (N,)
 * @param num_boxes   Total number of boxes
 * @param iou_thresh  IoU threshold for suppression
 * @param mask        Output suppression bitmask, shape (N, num_col_blocks)
 */
__global__ void nms_kernel(
    const float* __restrict__ boxes,
    const int32_t* __restrict__ classes,
    int num_boxes,
    float iou_thresh,
    unsigned long long* __restrict__ mask
) {
    // Row block index: which set of boxes we're checking for suppression
    const int row_block = blockIdx.x;
    // Column block index: which set of boxes might suppress them
    const int col_block = blockIdx.y;

    // The box index in the row that this thread handles
    const int row_idx = row_block * NMS_BLOCK_SIZE + threadIdx.x;

    // Early exit if this thread's row box is out of range
    if (row_idx >= num_boxes) {
        return;
    }

    // Number of column blocks
    const int num_col_blocks = (num_boxes + NMS_BLOCK_SIZE - 1) / NMS_BLOCK_SIZE;

    // Load the row box into registers
    const float row_box[4] = {
        boxes[row_idx * 4 + 0],
        boxes[row_idx * 4 + 1],
        boxes[row_idx * 4 + 2],
        boxes[row_idx * 4 + 3]
    };
    const int32_t row_class = classes[row_idx];

    // Load column block boxes into shared memory for coalesced access
    __shared__ float shared_boxes[NMS_BLOCK_SIZE * 4];
    __shared__ int32_t shared_classes[NMS_BLOCK_SIZE];

    const int col_start = col_block * NMS_BLOCK_SIZE;
    const int col_idx = col_start + threadIdx.x;

    // Each thread loads one box from the column block into shared memory
    if (col_idx < num_boxes) {
        shared_boxes[threadIdx.x * 4 + 0] = boxes[col_idx * 4 + 0];
        shared_boxes[threadIdx.x * 4 + 1] = boxes[col_idx * 4 + 1];
        shared_boxes[threadIdx.x * 4 + 2] = boxes[col_idx * 4 + 2];
        shared_boxes[threadIdx.x * 4 + 3] = boxes[col_idx * 4 + 3];
        shared_classes[threadIdx.x] = classes[col_idx];
    }
    __syncthreads();

    // Compute IoU between this thread's row box and all column boxes
    // Build a 64-bit suppression mask for this row box against this col block
    unsigned long long suppression_bits = 0ULL;

    const int col_end = min(col_start + NMS_BLOCK_SIZE, num_boxes);
    const int num_cols_in_block = col_end - col_start;

    for (int j = 0; j < num_cols_in_block; j++) {
        const int col_abs_idx = col_start + j;

        // Only a higher-scored box (lower index) can suppress a lower-scored box
        // In sorted order, index col_abs_idx < row_idx means col has higher score
        if (col_abs_idx >= row_idx) {
            continue;
        }

        // Only suppress within the same class (per-class NMS)
        if (shared_classes[j] != row_class) {
            continue;
        }

        // Compute IoU
        const float col_box[4] = {
            shared_boxes[j * 4 + 0],
            shared_boxes[j * 4 + 1],
            shared_boxes[j * 4 + 2],
            shared_boxes[j * 4 + 3]
        };

        const float iou = compute_iou(row_box, col_box);

        // Set suppression bit if IoU exceeds threshold
        if (iou > iou_thresh) {
            suppression_bits |= (1ULL << j);
        }
    }

    // Write the suppression mask for this row box and column block
    mask[row_idx * num_col_blocks + col_block] = suppression_bits;
}

namespace metropolis {

/**
 * @brief GPU-accelerated batched Non-Maximum Suppression.
 *
 * Host function that orchestrates the CUDA NMS pipeline:
 *   1. Filter boxes by score_threshold
 *   2. Sort remaining boxes by score (descending)
 *   3. Allocate device memory for boxes, classes, and suppression mask
 *   4. Launch NMS kernel to compute pairwise IoU and suppression bits
 *   5. Copy mask back to host and iterate to collect kept indices
 *   6. Map kept indices back to original box indices
 *
 * The greedy NMS algorithm processes boxes in score order:
 * for each box i (highest score first), if it hasn't been suppressed
 * by any previously kept box, keep it and mark all lower-scored boxes
 * with IoU > threshold as suppressed.
 *
 * @param boxes          Bounding boxes in (x1, y1, x2, y2) format, shape (N, 4)
 * @param scores         Confidence scores, shape (N,)
 * @param classes        Class indices, shape (N,)
 * @param num_boxes      Total number of input boxes
 * @param iou_threshold  IoU threshold for suppression (default: 0.45)
 * @param score_threshold Minimum score to keep (default: 0.25)
 * @return Indices of kept boxes (in original input order)
 *
 * @throws std::invalid_argument if num_boxes is negative
 * @throws std::runtime_error on CUDA allocation or kernel launch failure
 */
std::vector<int> batched_nms(
    const float* boxes,
    const float* scores,
    const int32_t* classes,
    int num_boxes,
    float iou_threshold,
    float score_threshold
) {
    // Handle edge cases
    if (num_boxes <= 0) {
        return {};
    }

    // Step 1: Filter boxes by score threshold and build sorted index
    // Collect indices of boxes that pass the score threshold
    std::vector<int> candidate_indices;
    candidate_indices.reserve(num_boxes);

    for (int i = 0; i < num_boxes; i++) {
        if (scores[i] >= score_threshold) {
            candidate_indices.push_back(i);
        }
    }

    if (candidate_indices.empty()) {
        return {};
    }

    // Step 2: Sort candidates by score (descending)
    std::sort(candidate_indices.begin(), candidate_indices.end(),
              [&scores](int a, int b) {
                  return scores[a] > scores[b];
              });

    const int n = static_cast<int>(candidate_indices.size());

    // For very small inputs, use CPU-only NMS (no kernel launch overhead)
    if (n == 1) {
        return {candidate_indices[0]};
    }

    // Step 3: Prepare sorted box and class arrays for the kernel
    std::vector<float> sorted_boxes(n * 4);
    std::vector<int32_t> sorted_classes(n);

    for (int i = 0; i < n; i++) {
        const int orig_idx = candidate_indices[i];
        sorted_boxes[i * 4 + 0] = boxes[orig_idx * 4 + 0];
        sorted_boxes[i * 4 + 1] = boxes[orig_idx * 4 + 1];
        sorted_boxes[i * 4 + 2] = boxes[orig_idx * 4 + 2];
        sorted_boxes[i * 4 + 3] = boxes[orig_idx * 4 + 3];
        sorted_classes[i] = classes[orig_idx];
    }

    // Step 4: Allocate device memory
    const int num_col_blocks = (n + NMS_BLOCK_SIZE - 1) / NMS_BLOCK_SIZE;
    const size_t mask_size = static_cast<size_t>(n) * num_col_blocks;

    float* d_boxes = nullptr;
    int32_t* d_classes = nullptr;
    unsigned long long* d_mask = nullptr;

    CUDA_CHECK(cudaMalloc(&d_boxes, n * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_classes, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_mask, mask_size * sizeof(unsigned long long)));

    // Initialize mask to zero (no suppressions)
    CUDA_CHECK(cudaMemset(d_mask, 0, mask_size * sizeof(unsigned long long)));

    // Copy sorted data to device
    CUDA_CHECK(cudaMemcpy(d_boxes, sorted_boxes.data(),
                          n * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_classes, sorted_classes.data(),
                          n * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Step 5: Launch NMS kernel
    // Grid: (num_row_blocks, num_col_blocks) where each block has NMS_BLOCK_SIZE threads
    const int num_row_blocks = (n + NMS_BLOCK_SIZE - 1) / NMS_BLOCK_SIZE;

    dim3 grid(num_row_blocks, num_col_blocks);
    dim3 block(NMS_BLOCK_SIZE);

    nms_kernel<<<grid, block>>>(
        d_boxes, d_classes, n, iou_threshold, d_mask
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 6: Copy mask back to host
    std::vector<unsigned long long> h_mask(mask_size);
    CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask,
                          mask_size * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_boxes);
    cudaFree(d_classes);
    cudaFree(d_mask);

    // Step 7: Greedy NMS using the suppression mask
    // Process boxes in score order. A box is kept if none of the
    // previously kept boxes have suppressed it.
    //
    // We maintain a "removed" bitmask per column block that accumulates
    // the suppression bits from all kept boxes.
    std::vector<unsigned long long> removed(num_col_blocks, 0ULL);
    std::vector<int> kept_indices;
    kept_indices.reserve(n);

    for (int i = 0; i < n; i++) {
        // Check if box i has been suppressed by any previously kept box
        const int block_idx = i / NMS_BLOCK_SIZE;
        const int bit_idx = i % NMS_BLOCK_SIZE;

        if (removed[block_idx] & (1ULL << bit_idx)) {
            // This box was suppressed, skip it
            continue;
        }

        // Keep this box
        kept_indices.push_back(candidate_indices[i]);

        // Mark all boxes that this kept box suppresses
        // Box i's suppression mask tells us which higher-scored boxes suppress it,
        // but we need the reverse: which lower-scored boxes does box i suppress.
        // The mask stores mask[j][col_block_of_i] bit (i % 64) set means
        // box i (higher score) suppresses box j (lower score).
        // Actually, our kernel stores: mask[row][col_block] bit j means
        // col_block*64+j suppresses row. So to find what box i suppresses,
        // we look at all rows j > i and check if mask[j] has bit i set.
        //
        // More efficiently: we OR the suppression info into the removed mask.
        // For each subsequent box j, if mask[j] has bit i set in the appropriate
        // col_block, then j is suppressed by i.
        //
        // Alternative approach: iterate all subsequent boxes and check their mask.
        // But that's O(N^2) on host. Instead, we use the following approach:
        // For each box j > i, check mask[j][block_of_i] & (1 << (i%64)).
        // If set, mark j as removed.
        //
        // Optimized: scan all rows and OR their mask entries where bit i is set.
        // Actually the simplest correct approach: for each subsequent box j,
        // if it has bit i set in its mask, it means i suppresses j.
        const int i_block = i / NMS_BLOCK_SIZE;
        const int i_bit = i % NMS_BLOCK_SIZE;
        const unsigned long long i_mask_bit = 1ULL << i_bit;

        for (int j = i + 1; j < n; j++) {
            // Check if box i suppresses box j
            // mask[j][i_block] bit i_bit means box i suppresses box j
            if (h_mask[j * num_col_blocks + i_block] & i_mask_bit) {
                // Mark box j as removed
                const int j_block = j / NMS_BLOCK_SIZE;
                const int j_bit = j % NMS_BLOCK_SIZE;
                removed[j_block] |= (1ULL << j_bit);
            }
        }
    }

    return kept_indices;
}

}  // namespace metropolis
