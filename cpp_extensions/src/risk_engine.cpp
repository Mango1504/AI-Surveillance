/**
 * @file risk_engine.cpp
 * @brief C++ implementation of exponential-recency-weighted risk score computation.
 *
 * Provides sub-millisecond risk scoring using exponential decay weighting
 * over a sliding time window. Events closer to the current time contribute
 * more to the overall risk score.
 *
 * The risk score is computed as:
 *   score = sum(weight_i * exp(-(current_time - timestamp_i) / tau)) / normalizer
 *
 * where tau = window_secs / 3.0 controls the decay rate within the time window.
 * The normalizer is the sum of all weights (representing the maximum possible
 * score if all events occurred at current_time). The final result is clamped
 * to [0.0, 1.0].
 */

#include "metropolis_cpp.h"
#include <cmath>
#include <algorithm>

namespace metropolis {

float compute_risk_score(
    const double* timestamps,
    const double* weights,
    int num_events,
    double window_secs,
    double current_time
) {
    // Edge case: no events
    if (num_events <= 0) {
        return 0.0f;
    }

    // Edge case: invalid window
    if (window_secs <= 0.0) {
        return 0.0f;
    }

    // Decay time constant: tau = window_secs / 3.0
    // This means events at the window boundary have decayed to exp(-3) ≈ 5%
    const double tau = window_secs / 3.0;

    // Window boundary
    const double window_start = current_time - window_secs;

    double score = 0.0;
    double normalization_factor = 0.0;

    for (int i = 0; i < num_events; ++i) {
        const double timestamp = timestamps[i];
        const double weight = weights[i];

        // Accumulate normalization factor from all weights
        // (max possible score if all events were at current_time)
        normalization_factor += weight;

        // Only include events within the time window
        if (timestamp < window_start || timestamp > current_time) {
            continue;
        }

        // Compute time delta and exponential decay
        const double time_delta = current_time - timestamp;
        const double decay = std::exp(-time_delta / tau);

        // Accumulate weighted, decayed score
        score += weight * decay;
    }

    // Edge case: all weights are zero or negative normalization
    if (normalization_factor <= 0.0) {
        return 0.0f;
    }

    // Normalize to [0, 1] range
    double result = score / normalization_factor;

    // Clamp to [0.0, 1.0]
    result = std::min(1.0, std::max(0.0, result));

    return static_cast<float>(result);
}

}  // namespace metropolis
