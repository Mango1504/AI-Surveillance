"""NVIDIA Metropolis integration components for the AI Surveillance system.

This package provides Metropolis-aligned components including TensorRT model
optimization, DeepStream video pipelines, Triton Inference Server integration,
multi-camera tracking, structured analytics metadata, event streaming, and
C++ performance extensions.

The architecture preserves the existing Python pipeline as a fallback while
introducing high-performance NVIDIA GPU-accelerated alternatives selectable
via configuration.
"""

from .association import bytetrack_associate, compute_iou_cost_matrix, iou_batch
from .config import MetropolisConfig
from .embedding import EMBEDDING_DIM, EmbeddingExtractor
from .kalman_filter import KalmanBoxTracker
from .orchestrator import Capabilities, Detection, PipelineOrchestrator

__all__ = [
    "Capabilities",
    "CPP_EXTENSIONS_AVAILABLE",
    "Detection",
    "EMBEDDING_DIM",
    "EmbeddingExtractor",
    "KalmanBoxTracker",
    "MetropolisConfig",
    "PipelineOrchestrator",
    "batched_nms",
    "bytetrack_associate",
    "compute_iou_cost_matrix",
    "compute_risk_score",
    "cuda_preprocess",
    "iou_batch",
]

# C++ extension import with Python fallback.
# The compiled metropolis_cpp module provides CUDA-accelerated implementations.
# If it cannot be loaded (missing .so/.pyd, ABI mismatch, no CUDA runtime),
# we fall back to equivalent pure-Python/NumPy implementations from cpp_fallback.
import logging as _logging

_logger = _logging.getLogger(__name__)

try:
    from metropolis_cpp import batched_nms, compute_risk_score, cuda_preprocess

    CPP_EXTENSIONS_AVAILABLE = True
    _logger.debug("C++ extensions loaded successfully (metropolis_cpp)")
except ImportError:
    _logger.warning(
        "C++ extension module 'metropolis_cpp' could not be loaded. "
        "Falling back to Python implementations. Performance may be reduced."
    )
    from .cpp_fallback import batched_nms, compute_risk_score, cuda_preprocess

    CPP_EXTENSIONS_AVAILABLE = False
