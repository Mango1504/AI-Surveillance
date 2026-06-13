"""Kalman filter wrapper for bounding box position and velocity state estimation.

Provides a KalmanBoxTracker class that wraps filterpy's KalmanFilter to maintain
per-track state estimation using a constant velocity motion model. Falls back to
a numpy-based implementation if filterpy is not available.

State vector: [cx, cy, s, r, dx, dy, ds]
    cx, cy = center x, y of bounding box
    s = area (width * height)
    r = aspect ratio (width / height)
    dx, dy = velocity of center
    ds = rate of change of area

Measurement vector: [cx, cy, s, r] (from detected bounding box)

Validates: Requirements 5.4
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Try to import filterpy; fall back to numpy-based implementation if unavailable
try:
    from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter

    _HAS_FILTERPY = True
    logger.debug("filterpy available, using FilterPyKalmanFilter backend.")
except ImportError:
    _HAS_FILTERPY = False
    logger.warning(
        "filterpy not available. Using numpy-based Kalman filter fallback."
    )


def bbox_to_state(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Convert bounding box (x1, y1, x2, y2) to state measurement [cx, cy, s, r].

    Args:
        bbox: Bounding box as (x1, y1, x2, y2) pixel coordinates.

    Returns:
        Measurement vector as shape (4, 1) numpy array [cx, cy, s, r].
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h  # area
    r = w / h if h > 0 else 1.0  # aspect ratio
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def state_to_bbox(state: np.ndarray) -> tuple[int, int, int, int]:
    """Convert state vector [cx, cy, s, r, ...] to bounding box (x1, y1, x2, y2).

    Args:
        state: State vector with at least 4 elements [cx, cy, s, r, ...].

    Returns:
        Bounding box as (x1, y1, x2, y2) integer pixel coordinates.
    """
    cx = state[0, 0]
    cy = state[1, 0]
    s = state[2, 0]
    r = state[3, 0]

    # Clamp area to be positive
    s = max(s, 1.0)
    # Clamp aspect ratio to be positive
    r = max(r, 0.01)

    w = np.sqrt(s * r)
    h = s / w if w > 0 else np.sqrt(s)

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


class _NumpyKalmanFilter:
    """Minimal Kalman filter implementation using numpy as a fallback.

    Implements the standard linear Kalman filter predict/update cycle
    for the constant velocity bounding box tracking model.
    """

    def __init__(self, dim_x: int, dim_z: int) -> None:
        """Initialize filter dimensions and matrices.

        Args:
            dim_x: State dimension.
            dim_z: Measurement dimension.
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State vector
        self.x = np.zeros((dim_x, 1), dtype=np.float64)

        # State covariance matrix
        self.P = np.eye(dim_x, dtype=np.float64)

        # State transition matrix
        self.F = np.eye(dim_x, dtype=np.float64)

        # Measurement matrix
        self.H = np.zeros((dim_z, dim_x), dtype=np.float64)

        # Measurement noise covariance
        self.R = np.eye(dim_z, dtype=np.float64)

        # Process noise covariance
        self.Q = np.eye(dim_x, dtype=np.float64)

    def predict(self) -> None:
        """Predict the next state using the motion model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """Update state with a new measurement.

        Args:
            z: Measurement vector of shape (dim_z, 1).
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.x = self.x + K @ y

        # Updated covariance estimate
        I = np.eye(self.dim_x, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P


class KalmanBoxTracker:
    """Kalman filter wrapper for bounding box tracking with position/velocity estimation.

    Uses a 7-dimensional state vector [cx, cy, s, r, dx, dy, ds] with a constant
    velocity motion model. Wraps filterpy's KalmanFilter when available, otherwise
    uses a numpy-based fallback implementation.

    Args:
        bbox: Initial bounding box as (x1, y1, x2, y2) pixel coordinates.
    """

    # Class-level counter for unique tracker IDs
    _count: int = 0

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        """Initialize Kalman filter with the first detection bounding box.

        Sets up the state transition matrix (constant velocity model),
        measurement matrix, and noise covariance matrices.

        Args:
            bbox: Initial bounding box as (x1, y1, x2, y2) pixel coordinates.
        """
        # Dimensions
        dim_x = 7  # [cx, cy, s, r, dx, dy, ds]
        dim_z = 4  # [cx, cy, s, r]

        # Create the underlying Kalman filter
        if _HAS_FILTERPY:
            self._kf = FilterPyKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        else:
            self._kf = _NumpyKalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # State transition matrix F (constant velocity model)
        # State: [cx, cy, s, r, dx, dy, ds]
        # Next state: cx' = cx + dx, cy' = cy + dy, s' = s + ds, r' = r
        self._kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],  # cx' = cx + dx
                [0, 1, 0, 0, 0, 1, 0],  # cy' = cy + dy
                [0, 0, 1, 0, 0, 0, 1],  # s'  = s + ds
                [0, 0, 0, 1, 0, 0, 0],  # r'  = r (constant)
                [0, 0, 0, 0, 1, 0, 0],  # dx' = dx
                [0, 0, 0, 0, 0, 1, 0],  # dy' = dy
                [0, 0, 0, 0, 0, 0, 1],  # ds' = ds
            ],
            dtype=np.float64,
        )

        # Measurement matrix H (we observe [cx, cy, s, r])
        self._kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        # Measurement noise covariance R
        # Position measurements are more reliable than area/ratio
        self._kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        # Initial state covariance P
        # High uncertainty for velocity components (unobserved initially)
        self._kf.P = np.diag(
            [10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0]
        ).astype(np.float64)

        # Process noise covariance Q
        # Small noise for position/size, larger for velocity
        self._kf.Q = np.diag(
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]
        ).astype(np.float64)

        # Initialize state with first measurement
        measurement = bbox_to_state(bbox)
        self._kf.x[:4] = measurement  # Position/size from bbox
        self._kf.x[4:] = 0.0  # Zero initial velocity

        # Track prediction count (number of predict calls without update)
        self._time_since_update: int = 0
        self._hits: int = 0
        self._age: int = 0

        # Assign unique ID
        KalmanBoxTracker._count += 1
        self.id: int = KalmanBoxTracker._count

    def predict(self) -> tuple[int, int, int, int]:
        """Predict the next bounding box state using the motion model.

        Advances the Kalman filter state by one time step using the
        constant velocity model. Clamps area to be positive after prediction.

        Returns:
            Predicted bounding box as (x1, y1, x2, y2) integer pixel coordinates.
        """
        # Prevent area from going negative
        if self._kf.x[2, 0] + self._kf.x[6, 0] <= 0:
            self._kf.x[6, 0] = 0.0

        self._kf.predict()
        self._age += 1
        self._time_since_update += 1

        return state_to_bbox(self._kf.x)

    def update(self, bbox: tuple[int, int, int, int]) -> None:
        """Update the Kalman filter state with a new detection measurement.

        Args:
            bbox: Detected bounding box as (x1, y1, x2, y2) pixel coordinates.
        """
        measurement = bbox_to_state(bbox)
        self._kf.update(measurement)
        self._time_since_update = 0
        self._hits += 1

    def get_state(self) -> tuple[int, int, int, int]:
        """Get the current estimated bounding box from the filter state.

        Returns:
            Current estimated bounding box as (x1, y1, x2, y2) integer pixel coordinates.
        """
        return state_to_bbox(self._kf.x)

    def get_velocity(self) -> tuple[float, float]:
        """Get the estimated velocity of the bounding box center.

        Returns:
            Estimated velocity as (dx, dy) in pixels per frame.
        """
        dx = float(self._kf.x[4, 0])
        dy = float(self._kf.x[5, 0])
        return (dx, dy)

    @property
    def time_since_update(self) -> int:
        """Number of frames since the last measurement update."""
        return self._time_since_update

    @property
    def hits(self) -> int:
        """Number of times this tracker has been updated with a measurement."""
        return self._hits

    @property
    def age(self) -> int:
        """Total number of frames since this tracker was created."""
        return self._age
