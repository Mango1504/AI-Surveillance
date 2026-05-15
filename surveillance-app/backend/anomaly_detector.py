from behavior import BehaviorAnalyzer, GazeTracker


class AnomalyDetector:
    def __init__(self):
        self.behavior = BehaviorAnalyzer()
        self.gaze = GazeTracker()

    def analyze(self, frame, tracking_id="default", keypoints=None):
        movement_score = self.behavior.analyze_pose(tracking_id, keypoints or [])
        looking_away, duration = self.gaze.analyze_gaze(frame)
        gaze_score = min(1.0, duration / 3.0) if looking_away else 0.0
        return {
            "movement_score": movement_score,
            "gaze_score": gaze_score,
            "aggregate_score": max(movement_score, gaze_score),
            "looking_away": looking_away,
            "looking_away_duration": duration,
        }
