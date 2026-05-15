import numpy as np
import collections
import cv2
import time
from config import get_config

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except (ImportError, AttributeError):
    mp = None
    mp_face_mesh = None
    print("[WARNING] MediaPipe not installed or broken. Gaze tracking will degrade to Haar.")

class BehaviorAnalyzer:
    """Spatio-Temporal Behavior Analysis via Pose Tracking."""
    def __init__(self, history_size=30):
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=history_size))
        
    def analyze_pose(self, tracking_id, keypoints):
        self.history[tracking_id].append(keypoints)
        if len(self.history[tracking_id]) < 10:
            return 0.0
        nose_xs = [kp[0][0] for kp in self.history[tracking_id] if len(kp) > 0]
        if len(nose_xs) < 10:
            return 0.0
        std_dev = np.std(nose_xs)
        risk_score = min(1.0, std_dev / 50.0) 
        return risk_score


class MotionAnalyzer:
    """Frame-difference movement detector for unusual large movements."""

    def __init__(self, min_changed_ratio=0.12, trigger_seconds=2.0):
        self.prev_gray = None
        self.movement_start = None
        self.min_changed_ratio = min_changed_ratio
        self.trigger_seconds = trigger_seconds

    def analyze(self, frame_bgr):
        small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, thresh = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        changed_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
        score = min(1.0, changed_ratio / max(self.min_changed_ratio, 0.001))

        if changed_ratio >= self.min_changed_ratio:
            if self.movement_start is None:
                self.movement_start = time.time()
            if time.time() - self.movement_start >= self.trigger_seconds:
                return True, score
        else:
            self.movement_start = None

        return False, score

class GazeTracker:
    def __init__(self):
        self.config = get_config()
        self.looking_away_start = None
        self.call_counter = 0
        
        self.use_mp = mp is not None and mp_face_mesh is not None
        if self.use_mp:
            refine = (self.config.profile.tier in ["ULTRA", "HIGH"])
            conf = 0.5 if refine else (0.4 if self.config.profile.tier == "MID" else 0.3)
            
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.config.MAX_FACES_PER_FRAME or 5,
                refine_landmarks=refine,
                min_detection_confidence=conf,
                min_tracking_confidence=conf
            )
        else:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_gaze(self, frame_bgr):
        """
        Returns (is_looking_away, duration_looking_away).
        """
        if not self.config.ENABLE_GAZE:
            return False, 0.0
            
        self.call_counter += 1
        if self.config.profile.tier == "LOW" and self.call_counter % 2 != 0:
            return False, 0.0
            
        if self.use_mp:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                xs = [lm.x for lm in landmarks]
                face_center_x = sum(xs) / len(xs)
                nose_x = landmarks[1].x if len(landmarks) > 1 else face_center_x
                yaw_proxy = abs(nose_x - face_center_x)
                off_center = face_center_x < 0.28 or face_center_x > 0.72
                looking_away = yaw_proxy > 0.07 or off_center

                if looking_away:
                    if self.looking_away_start is None:
                        self.looking_away_start = time.time()
                    return True, time.time() - self.looking_away_start

                self.looking_away_start = None
                return False, 0.0
            else:
                if self.looking_away_start is None:
                    self.looking_away_start = time.time()
                duration = time.time() - self.looking_away_start
                return True, duration
        else:
            # Fallback Haar
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                for (x,y,w,h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    
                    if len(eyes) >= 2:
                        self.looking_away_start = None
                        return False, 0.0
                        
                if self.looking_away_start is None:
                    self.looking_away_start = time.time()
                duration = time.time() - self.looking_away_start
                return True, duration

            self.looking_away_start = None
            return False, 0.0
