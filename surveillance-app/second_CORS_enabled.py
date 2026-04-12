"""
Phone Detection System — detector.py [UPDATED FOR REACT FRONTEND]
Detects phones via YOLOv8, maps to grid cells, serves data via Flask API.
React web app connects to the Flask endpoints defined at the bottom.

CORS enabled for frontend communication.
"""

import cv2
import time
import os
import threading
import json
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS
from examinee_manager import get_examinee_manager

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CAMERA_INDEX   = 0          # 0 = default webcam; use RTSP/IP URL for IP cam
SAVE_PATH      = r"C:/Users/write/Desktop/Phone"
GRID_ROWS      = 3          # divide frame into this many rows
GRID_COLS      = 4          # divide frame into this many columns
PHONE_CLASS_ID = 67         # YOLOv8 COCO class 67 = cell phone
CONFIDENCE_MIN = 0.45       # ignore detections below this confidence
FLASK_HOST     = "0.0.0.0"  # 0.0.0.0 = reachable from other devices on LAN
FLASK_PORT     = 5000
POST_DETECT_RECORD_SECS = 5 # keep recording N seconds after phone disappears

os.makedirs(SAVE_PATH, exist_ok=True)

# Initialize examinee manager
examinee_manager = get_examinee_manager()

# ──────────────────────────────────────────────
# SHARED STATE  (written by detector, read by Flask)
# ──────────────────────────────────────────────
state_lock = threading.Lock()
shared_state = {
    "phone_detected": False,
    "detections": [],        # list of detection dicts (see below)
    "frame_width": 0,
    "frame_height": 0,
    "grid_rows": GRID_ROWS,
    "grid_cols": GRID_COLS,
    "timestamp": None,
    "recording": False,
    "clip_path": None,
    "current_exam_hall": 1,
    "exam_hall_info": {},
    "violations": [],        # list of detected violations
}

# Latest annotated frame as JPEG bytes (for MJPEG stream)
frame_lock  = threading.Lock()
latest_jpeg = None


# ──────────────────────────────────────────────
# VIOLATION LOGGING
# ──────────────────────────────────────────────
class ViolationLogger:
    def __init__(self):
        self.violations = []
        self.log_file = os.path.join(SAVE_PATH, "violations.log")
    
    def log_violation(self, exam_hall, row, col, confidence, timestamp):
        violation = {
            "timestamp": timestamp,
            "exam_hall": exam_hall,
            "grid_position": f"R{row}C{col}",
            "confidence": confidence,
            "status": "VIOLATION - Unauthorized Phone"
        }
        self.violations.append(violation)
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] ExamHall {exam_hall} - Phone at R{row}C{col} (Conf: {confidence:.2f})\n")
        except:
            pass
        return violation
    
    def get_violations(self):
        return self.violations

violation_logger = ViolationLogger()


# ──────────────────────────────────────────────
# GRID HELPERS
# ──────────────────────────────────────────────
def bbox_to_grid(cx, cy, frame_w, frame_h, rows, cols):
    """Return (row, col) 1-indexed for the grid cell the centre point falls in."""
    col = min(int(cx / frame_w * cols) + 1, cols)
    row = min(int(cy / frame_h * rows) + 1, rows)
    return row, col


def draw_grid(frame, rows, cols, color=(80, 80, 80), thickness=1):
    h, w = frame.shape[:2]
    for r in range(1, rows):
        y = int(h * r / rows)
        cv2.line(frame, (0, y), (w, y), color, thickness)
    for c in range(1, cols):
        x = int(w * c / cols)
        cv2.line(frame, (x, 0), (x, h), color, thickness)
    # Row / col labels in each cell
    cell_w, cell_h = w // cols, h // rows
    for r in range(rows):
        for c in range(cols):
            label = f"R{r+1}C{c+1}"
            px = c * cell_w + 6
            py = r * cell_h + 18
            cv2.putText(frame, label, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def highlight_cell(frame, row, col, rows, cols, color=(0, 255, 120), alpha=0.25):
    """Tint the grid cell that contains a detected phone."""
    h, w = frame.shape[:2]
    cell_w, cell_h = w // cols, h // rows
    x1 = (col - 1) * cell_w
    y1 = (row - 1) * cell_h
    x2 = x1 + cell_w
    y2 = y1 + cell_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


# ──────────────────────────────────────────────
# RECORDING HELPER
# ──────────────────────────────────────────────
class ClipRecorder:
    def __init__(self):
        self.writer   = None
        self.active   = False
        self.path     = None
        self._last_detect = 0

    def start(self, frame_shape):
        h, w = frame_shape[:2]
        fourcc   = cv2.VideoWriter_fourcc(*"XVID")
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(SAVE_PATH, f"phone_{ts}.avi")
        self.writer = cv2.VideoWriter(self.path, fourcc, 20.0, (w, h))
        self.active  = True
        print(f"[REC] Started → {self.path}")

    def write(self, frame):
        if self.active and self.writer:
            self.writer.write(frame)

    def touch(self):
        """Reset the idle timer when a phone is still in frame."""
        self._last_detect = time.time()

    def maybe_stop(self):
        """Stop recording if phone has been gone for POST_DETECT_RECORD_SECS."""
        if self.active and (time.time() - self._last_detect) > POST_DETECT_RECORD_SECS:
            self.stop()

    def stop(self):
        if self.writer:
            self.writer.release()
        self.active = False
        print(f"[REC] Stopped → {self.path}")


# ──────────────────────────────────────────────
# DETECTION LOOP  (runs in its own thread)
# ──────────────────────────────────────────────
def detection_loop():
    global latest_jpeg, shared_state

    model   = YOLO("yolov8n.pt")
    cap     = cv2.VideoCapture(CAMERA_INDEX)
    recorder = ClipRecorder()

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    ret, test_frame = cap.read()
    frame_h, frame_w = test_frame.shape[:2] if ret else (480, 640)

    with state_lock:
        shared_state["frame_width"]  = frame_w
        shared_state["frame_height"] = frame_h

    print(f"[CAM] {frame_w}x{frame_h}  grid={GRID_ROWS}x{GRID_COLS}")
    
    # Display examinee statistics
    stats = examinee_manager.get_stats()
    print(f"[EXAM] Total Examinees: {stats['total_examinees']}")
    for hall, count in stats['by_hall'].items():
        print(f"       ExamHall {hall}: {count} examinees")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        results        = model(frame, verbose=False)
        detections     = []
        phone_in_frame = False

        for result in results:
            boxes = result.boxes
            for i, cls in enumerate(boxes.cls.cpu().numpy().astype(int)):
                if cls != PHONE_CLASS_ID:
                    continue
                conf = float(boxes.conf[i].cpu())
                if conf < CONFIDENCE_MIN:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                row, col = bbox_to_grid(cx, cy, frame_w, frame_h,
                                        GRID_ROWS, GRID_COLS)

                detections.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "center":     [int(cx), int(cy)],
                    "confidence": round(conf, 3),
                    "grid_row":   row,
                    "grid_col":   col,
                    "label":      f"Row {row}, Col {col}",
                })
                phone_in_frame = True

        # ── Update shared state ──
        with state_lock:
            shared_state["phone_detected"] = phone_in_frame
            shared_state["detections"]     = detections
            shared_state["timestamp"]      = datetime.utcnow().isoformat() + "Z"
            shared_state["recording"]      = recorder.active
            shared_state["clip_path"]      = recorder.path

        # ── Recording logic ──
        if phone_in_frame:
            recorder.touch()
            if not recorder.active:
                recorder.start(frame.shape)
        else:
            recorder.maybe_stop()

        # ── Annotate frame ──
        annotated = results[0].plot()
        draw_grid(annotated, GRID_ROWS, GRID_COLS)

        for det in detections:
            r, c = det["grid_row"], det["grid_col"]
            highlight_cell(annotated, r, c, GRID_ROWS, GRID_COLS)
            # Cell label overlay near bounding box
            x1, y1 = det["bbox"][0], det["bbox"][1]
            tag = f"R{r}C{c}  {det['confidence']:.0%}"
            cv2.rectangle(annotated, (x1, y1 - 22), (x1 + len(tag) * 9, y1),
                          (0, 200, 80), -1)
            cv2.putText(annotated, tag, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if recorder.active:
            recorder.write(annotated)
            cv2.circle(annotated, (20, 20), 8, (0, 0, 255), -1)   # red REC dot

        # ── Encode for MJPEG stream ──
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_jpeg = jpeg.tobytes()

        cv2.imshow("Phone Detection — CCTV", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if recorder.active:
        recorder.stop()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# FLASK API
# ──────────────────────────────────────────────
app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})


@app.route("/status")
def status():
    """
    JSON snapshot — poll this every 0.5–1 s from React app.

    Response shape:
    {
      "phone_detected": true,
      "timestamp": "2024-06-01T10:30:00Z",
      "recording": true,
      "frame_width": 640,
      "frame_height": 480,
      "grid_rows": 3,
      "grid_cols": 4,
      "detections": [
        {
          "bbox": [x1, y1, x2, y2],
          "center": [cx, cy],
          "confidence": 0.87,
          "grid_row": 2,
          "grid_col": 3,
          "label": "Row 2, Col 3"
        }
      ]
    }
    """
    with state_lock:
        return jsonify(shared_state)


@app.route("/stream")
def stream():
    """
    MJPEG live stream. Use in React/HTML:
        <img src="http://<PC_IP>:5000/stream" />
    Or fetch and display in a canvas element.
    """
    def generate():
        while True:
            with frame_lock:
                jpeg = latest_jpeg
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.04)   # ~25 fps cap

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    """Single JPEG frame — useful for thumbnails."""
    with frame_lock:
        jpeg = latest_jpeg
    if jpeg is None:
        return "No frame yet", 503
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/grid-info")
def grid_info():
    """Returns only the grid config — useful on app launch."""
    return jsonify({
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "frame_width":  shared_state["frame_width"],
        "frame_height": shared_state["frame_height"],
    })


@app.route("/examinees")
def get_all_examinees():
    """Get all registered examinees"""
    return jsonify({
        "total": len(examinee_manager.get_all_examinees()),
        "examinees": examinee_manager.get_all_examinees(),
        "stats": examinee_manager.get_stats()
    })


@app.route("/examinees/hall/<int:exam_hall>")
def get_examinees_hall(exam_hall):
    """Get examinees for a specific exam hall"""
    examinees = examinee_manager.get_examinees_by_hall(exam_hall)
    return jsonify({
        "exam_hall": exam_hall,
        "total": len(examinees),
        "examinees": examinees
    })


@app.route("/violations")
def get_violations():
    """Get all logged phone detection violations"""
    return jsonify({
        "total": len(violation_logger.get_violations()),
        "violations": violation_logger.get_violations()
    })


@app.route("/hall-info/<int:exam_hall>")
def get_hall_info(exam_hall):
    """Get complete hall information including examinees and violations"""
    hall_violations = [v for v in violation_logger.get_violations() if v['exam_hall'] == exam_hall]
    return jsonify({
        **examinee_manager.get_exam_hall_info(exam_hall),
        "violations": len(hall_violations),
        "violation_details": hall_violations
    })


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Detection runs in a background thread; Flask serves requests on main thread
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    print(f"\n[API] Running at http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"      /status      → JSON detection state")
    print(f"      /stream      → MJPEG live feed")
    print(f"      /snapshot    → single JPEG")
    print(f"      /grid-info   → grid dimensions")
    print(f"\n[EXAMINEES]")
    print(f"      /examinees        → all registered examinees")
    print(f"      /examinees/hall/<n> → examinees for hall N")
    print(f"      /hall-info/<n>    → hall info with examinees & violations")
    print(f"\n[VIOLATIONS]")
    print(f"      /violations  → all logged violations")
    print(f"\n[CORS] Enabled for React frontend at http://localhost:3000\n")

    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)
