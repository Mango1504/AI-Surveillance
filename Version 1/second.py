"""
Phone Detection System — second.py
Detects phones via YOLO26, maps to grid cells, records violations.

YOLO26 changes vs YOLOv8:
  - NMS-free end-to-end inference (no duplicate boxes, faster per frame)
  - No Distribution Focal Loss (simpler export, better edge perf)
  - Small-Target-Aware Label Assignment (STAL) — better for small phones
  - Load with task='detect' explicitly for clarity
"""

import cv2
import time
import os
import threading
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CAMERA_INDEX             = 0
SAVE_PATH                = r"C:/Users/write/Desktop/Phone"
APPLICANTS_PATH          = r"C:\Users\write\Desktop\AI Surveillance\applicants"
GRID_ROWS                = 5
GRID_COLS                = 6
PHONE_CLASS_ID           = 67       # COCO class 67 = cell phone (unchanged in YOLO26)
CONFIDENCE_MIN           = 0.45
FLASK_HOST               = "0.0.0.0"
FLASK_PORT               = 5000
POST_DETECT_RECORD_SECS  = 5

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(APPLICANTS_PATH, exist_ok=True)


# ──────────────────────────────────────────────
# SHARED STATE
# ──────────────────────────────────────────────
state_lock = threading.Lock()
shared_state = {
    "phone_detected": False,
    "detections":     [],
    "frame_width":    0,
    "frame_height":   0,
    "grid_rows":      GRID_ROWS,
    "grid_cols":      GRID_COLS,
    "timestamp":      None,
    "recording":      False,
    "clip_path":      None,
    "model":          "yolo26n",     # NEW: expose active model to API consumers
}

frame_lock  = threading.Lock()
latest_jpeg = None


# ──────────────────────────────────────────────
# GRID HELPERS
# ──────────────────────────────────────────────
def bbox_to_grid(cx, cy, frame_w, frame_h, rows, cols):
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
        self.writer       = None
        self.active       = False
        self.path         = None
        self._last_detect = 0

    def start(self, frame_shape):
        h, w     = frame_shape[:2]
        fourcc   = cv2.VideoWriter_fourcc(*"XVID")
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path   = os.path.join(SAVE_PATH, f"phone_{ts}.avi")
        self.writer = cv2.VideoWriter(self.path, fourcc, 20.0, (w, h))
        self.active = True
        print(f"[REC] Started → {self.path}")

    def write(self, frame):
        if self.active and self.writer:
            self.writer.write(frame)

    def touch(self):
        self._last_detect = time.time()

    def maybe_stop(self):
        if self.active and (time.time() - self._last_detect) > POST_DETECT_RECORD_SECS:
            self.stop()

    def stop(self):
        if self.writer:
            self.writer.release()
        self.active = False
        print(f"[REC] Stopped → {self.path}")


# ──────────────────────────────────────────────
# DETECTION LOOP
# ──────────────────────────────────────────────
def detection_loop():
    global latest_jpeg

    # ── YOLO26 model load ──────────────────────────────────────────────
    # task='detect' is explicit — YOLO26 supports detect/segment/pose/classify
    # verbose=False on model load suppresses the ultralytics banner
    # YOLO26 is NMS-free so you will NOT get duplicate bounding boxes
    # that you sometimes had to filter out with YOLOv8.
    model = YOLO("yolo26n.pt", task="detect", verbose=False)

    cap      = cv2.VideoCapture(CAMERA_INDEX)
    recorder = ClipRecorder()

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    ret, test_frame = cap.read()
    frame_h, frame_w = test_frame.shape[:2] if ret else (480, 640)

    with state_lock:
        shared_state["frame_width"]  = frame_w
        shared_state["frame_height"] = frame_h

    print(f"[CAM]   {frame_w}x{frame_h}  grid={GRID_ROWS}x{GRID_COLS}")
    print(f"[MODEL] YOLO26n  NMS-free=True  class={PHONE_CLASS_ID}  conf≥{CONFIDENCE_MIN}")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── Run inference ──────────────────────────────────────────────
        # YOLO26 is NMS-free: conf threshold is the only filter needed.
        # No iou / agnostic_nms args required.
        results        = model(frame, conf=CONFIDENCE_MIN, verbose=False)
        detections     = []
        phone_in_frame = False

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i, cls in enumerate(boxes.cls.cpu().numpy().astype(int)):
                if cls != PHONE_CLASS_ID:
                    continue

                conf        = float(boxes.conf[i].cpu())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                cx, cy      = (x1 + x2) // 2, (y1 + y2) // 2
                row, col    = bbox_to_grid(cx, cy, frame_w, frame_h,
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

        # ── Update shared state ────────────────────────────────────────
        with state_lock:
            shared_state["phone_detected"] = phone_in_frame
            shared_state["detections"]     = detections
            shared_state["timestamp"]      = datetime.utcnow().isoformat() + "Z"
            shared_state["recording"]      = recorder.active
            shared_state["clip_path"]      = recorder.path

        # ── Recording logic ────────────────────────────────────────────
        if phone_in_frame:
            recorder.touch()
            if not recorder.active:
                recorder.start(frame.shape)
        else:
            recorder.maybe_stop()

        # ── Annotate frame ─────────────────────────────────────────────
        annotated = results[0].plot()
        draw_grid(annotated, GRID_ROWS, GRID_COLS)

        for det in detections:
            r, c = det["grid_row"], det["grid_col"]
            highlight_cell(annotated, r, c, GRID_ROWS, GRID_COLS)
            x1, y1 = det["bbox"][0], det["bbox"][1]
            tag = f"R{r}C{c}  {det['confidence']:.0%}"
            cv2.rectangle(annotated, (x1, y1 - 22),
                          (x1 + len(tag) * 9, y1), (0, 200, 80), -1)
            cv2.putText(annotated, tag, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if recorder.active:
            recorder.write(annotated)
            cv2.circle(annotated, (20, 20), 8, (0, 0, 255), -1)  # red REC dot

        # ── YOLO26 model label (top-right corner) ──────────────────────
        cv2.putText(annotated, "YOLO26n", (frame_w - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # ── MJPEG encode ───────────────────────────────────────────────
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock:
            latest_jpeg = jpeg.tobytes()

        cv2.imshow("Phone Detection System — YOLO26", annotated)
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
CORS(app)


@app.route("/status")
def status():
    with state_lock:
        return jsonify(shared_state)


@app.route("/stream")
def stream():
    def generate():
        while True:
            with frame_lock:
                jpeg = latest_jpeg
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.04)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    with frame_lock:
        jpeg = latest_jpeg
    if jpeg is None:
        return "No frame yet", 503
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/grid-info")
def grid_info():
    return jsonify({
        "rows":         GRID_ROWS,
        "cols":         GRID_COLS,
        "frame_width":  shared_state["frame_width"],
        "frame_height": shared_state["frame_height"],
        "model":        shared_state["model"],
    })


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    print(f"\n[API] Running at http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"      /status     → JSON detection state")
    print(f"      /stream     → MJPEG live feed")
    print(f"      /snapshot   → single JPEG")
    print(f"      /grid-info  → grid + model info\n")

    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)