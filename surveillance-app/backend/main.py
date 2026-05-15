import os
import time
import threading
import cv2
import datetime
import queue
import signal
import atexit

from flask import Flask, jsonify, Response, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# Adaptive Modules
from config import get_config
from adaptive_throttle import AdaptiveThrottle
from frame_queue import FrameQueue
from disk_writer import DiskWriter

# Feature Modules
from camera import MultiCameraStream, ExamDetector
from identity import IdentityManager
from behavior import GazeTracker, MotionAnalyzer
from vlm_reporter import VLMReporter
from evidence import EvidenceCache
from risk_engine import RiskEngine

# ──────────────────────────────────────────────
# SYSTEM INITIALIZATION
# ──────────────────────────────────────────────
config = get_config()
frame_queue = FrameQueue()
throttle = AdaptiveThrottle(frame_queue=frame_queue).start()
result_queue = queue.Queue(maxsize=config.FRAME_QUEUE_MAXSIZE)
shutdown_event = threading.Event()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

latest_jpeg = None
frame_lock = threading.Lock()
shared_state = {
    "phone_detected": False,
    "anomaly_detected": False,
    "detections": [],
    "timestamp": None,
    "recording": False,
    "auto_record": True,
    "manual_record": False,
    "clip_path": None,
    "frame_width": 1920,
    "frame_height": 1080,
    "grid_rows": 3,
    "grid_cols": 4,
    "fps": 0.0,
    "alert_latency_ms": 0.0,
}
state_lock = threading.Lock()

proj_root = os.path.dirname(os.path.dirname(__file__))
VIDEOS_DIR = os.path.join(proj_root, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

cam_stream = MultiCameraStream(config.CAMERA_SOURCES, frame_queue)
disk_writer = DiskWriter(output_dir=VIDEOS_DIR).start()
evidence = EvidenceCache(key_path=os.path.join(proj_root, "secret.key"))
session_id = evidence.start_session()

detectors = [ExamDetector() for _ in range(config.NUM_DETECTION_WORKERS)]
identity_mgr = IdentityManager(os.path.join(proj_root, "examinees.json"), os.path.join(proj_root, "..", "applicants"))
gaze_tracker = GazeTracker()
motion_analyzers = {}
vlm = VLMReporter()
risk_engine = RiskEngine()

# ──────────────────────────────────────────────
# DETECTION POOL
# ──────────────────────────────────────────────
def detection_worker(worker_id):
    detector = detectors[worker_id]
    print(f"[WORKER-{worker_id}] Detection thread ready.")
    
    frame_counter = 0
    intruder_last_alert = {}  # cam_id -> last alert timestamp

    while not shutdown_event.is_set():
        # Pull from FrameQueue
        data = frame_queue.pop(timeout=0.1)
        if data is None:
            continue
        
        try:
            timestamp, cam_id, frame = data
            frame_counter += 1
            
            # Adaptive Frame Skip (N)
            if frame_counter % max(1, config.DETECT_EVERY_N) != 0:
                continue
                
            # 1. Object Detection
            detections = []
            if config.ENABLE_OBJECT_DET:
                annotated_frame, detections = detector.detect(frame)
            else:
                annotated_frame = frame.copy()

            # 2. Intruder Check — per detected person, cross-reference DII roster
            for det in detections:
                if det.get("class") == "person" and identity_mgr.known_embeddings:
                    now = time.time()
                    if now - intruder_last_alert.get(cam_id, 0) > 10.0:
                        # Crop person bounding box for face check
                        x1, y1, x2, y2 = det["bbox"]
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size > 0 and identity_mgr.is_intruder(crop):
                            intruder_last_alert[cam_id] = now
                            socketio.emit("intruder_detected", {
                                "cam_id": cam_id,
                                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                                "confidence": det.get("confidence", 0),
                            })
                            evidence.log_incident(
                                "INTRUDER",
                                ["unauthorized person"],
                                f"Camera {cam_id}: Unregistered person detected in exam hall.",
                                annotated_frame.copy(),
                                flagged=1,
                                clip_path=None,
                            )

            # 3. Gaze / Head Pose
            gaze_anomaly = False
            movement_anomaly = False
            movement_score = 0.0
            if config.ENABLE_GAZE:
                is_looking_away, duration = gaze_tracker.analyze_gaze(frame)
                if is_looking_away and duration > 3.0:
                    gaze_anomaly = True
                    gaze_tracker.looking_away_start = None

            analyzer = motion_analyzers.setdefault(cam_id, MotionAnalyzer())
            movement_anomaly, movement_score = analyzer.analyze(frame)

            # 4. Push detections into Risk Engine sliding window
            risk_engine.push(cam_id, detections, gaze_anomaly, movement_anomaly)
                    
            # 5. Push to ResultQueue
            try:
                result_queue.put_nowait({
                    "timestamp": timestamp,
                    "cam_id": cam_id,
                    "frame": frame,
                    "annotated_frame": annotated_frame,
                    "detections": detections,
                    "gaze_anomaly": gaze_anomaly,
                    "movement_anomaly": movement_anomaly,
                    "movement_score": movement_score,
                    "detect_ts": time.time(),
                })
            except queue.Full:
                pass  # Drop results if pipeline is backed up
        except Exception as e:
            print(f"[WORKER-{worker_id}] Error: {e}")

# ──────────────────────────────────────────────
# DASHBOARD / ALERT ROUTER LOOP
# ──────────────────────────────────────────────
def alert_router_loop():
    print("[ROUTER] Alert & Dashboard stream started.")
    last_vlm_call = 0
    
    # Recording tracking per camera
    recording_active = {}
    last_detect_time = {}
    prev_manual_record = False
    last_incident_log = {}
    
    while not shutdown_event.is_set():
        try:
            res = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        try:
            cam_id = res["cam_id"]
            frame = res["frame"]
            annotated_frame = res["annotated_frame"]
            detections = res["detections"]
            gaze_anomaly = res["gaze_anomaly"]
            movement_anomaly = res.get("movement_anomaly", False)
            movement_score = res.get("movement_score", 0.0)
                
            # Detect Anomalies
            phone_detected = any(d['class'] == 'cell phone' for d in detections)
            anomaly_detected = phone_detected or gaze_anomaly or movement_anomaly
            
            with state_lock:
                auto_record_enabled = shared_state.get("auto_record", True)
                manual_record_enabled = shared_state.get("manual_record", False)
            
            # Detect manual record toggle OFF → immediately close clips
            if prev_manual_record and not manual_record_enabled:
                for cid in list(recording_active.keys()):
                    if recording_active.get(cid, False):
                        disk_writer.close_clip(cid)
                        recording_active[cid] = False
            prev_manual_record = manual_record_enabled
            
            # Clip Recording logic
            clip_path_url = None
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            codec_ext = "mp4" if config.CLIP_CODEC != "MJPG" else "avi"
            fname = f"clip_{cam_id}_{ts_str}.{codec_ext}"
            local_path = os.path.join(VIDEOS_DIR, fname)
            
            should_record = (anomaly_detected and auto_record_enabled) or manual_record_enabled
            
            if should_record:
                last_detect_time[cam_id] = time.time()
                if not recording_active.get(cam_id, False):
                    recording_active[cam_id] = local_path
            else:
                if recording_active.get(cam_id, False):
                    if (time.time() - last_detect_time.get(cam_id, 0)) > config.PRE_BUFFER_SECS:
                        disk_writer.close_clip(cam_id)
                        recording_active[cam_id] = False
                        
            is_recording = bool(recording_active.get(cam_id, False))
            if is_recording:
                disk_writer.enqueue_frame_for_clip(cam_id, annotated_frame, recording_active[cam_id])
                clip_path_url = f"http://localhost:5000/videos/{os.path.basename(recording_active[cam_id])}"

            # Update Shared State — map ALL detections to grid cells
            frame_h, frame_w = frame.shape[:2]
            grid_rows = shared_state.get("grid_rows", 3)
            grid_cols = shared_state.get("grid_cols", 4)
            processed_detections = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                row = min(int(cy / max(frame_h, 1) * grid_rows) + 1, grid_rows)
                col = min(int(cx / max(frame_w, 1) * grid_cols) + 1, grid_cols)
                d_new = dict(det)
                d_new["grid_row"] = row
                d_new["grid_col"] = col
                d_new["label"] = f"Row {row}, Col {col}"
                processed_detections.append(d_new)

            if gaze_anomaly:
                processed_detections.append({
                    "class": "gaze anomaly",
                    "confidence": 0.7,
                    "bbox": [0, 0, max(frame_w, 1), max(frame_h, 1)],
                    "grid_row": 2,
                    "grid_col": 2,
                    "label": "Gaze / Head Pose",
                })

            if movement_anomaly:
                processed_detections.append({
                    "class": "unusual movement",
                    "confidence": round(float(movement_score), 3),
                    "bbox": [0, 0, max(frame_w, 1), max(frame_h, 1)],
                    "grid_row": 2,
                    "grid_col": 2,
                    "label": "Unusual Movement",
                })
                    
            # Measure end-to-end alert latency
            detect_ts = res.get("detect_ts", time.time())
            emit_ts = time.time()
            latency_ms = round((emit_ts - detect_ts) * 1000, 1)

            # Compute risk score in router thread (safe — no Flask thread involved)
            risk_info = risk_engine.get_score(cam_id)

            with state_lock:
                shared_state["phone_detected"] = phone_detected
                shared_state["anomaly_detected"] = anomaly_detected
                shared_state["detections"] = processed_detections
                shared_state["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
                shared_state["recording"] = is_recording
                shared_state["clip_path"] = clip_path_url
                shared_state["frame_width"] = frame_w
                shared_state["frame_height"] = frame_h
                shared_state["auto_record"] = auto_record_enabled
                shared_state["manual_record"] = manual_record_enabled
                shared_state["alert_latency_ms"] = latency_ms
                shared_state["risk_score"] = risk_info["score"]
                shared_state["risk_sustained_secs"] = risk_info["sustained_secs"]

            # ── Risk-gated incident logging (Layer 4) ──
            # Only log incidents and fire alerts when risk engine clears both
            # threshold AND minimum sustained duration checks.
            risk_cleared = risk_engine.should_alert(cam_id)

            if anomaly_detected:
                severity = "HIGH" if phone_detected else "LOW"
                event_type = "unauthorized_object" if phone_detected else "gaze_anomaly" if gaze_anomaly else "unusual_movement"
                labels_for_event = [det["class"] for det in detections]
                if gaze_anomaly:
                    labels_for_event.append("gaze anomaly")
                if movement_anomaly:
                    labels_for_event.append("unusual movement")

                # Always log proctor events + anomaly scores for audit trail
                evidence.log_proctor_event(
                    session_id=session_id,
                    camera_id=cam_id,
                    severity=severity,
                    event_type=event_type,
                    score=risk_engine.get_score(cam_id)["score"],
                    labels=labels_for_event,
                    clip_path=clip_path_url,
                )
                evidence.log_anomaly_score(
                    session_id=session_id,
                    candidate_id=None,
                    gaze=0.7 if gaze_anomaly else 0.0,
                    movement=movement_score if movement_anomaly else 0.0,
                    objects=1.0 if phone_detected else 0.0,
                )

                # Log every rate-limited anomaly to DB for the archive.
                # flagged=1 only when risk engine clears (confirmed high-confidence alert).
                incident_key = (cam_id, event_type)
                now = time.time()
                if now - last_incident_log.get(incident_key, 0) > 5.0:
                    last_incident_log[incident_key] = now
                    incident_flagged = 1 if risk_cleared else 0
                    report = (
                        "Phone detected in the exam hall."
                        if phone_detected else
                        "Candidate gaze/head pose anomaly detected."
                        if gaze_anomaly else
                        "Unusual movement detected in the exam hall."
                    )
                    evidence.log_incident(
                        "Unknown",
                        labels_for_event,
                        report,
                        annotated_frame.copy(),
                        flagged=incident_flagged,
                        clip_path=clip_path_url,
                    )

                # Fire SocketIO alert only when risk engine confirms
                if risk_cleared:
                    socketio.emit("alert", {
                        "event_type": event_type,
                        "severity": severity,
                        "cam_id": cam_id,
                        "timestamp": shared_state["timestamp"],
                        "clip_path": clip_path_url,
                    })

            # ── VLM reporting (throttled + risk-gated) ──
            if risk_cleared and anomaly_detected and (time.time() - last_vlm_call > max(config.profile.cpu_cores_logical, 5)):
                labels = [det['class'] for det in detections]
                if gaze_anomaly:
                    labels.append("suspicious_gaze")
                if movement_anomaly:
                    labels.append("unusual_movement")

                # Determine primary grid position for VLM context
                primary_det = next((d for d in processed_detections if d.get("class") == "cell phone"), None) or \
                              (processed_detections[0] if processed_detections else None)
                vlm_row = primary_det.get("grid_row") if primary_det else None
                vlm_col = primary_det.get("grid_col") if primary_det else None

                def _async_report(frm, lbls, c_url, _cam_id, _row, _col):
                    candidate_id = "Unknown"
                    if "person" in lbls:
                        candidate_id = identity_mgr.verify_face(frm)
                    report = vlm.generate_report(
                        frm, lbls,
                        camera_id=_cam_id,
                        grid_row=_row,
                        grid_col=_col,
                        candidate_id=candidate_id,
                    )
                    socketio.emit('anomaly_detected', {
                        'timestamp': time.time(),
                        'candidate_id': candidate_id,
                        'report': report,
                        'labels': lbls,
                        'clip_path': c_url,
                        'camera_id': _cam_id,
                        'grid_row': _row,
                        'grid_col': _col,
                        'risk_score': risk_engine.get_score(_cam_id)["score"],
                    })

                threading.Thread(
                    target=_async_report,
                    args=(frame.copy(), labels, clip_path_url, cam_id, vlm_row, vlm_col),
                    daemon=True
                ).start()
                last_vlm_call = time.time()

        except Exception as e:
            print(f"[ROUTER] Error processing frame: {e}")

        result_queue.task_done()

# ──────────────────────────────────────────────
# ZERO-LATENCY MJPEG ENCODER
# ──────────────────────────────────────────────
def mjpeg_encoder_loop():
    print("[ENCODER] Zero-latency MJPEG stream started.")
    cam_id = config.CAMERA_SOURCES[0]
    fps_counter = 0
    fps_timer = time.time()

    while not shutdown_event.is_set():
        frame = cam_stream.latest_frames.get(cam_id)
        if frame is None:
            time.sleep(0.05)
            continue
            
        with state_lock:
            detections = shared_state["detections"]
            
        draw_frame = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d["confidence"]
            label = d.get("class", "alert")
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(draw_frame, f"{label} {conf:.2f}", (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        _, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_QUALITY])
        with frame_lock:
            global latest_jpeg
            latest_jpeg = buffer.tobytes()

        # FPS tracking — update shared_state every second
        fps_counter += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = round(fps_counter / max(now - fps_timer, 0.001), 1)
            with state_lock:
                shared_state["fps"] = fps
            fps_counter = 0
            fps_timer = now

        time.sleep(1.0 / 60.0)  # ~60 FPS cap

# ──────────────────────────────────────────────
# FLASK ROUTES
# ──────────────────────────────────────────────
@app.route('/status')
def status():
    with state_lock:
        return jsonify(shared_state)

@app.route('/grid-info')
def grid_info():
    with state_lock:
        return jsonify({
            "rows": shared_state["grid_rows"],
            "cols": shared_state["grid_cols"],
            "frame_width": shared_state["frame_width"],
            "frame_height": shared_state["frame_height"]
        })

@app.route('/incidents')
def incidents():
    # Return ALL incident records (flagged=1 are confirmed alerts, flagged=0 are detections)
    records = evidence.get_all_incidents()
    for r in records:
        if r.get("clip_path") and not r["clip_path"].startswith("http"):
            basename = os.path.basename(r["clip_path"])
            r["clip_path"] = f"http://localhost:5000/videos/{basename}"
    return jsonify(records)

@app.route('/videos/scan', methods=['POST'])
def scan_videos():
    """Scan the videos directory and register any untracked .mp4 files as archive entries.
    Allows the operator to import all existing clips into the Incident Archive with one call.
    """
    if not os.path.isdir(VIDEOS_DIR):
        return jsonify({"added": 0, "message": "Videos directory not found"})
    inserted = 0
    for fname in sorted(os.listdir(VIDEOS_DIR)):
        if fname.lower().endswith(".mp4"):
            url = f"http://localhost:5000/videos/{fname}"
            if evidence.insert_video_record(fname, url):
                inserted += 1
    return jsonify({"added": inserted, "message": f"{inserted} new clips registered in archive"})

@app.route('/videos/list')
def list_videos():
    """Return all .mp4 files in the videos directory."""
    if not os.path.isdir(VIDEOS_DIR):
        return jsonify([])
    files = []
    for fname in sorted(os.listdir(VIDEOS_DIR), reverse=True):
        if fname.lower().endswith(".mp4"):
            fpath = os.path.join(VIDEOS_DIR, fname)
            files.append({
                "filename": fname,
                "url": f"http://localhost:5000/videos/{fname}",
                "thumbnail": f"http://localhost:5000/videos/thumbnail/{fname}",
                "size_mb": round(os.path.getsize(fpath) / 1024 / 1024, 1),
                "modified": os.path.getmtime(fpath),
            })
    return jsonify(files)

@app.route('/incidents/delete', methods=['POST'])
def delete_incidents():
    data = request.json
    ids = data.get("ids", [])
    if ids:
        evidence.delete_incidents(ids)
    return jsonify({"success": True, "deleted": len(ids)})

@app.route('/toggle-recording', methods=['POST'])
def toggle_recording():
    data = request.json
    enabled = data.get("enabled", True)
    with state_lock:
        shared_state["auto_record"] = enabled
    return jsonify({"success": True, "auto_record": enabled})

@app.route('/manual-record', methods=['POST'])
def manual_record():
    data = request.json
    enabled = data.get("enabled", True)
    with state_lock:
        shared_state["manual_record"] = enabled
    return jsonify({"success": True, "manual_record": enabled})

# ── NEW: System Info Endpoint ──
@app.route('/system-info')
def system_info():
    import psutil
    cpu_percent = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory()
    return jsonify({
        "tier": config.profile.tier,
        "cpu_cores": config.profile.cpu_cores_physical,
        "cpu_logical": config.profile.cpu_cores_logical,
        "cpu_percent": cpu_percent,
        "ram_total_mb": config.profile.ram_total_mb,
        "ram_used_mb": int((ram.total - ram.available) / (1024 * 1024)),
        "ram_percent": ram.percent,
        "gpu": "CUDA" if config.profile.has_cuda else "MPS" if config.profile.has_mps else "None",
        "gpu_name": config.profile.gpu_name,
        "acceleration_backend": config.ACCELERATION_BACKEND,
        "onnx_providers": config.profile.onnx_providers,
        "yolo_conf": detectors[0].conf_thresh if detectors else 0.3,
        "detect_every_n": config.DETECT_EVERY_N,
        "detection_resolution": f"{config.DETECTION_RESOLUTION[0]}x{config.DETECTION_RESOLUTION[1]}",
        "clip_resolution": f"{config.CLIP_RESOLUTION[0]}x{config.CLIP_RESOLUTION[1]}",
        "clip_codec": config.CLIP_CODEC,
        "object_det_enabled": config.ENABLE_OBJECT_DET,
        "gaze_enabled": config.ENABLE_GAZE,
        "frame_queue_size": frame_queue.size(),
        "frame_drop_rate": round(frame_queue.get_drop_rate() * 100, 1),
        "num_workers": config.NUM_DETECTION_WORKERS,
        "model_name": config.OBJECT_MODEL.replace(".pt", ""),
        "camera_caps": config.profile.camera_caps,
    })

# ── NEW: Config Read/Write ──
@app.route('/config', methods=['GET'])
def get_config_route():
    return jsonify({
        "yolo_confidence": int(detectors[0].conf_thresh * 100) if detectors else 30,
        "vlm_frequency": max(config.profile.cpu_cores_logical, 5),
        "object_det_enabled": config.ENABLE_OBJECT_DET,
        "gaze_enabled": config.ENABLE_GAZE,
        "detect_every_n": config.DETECT_EVERY_N,
        "resolution": f"{config.DEFAULT_CAPTURE_RES[0]}x{config.DEFAULT_CAPTURE_RES[1]}",
        "detection_resolution": f"{config.DETECTION_RESOLUTION[0]}x{config.DETECTION_RESOLUTION[1]}",
        "clip_resolution": f"{config.CLIP_RESOLUTION[0]}x{config.CLIP_RESOLUTION[1]}",
        "auto_record": shared_state.get("auto_record", True),
    })

@app.route('/config', methods=['POST'])
def update_config():
    data = request.json
    
    # YOLO confidence
    if "yolo_confidence" in data:
        new_conf = max(0.1, min(0.95, data["yolo_confidence"] / 100.0))
        for det in detectors:
            det.conf_thresh = new_conf
    
    # Detection toggle
    if "object_det_enabled" in data:
        config.ENABLE_OBJECT_DET = bool(data["object_det_enabled"])
    
    # Gaze toggle
    if "gaze_enabled" in data:
        config.ENABLE_GAZE = bool(data["gaze_enabled"])
    
    # Frame skip
    if "detect_every_n" in data:
        config.DETECT_EVERY_N = max(1, min(10, int(data["detect_every_n"])))

    # Risk engine parameters
    if "risk_threshold" in data:
        config.RISK_THRESHOLD = max(0.1, min(1.0, float(data["risk_threshold"])))
    if "temporal_window" in data:
        config.TEMPORAL_WINDOW_SECS = max(1.0, min(15.0, float(data["temporal_window"])))
    if "min_detection_duration" in data:
        config.MIN_DETECTION_DURATION_SECS = max(0.5, min(10.0, float(data["min_detection_duration"])))

    return jsonify({"success": True, "message": "Configuration updated"})

# ── NEW: Identified Students ──
@app.route('/identified-students')
def identified_students():
    # Returns currently identified students from identity manager
    # In production this would return real DeepFace matches
    return jsonify({"identified_students": []})

# ── NEW: Purge Biometrics ──
@app.route('/purge-biometrics', methods=['POST'])
def purge_biometrics():
    identity_mgr.purge_data()
    return jsonify({"success": True, "message": "All biometric embeddings purged"})

# ── NEW: Reset Session ──
@app.route('/reset-session', methods=['POST'])
def reset_session_route():
    evidence.reset_session()
    return jsonify({"success": True, "message": "Session reset, all incidents cleared"})


# ── NEW: Risk Score (Layer 4 visibility) ──
@app.route('/risk-score')
def get_risk_score():
    """Return current risk score for the primary camera from shared_state (no extra locking)."""
    with state_lock:
        score = shared_state.get("risk_score", 0.0)
        sustained = shared_state.get("risk_sustained_secs", 0.0)
    return jsonify({
        "cameras": {
            "0": {
                "score": score,
                "sustained_secs": sustained,
                "threshold": config.RISK_THRESHOLD,
                "window_secs": config.TEMPORAL_WINDOW_SECS,
            }
        },
        "risk_threshold": config.RISK_THRESHOLD,
        "temporal_window_secs": config.TEMPORAL_WINDOW_SECS,
        "min_duration_secs": config.MIN_DETECTION_DURATION_SECS,
    })


# ── NEW: Session Start (DII provisioning) ──
@app.route('/session/start', methods=['POST'])
def session_start():
    """Provision a new exam session with a candidate roster.

    Body: { "candidates": [{"roll_number": "...", "name": "..."}, ...] }
    Optionally the roster JSON is written to examinees.json and the identity
    manager reloads embeddings for the new session.
    """
    global session_id
    data = request.json or {}
    candidates = data.get("candidates", [])

    # Hard-wipe previous session biometrics (B-05 clean separation)
    identity_mgr.purge_data()
    risk_engine.reset()

    # Reset evidence session
    evidence.reset_session()
    session_id = evidence.start_session()

    # Write new roster to examinees.json if provided; load embeddings in background
    # (DeepFace model load is expensive ~3-8s — must not block the Flask thread)
    loading_async = False
    if candidates:
        import json as _json
        roster_path = os.path.join(proj_root, "examinees.json")
        with open(roster_path, "w", encoding="utf-8") as f:
            _json.dump(candidates, f, indent=2)
        identity_mgr.examinees_path = roster_path

        def _load_bg():
            identity_mgr._load_examinees()
            cnt = len(identity_mgr.known_embeddings)
            print(f"[SESSION] Embeddings ready: {cnt} candidates")
            socketio.emit("embeddings_ready", {"candidates_loaded": cnt})

        threading.Thread(target=_load_bg, daemon=True, name="EmbeddingLoader").start()
        loading_async = True

    socketio.emit("session_started", {
        "session_id": session_id,
        "candidates": len(candidates),
    })
    print(f"[SESSION] New session {session_id} started with {len(candidates)} candidates.")
    return jsonify({
        "success": True,
        "session_id": session_id,
        "candidates_loaded": len(identity_mgr.known_embeddings),
        "loading": loading_async,
    })


# ── NEW: Session End (purge + biometric wipe) ──
@app.route('/session/end', methods=['POST'])
def session_end():
    """End the current session: close clips, purge non-flagged data, wipe biometrics."""
    global session_id

    # Close any open video clips immediately
    for cid in list(disk_writer.clip_writers.keys()):
        disk_writer.close_clip(cid)

    # Purge non-flagged incidents + videos (runs in background thread, ≤5 min)
    evidence.purge_non_flagged(videos_dir=VIDEOS_DIR)

    # Wipe biometric embeddings from memory
    identity_mgr.purge_data()

    # Reset risk windows
    risk_engine.reset()

    # Mark session as ended in DB
    evidence.end_session(session_id)
    session_id = None

    socketio.emit("session_ended", {"message": "Session ended. Non-flagged data purge initiated."})
    print("[SESSION] Session ended. Purge job running in background.")
    return jsonify({
        "success": True,
        "message": "Session ended. Non-flagged video purge initiated (≤5 min).",
    })

@app.route('/proctor')
def proctor_panel():
    return jsonify({
        "session_id": session_id,
        "tier": config.profile.tier,
        "status": shared_state,
        "hardware": {
            "cpu_cores": config.profile.cpu_cores_logical,
            "ram_budget_mb": config.profile.ram_budget_mb,
            "gpu": config.profile.gpu_name,
            "backend": config.ACCELERATION_BACKEND,
        },
    })

# Simple thumbnail cache to avoid re-reading video files on every request
_thumbnail_cache = {}

@app.route('/videos/thumbnail/<path:filename>')
def serve_thumbnail(filename):
    """Extract and serve the first frame of a video as a JPEG thumbnail (cached)."""
    if filename in _thumbnail_cache:
        return Response(_thumbnail_cache[filename], mimetype='image/jpeg')
    filepath = os.path.join(VIDEOS_DIR, filename)
    if not os.path.exists(filepath):
        return "Video not found", 404
    cap = cv2.VideoCapture(filepath)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return "Could not read video frame", 500
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_bytes = jpeg.tobytes()
    # Cache up to 100 thumbnails
    if len(_thumbnail_cache) < 100:
        _thumbnail_cache[filename] = jpeg_bytes
    return Response(jpeg_bytes, mimetype='image/jpeg')


@app.route('/videos/play/<path:filename>')
def play_video(filename):
    """Stream a recorded video as MJPEG for browser playback.
    
    OpenCV's mp4v codec produces MPEG-4 Part 2 files which browsers cannot
    decode natively (they only support H.264). This endpoint re-encodes the
    video frame-by-frame as an MJPEG stream that any browser can display
    via an <img> tag.
    """
    filepath = os.path.join(VIDEOS_DIR, filename)
    if not os.path.exists(filepath):
        return "Video not found", 404

    def generate_frames():
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        delay = 1.0 / fps
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(delay)
        cap.release()

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve raw video file for proper browser playback with seeking and controls.
    NOTE: This catch-all must be registered AFTER the specific /videos/thumbnail/ and
    /videos/play/ routes so Flask's greedy <path:> converter does not shadow them.
    """
    filepath = os.path.join(VIDEOS_DIR, filename)
    if not os.path.exists(filepath):
        return "Video not found", 404
    return send_from_directory(VIDEOS_DIR, filename, mimetype='video/mp4')

@app.route('/stream')
def stream():
    def generate():
        while True:
            with frame_lock:
                jpeg = latest_jpeg
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(1.0 / 60.0)  # 60 FPS live feed
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def shutdown():
    if shutdown_event.is_set():
        return
    print("[SHUTDOWN] Flushing queues, closing cameras, saving session.")
    shutdown_event.set()
    throttle.stop()
    cam_stream.stop()
    disk_writer.stop()
    evidence.end_session(session_id)
    evidence.close()


def _handle_signal(signum, frame):
    shutdown()
    raise SystemExit(0)


atexit.register(shutdown)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

_pipeline_started = False


def run_server():
    global _pipeline_started
    print("[ORCHESTRATOR] Initializing multi-threaded pipeline...")

    if not _pipeline_started:
        cam_stream.start()

        # Spawn Detection Pool
        for i in range(config.NUM_DETECTION_WORKERS):
            threading.Thread(target=detection_worker, args=(i,), daemon=True).start()

        # Spawn Router and Fast Encoder
        threading.Thread(target=alert_router_loop, daemon=True).start()
        threading.Thread(target=mjpeg_encoder_loop, daemon=True).start()
        _pipeline_started = True

    print("[SERVER] Starting Flask-SocketIO API on port 5000...")
    try:
        socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
    finally:
        shutdown()


if __name__ == "__main__":
    run_server()
