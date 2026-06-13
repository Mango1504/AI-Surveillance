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
# METROPOLIS INTEGRATION (optional)
# ──────────────────────────────────────────────
_METROPOLIS_ENABLED = False
_metropolis_orchestrator = None

def _init_metropolis():
    """Attempt to initialize the Metropolis pipeline orchestrator.

    Metropolis mode is activated when EITHER:
      - The environment variable METROPOLIS_ENABLED=1 is set, OR
      - The config file configs/metropolis.yaml exists

    When enabled, loads MetropolisConfig, creates a PipelineOrchestrator,
    detects hardware capabilities, and selects the optimal pipeline.

    Returns:
        PipelineOrchestrator instance if enabled and initialized, else None.
    """
    import logging

    # Check activation conditions
    env_enabled = os.environ.get("METROPOLIS_ENABLED", "0") == "1"
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "metropolis.yaml")
    config_exists = os.path.isfile(config_path)

    if not env_enabled and not config_exists:
        return None

    try:
        from metropolis import MetropolisConfig, PipelineOrchestrator

        # Load configuration
        if config_exists:
            metro_config = MetropolisConfig.from_yaml(config_path)
            print(f"[METROPOLIS] Config loaded from {config_path}")
        else:
            metro_config = MetropolisConfig()
            print("[METROPOLIS] Using default configuration")

        # Create orchestrator, detect capabilities, select pipeline
        orchestrator = PipelineOrchestrator(metro_config)
        caps = orchestrator.detect_capabilities()
        pipeline = orchestrator.select_pipeline()

        print(f"[METROPOLIS] Capabilities: GPU={caps.has_gpu} ({caps.gpu_name}), "
              f"TensorRT={caps.has_tensorrt}, DeepStream={caps.has_deepstream}, "
              f"Triton={caps.has_triton}")
        print(f"[METROPOLIS] Selected pipeline: '{pipeline}'")

        return orchestrator

    except ImportError as e:
        logging.getLogger(__name__).warning(
            "Metropolis package not available: %s. Falling back to legacy pipeline.", e
        )
        return None
    except Exception as e:
        logging.getLogger(__name__).error(
            "Failed to initialize Metropolis orchestrator: %s. Falling back to legacy pipeline.", e
        )
        return None

# Attempt Metropolis initialization at module load time
_metropolis_orchestrator = _init_metropolis()
_METROPOLIS_ENABLED = _metropolis_orchestrator is not None

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

# On CUDA, GPU inference is serialised by ExamDetector._model_lock, so
# extra workers beyond 2 only add thread overhead without increasing throughput.
import torch as _torch
_workers = 2 if config.profile.has_cuda else max(1, min(config.profile.cpu_cores_logical // 2, 4))
detectors = [ExamDetector() for _ in range(_workers)]
identity_mgr = IdentityManager(os.path.join(proj_root, "examinees.json"), os.path.join(proj_root, "..", "applicants"))
gaze_tracker = GazeTracker()
gaze_lock = threading.Lock()          # ← protects gaze_tracker state across workers
motion_analyzers = {}
motion_analyzers_lock = threading.Lock()  # ← protects per-cam MotionAnalyzer creation

# Cache of latest annotated frame per camera for the MJPEG encoder
latest_annotated_frames = {}
annotated_frames_lock = threading.Lock()

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
                            # Log intruder face in identity DB
                            evidence.log_intruder(
                                camera_id=cam_id,
                                face_frame=crop,
                                confidence=det.get("confidence", 0),
                                notes=f"Unregistered person detected in exam hall (cam {cam_id})",
                            )
                            evidence.log_incident(
                                "INTRUDER",
                                ["unauthorized person"],
                                f"Camera {cam_id}: Unregistered person detected in exam hall.",
                                annotated_frame.copy(),
                                flagged=1,
                                clip_path=None,
                            )

            # 3. Gaze / Head Pose — use lock to protect shared gaze_tracker state
            gaze_anomaly = False
            movement_anomaly = False
            movement_score = 0.0
            if config.ENABLE_GAZE:
                with gaze_lock:
                    is_looking_away, duration = gaze_tracker.analyze_gaze(frame)
                    if is_looking_away and duration > 3.0:
                        gaze_anomaly = True
                        # Reset via lock — safe; previously this was written from router thread (race)
                        gaze_tracker.looking_away_start = None

            # MotionAnalyzer per camera — protect dict access with lock
            with motion_analyzers_lock:
                if cam_id not in motion_analyzers:
                    motion_analyzers[cam_id] = MotionAnalyzer()
                analyzer = motion_analyzers[cam_id]
            movement_anomaly, movement_score = analyzer.analyze(frame)

            # 4. Push detections into Risk Engine sliding window
            risk_engine.push(cam_id, detections, gaze_anomaly, movement_anomaly)
                    
            # 5. Cache latest annotated frame for the MJPEG encoder
            with annotated_frames_lock:
                latest_annotated_frames[cam_id] = annotated_frame

            # 6. Push to ResultQueue
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
    prev_auto_record = True       # track auto_record transitions
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

            # Auto-record turned OFF mid-clip → close all active clips immediately.
            # Without this, clips kept recording for PRE_BUFFER_SECS after the toggle.
            if prev_auto_record and not auto_record_enabled:
                for cid in list(recording_active.keys()):
                    if recording_active.get(cid):
                        disk_writer.close_clip(cid)
                        recording_active[cid] = None
                        print(f"[ROUTER] Auto-record disabled — closed clip for cam {cid}")
            prev_auto_record = auto_record_enabled

            # Manual record toggled OFF → immediately close clips
            if prev_manual_record and not manual_record_enabled:
                for cid in list(recording_active.keys()):
                    if recording_active.get(cid):
                        disk_writer.close_clip(cid)
                        recording_active[cid] = None
                        print(f"[ROUTER] Manual record stopped — closed clip for cam {cid}")
            prev_manual_record = manual_record_enabled

            # Clip Recording logic
            # IMPORTANT: only generate a new filename when STARTING a new clip — not every frame.
            clip_path_url = None
            should_record = (anomaly_detected and auto_record_enabled) or manual_record_enabled

            if should_record:
                last_detect_time[cam_id] = time.time()
                if not recording_active.get(cam_id):      # start new clip
                    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    codec_ext = "mp4" if config.CLIP_CODEC != "MJPG" else "avi"
                    fname = f"clip_{cam_id}_{ts_str}.{codec_ext}"
                    local_path = os.path.join(VIDEOS_DIR, fname)
                    recording_active[cam_id] = local_path
            else:
                # Only apply PRE_BUFFER_SECS cooldown when auto-record is still ON
                # (anomaly just cleared). When auto-record is OFF this block is never
                # reached because the toggle-OFF guard above already closed the clip.
                if recording_active.get(cam_id):
                    if (time.time() - last_detect_time.get(cam_id, 0)) > config.PRE_BUFFER_SECS:
                        disk_writer.close_clip(cam_id)
                        recording_active[cam_id] = None

            is_recording = bool(recording_active.get(cam_id))
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
                # NOTE: auto_record and manual_record are NOT written back here.
                # They are user-controlled preferences owned exclusively by the
                # /toggle-recording and /toggle-manual-record endpoints.
                # Writing them back from the router loop caused a race condition
                # where a user toggle mid-frame would be silently overwritten.
                shared_state["alert_latency_ms"] = latency_ms
                shared_state["risk_score"] = risk_info["score"]
                shared_state["risk_sustained_secs"] = risk_info["sustained_secs"]

            # ── Risk-gated incident logging (Layer 4) ──
            # Only log incidents and fire alerts when risk engine clears both
            # threshold AND minimum sustained duration checks.
            risk_cleared = risk_engine.should_alert(cam_id)

            # ── Recording-gated: only write to archive if recording is enabled ──
            # When auto-record is OFF and manual-record is OFF, anomalies are
            # shown on the live feed but nothing is persisted to the incident DB.
            recording_is_enabled = auto_record_enabled or manual_record_enabled

            if anomaly_detected and recording_is_enabled:
                severity = "HIGH" if phone_detected else "LOW"
                event_type = "unauthorized_object" if phone_detected else "gaze_anomaly" if gaze_anomaly else "unusual_movement"
                labels_for_event = [det["class"] for det in detections]
                if gaze_anomaly:
                    labels_for_event.append("gaze anomaly")
                if movement_anomaly:
                    labels_for_event.append("unusual movement")

                # Log proctor events + anomaly scores for audit trail
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

                # Rate-limited incident log → Incident Archive.
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

            elif anomaly_detected and not recording_is_enabled:
                # Anomaly visible on live feed but not recorded — just update risk engine,
                # no DB writes, no archive entries, no SocketIO alert persisted.
                print(f"[ROUTER] Anomaly detected on cam {cam_id} — skipped (auto-record OFF)")

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
    """Serve the latest ANNOTATED frame (with YOLO boxes already drawn by the detector).
    Previously this read the raw camera frame and re-drew from shared_state detections,
    which produced stale/mismatched overlays.  Now we use latest_annotated_frames which
    is updated by each detection worker immediately after inference.
    """
    print("[ENCODER] Zero-latency MJPEG stream started.")
    cam_id = config.CAMERA_SOURCES[0]
    fps_counter = 0
    fps_timer = time.time()

    while not shutdown_event.is_set():
        # Prefer annotated (post-detection) frame; fall back to raw camera frame
        with annotated_frames_lock:
            draw_frame = latest_annotated_frames.get(cam_id)
        if draw_frame is None:
            draw_frame = cam_stream.latest_frames.get(cam_id)
        if draw_frame is None:
            time.sleep(0.05)
            continue

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
    import torch as _torch
    cpu_percent = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory()
    # VRAM stats when CUDA is active
    vram_total_mb = 0
    vram_used_mb = 0
    if config.profile.has_cuda and _torch.cuda.is_available():
        vram_total_mb = int(_torch.cuda.get_device_properties(0).total_memory / 1e6)
        vram_used_mb = int(_torch.cuda.memory_allocated(0) / 1e6)
    infer_device = detectors[0].device if detectors else config.ACCELERATION_BACKEND
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
        "vram_total_mb": vram_total_mb,
        "vram_used_mb": vram_used_mb,
        "inference_device": infer_device,
        "acceleration_backend": config.ACCELERATION_BACKEND,
        "onnx_providers": config.profile.onnx_providers,
        "yolo_conf": detectors[0].conf_thresh if detectors else 0.3,
        "yolo_half": detectors[0].use_half if detectors else False,
        "detect_every_n": config.DETECT_EVERY_N,
        "detection_resolution": f"{config.DETECTION_RESOLUTION[0]}x{config.DETECTION_RESOLUTION[1]}",
        "clip_resolution": f"{config.CLIP_RESOLUTION[0]}x{config.CLIP_RESOLUTION[1]}",
        "clip_codec": config.CLIP_CODEC,
        "object_det_enabled": config.ENABLE_OBJECT_DET,
        "gaze_enabled": config.ENABLE_GAZE,
        "frame_queue_size": frame_queue.size(),
        "frame_drop_rate": round(frame_queue.get_drop_rate() * 100, 1),
        "num_workers": len(detectors),
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
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            delay = 1.0 / fps
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
                time.sleep(delay)
        except GeneratorExit:
            pass  # Client disconnected — normal exit, release below
        finally:
            cap.release()  # Always release — previously leaked on client disconnect

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

@app.route('/enroll', methods=['POST'])
def enroll_student():
    """
    Enroll an authorized person using their ID card / admit card photo.
    Accepts multipart/form-data with:
      - file: image file of the ID card (JPEG/PNG)
      - roll_number: student roll number or ID string
      - name: student name (optional)
    Extracts the face from the ID card, saves the face image to the applicants
    directory, and updates examinees.json. The identity manager reloads embeddings
    in a background thread so the route returns immediately.
    """
    import json as _json

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    roll_number = request.form.get('roll_number', '').strip()
    name = request.form.get('name', roll_number).strip()
    if not roll_number:
        return jsonify({"success": False, "error": "roll_number is required"}), 400

    import numpy as np
    file = request.files['file']
    file_bytes = file.read()
    img_array = np.frombuffer(file_bytes, np.uint8)
    id_card_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if id_card_img is None:
        return jsonify({"success": False, "error": "Could not decode image"}), 400

    # Try to extract face from the ID card using DeepFace
    face_img = id_card_img  # fallback: use full card image if face detection fails
    deepface = __import__('identity').get_deepface()
    if deepface is not None:
        try:
            faces = deepface.extract_faces(
                img_path=id_card_img,
                detector_backend='opencv',
                enforce_detection=False
            )
            if faces and faces[0].get('confidence', 0) > 0.5:
                facial_area = faces[0]['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                # Add 20% padding for better embedding quality
                pad = int(max(w, h) * 0.2)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(id_card_img.shape[1], x + w + pad)
                y2 = min(id_card_img.shape[0], y + h + pad)
                face_img = id_card_img[y1:y2, x1:x2]
                print(f"[ENROLL] Face extracted from ID card for {roll_number} (conf={faces[0]['confidence']:.2f})")
            else:
                print(f"[ENROLL] No clear face found on ID card for {roll_number}, using full image")
        except Exception as e:
            print(f"[ENROLL] Face extraction failed for {roll_number}: {e}, using full card image")

    # Save face image to applicants directory
    applicants_dir = os.path.join(proj_root, "..", "applicants")
    os.makedirs(applicants_dir, exist_ok=True)
    face_path = os.path.join(applicants_dir, f"student_{roll_number}.jpg")
    cv2.imwrite(face_path, face_img)

    # Also save original ID card alongside
    card_path = os.path.join(applicants_dir, f"idcard_{roll_number}.jpg")
    cv2.imwrite(card_path, id_card_img)

    # Update examinees.json
    roster_path = os.path.join(proj_root, "examinees.json")
    roster = []
    if os.path.exists(roster_path):
        try:
            with open(roster_path, "r", encoding="utf-8") as f:
                roster = _json.load(f)
        except Exception:
            roster = []

    # Remove existing entry for this roll_number, then append updated one
    roster = [s for s in roster if str(s.get("roll_number")) != str(roll_number)]
    roster.append({"roll_number": roll_number, "name": name})
    with open(roster_path, "w", encoding="utf-8") as f:
        _json.dump(roster, f, indent=2)

    # Reload identity embeddings in background
    def _reload():
        identity_mgr.loaded = False
        identity_mgr.examinees_path = roster_path
        identity_mgr._load_examinees()
        print(f"[ENROLL] Embeddings reloaded: {len(identity_mgr.known_embeddings)} authorized identities")
        socketio.emit("enrollment_complete", {
            "roll_number": roll_number,
            "name": name,
            "total_enrolled": len(identity_mgr.known_embeddings)
        })

    threading.Thread(target=_reload, daemon=True, name="EnrollReload").start()

    return jsonify({
        "success": True,
        "roll_number": roll_number,
        "name": name,
        "face_saved": face_path,
        "message": f"Enrolled {name} ({roll_number}). Embeddings reloading in background."
    })


@app.route('/intruders')
def get_intruders():
    """Return all logged intruder detections from the identity DB."""
    records = evidence.get_all_intruders()
    for r in records:
        r['timestamp_local'] = datetime.datetime.fromtimestamp(r['timestamp']).isoformat()
    return jsonify(records)


@app.route('/intruders/delete', methods=['POST'])
def delete_intruder():
    data = request.json or {}
    ids = data.get('ids', [])
    if ids:
        with evidence.db_lock:
            placeholders = ','.join('?' * len(ids))
            evidence.conn.execute(f'DELETE FROM intruders WHERE id IN ({placeholders})', ids)
            evidence.conn.commit()
    return jsonify({'success': True, 'deleted': len(ids)})


@app.route('/intruders/<int:intruder_id>/adjudicate', methods=['POST'])
def adjudicate_intruder(intruder_id):
    """Admin-only: confirm or clear an intruder flag.
    Body: { "confirmed": true|false }
    """
    data = request.json or {}
    confirmed = bool(data.get('confirmed', False))
    evidence.adjudicate_intruder(intruder_id, confirmed)
    status = 'CONFIRMED_INTRUDER' if confirmed else 'CLEARED'
    print(f"[ADJUDICATE] Intruder #{intruder_id} → {status}")
    return jsonify({'success': True, 'id': intruder_id, 'confirmed': confirmed, 'status': status})


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


# ── Metropolis Status Endpoint ──
@app.route('/metropolis/status')
def metropolis_status():
    """Return current Metropolis pipeline status and capabilities."""
    if not _METROPOLIS_ENABLED or _metropolis_orchestrator is None:
        return jsonify({
            "enabled": False,
            "active_pipeline": "legacy",
            "capabilities": None,
        })
    caps = _metropolis_orchestrator.capabilities
    return jsonify({
        "enabled": True,
        "active_pipeline": _metropolis_orchestrator.active_pipeline,
        "is_running": _metropolis_orchestrator.is_running,
        "capabilities": {
            "has_gpu": caps.has_gpu if caps else False,
            "gpu_name": caps.gpu_name if caps else None,
            "has_tensorrt": caps.has_tensorrt if caps else False,
            "has_deepstream": caps.has_deepstream if caps else False,
            "has_triton": caps.has_triton if caps else False,
        } if caps else None,
        "config": _metropolis_orchestrator.config.to_dict(),
    })

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def shutdown():
    if shutdown_event.is_set():
        return
    print("[SHUTDOWN] Flushing queues, closing cameras, saving session.")
    shutdown_event.set()
    # Stop Metropolis orchestrator if running
    if _METROPOLIS_ENABLED and _metropolis_orchestrator is not None:
        try:
            if _metropolis_orchestrator.is_running:
                _metropolis_orchestrator.stop()
                print("[METROPOLIS] Pipeline stopped")
        except Exception as e:
            print(f"[METROPOLIS] Error during shutdown: {e}")
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
        # Start Metropolis orchestrator if enabled (non-legacy pipeline)
        if _METROPOLIS_ENABLED and _metropolis_orchestrator is not None:
            active = _metropolis_orchestrator.active_pipeline
            if active and active != "legacy":
                try:
                    _metropolis_orchestrator.start()
                    print(f"[METROPOLIS] Pipeline '{active}' started successfully")
                except Exception as e:
                    print(f"[METROPOLIS] Failed to start pipeline: {e}. Falling back to legacy.")

        # Legacy pipeline always starts — it serves as the default and fallback
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
