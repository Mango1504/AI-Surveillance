import os
import threading
import time
from typing import Any

import cv2
from ultralytics import YOLO

from config import get_config


class MultiCameraStream:
    """Capture one full-resolution stream per camera and push frames downstream."""

    def __init__(self, sources, frame_queue):
        self.config = get_config()
        self.sources = sources
        self.frame_queue = frame_queue
        self.streams = {}
        self.threads = {}
        self.latest_frames = {}
        self.stopped = False

        for source in sources:
            caps = self.config.profile.camera_caps.get(source, {})
            width, height = caps.get("max_res", self.config.DEFAULT_CAPTURE_RES)
            backend = cv2.CAP_FFMPEG if isinstance(source, str) and source.lower().startswith("rtsp") else 0
            stream = cv2.VideoCapture(source, backend) if backend else cv2.VideoCapture(source)
            stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            stream.set(cv2.CAP_PROP_FPS, caps.get("max_fps", self.config.DEFAULT_CAPTURE_FPS))
            if caps.get("mjpeg"):
                stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.streams[source] = stream

    def start(self):
        for source, stream in self.streams.items():
            thread = threading.Thread(target=self._update, args=(source, stream), daemon=True, name=f"Camera-{source}")
            self.threads[source] = thread
            thread.start()
        return self

    def _update(self, source, stream):
        while not self.stopped:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                time.sleep(0.05)
                continue
            self.latest_frames[source] = frame
            self.frame_queue.push((time.time(), source, frame))

    def stop(self):
        self.stopped = True
        for stream in self.streams.values():
            stream.release()


class ExamDetector:
    """Adaptive YOLO detector. Full-res frames are preserved; inference uses a working copy."""

    def __init__(self, conf_thresh=0.50):
        self.config = get_config()
        model_name = self.config.OBJECT_MODEL
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(project_root, "yolov8n.pt")
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"

        print(f"[YOLO] Initializing {os.path.basename(model_path)} tier={self.config.profile.tier} backend={self.config.ACCELERATION_BACKEND}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.target_classes = [0, 67]  # person, cell phone
        self.use_half = self.config.profile.has_cuda  # Half-precision on CUDA for ~2x speed

    def _working_frame(self, frame):
        target_w, target_h = self.config.DETECTION_RESOLUTION
        if frame.shape[1] == target_w and frame.shape[0] == target_h:
            return frame, 1.0, 1.0
        if self.config.profile.has_opencl:
            resized = cv2.UMat(frame)
            resized = cv2.resize(resized, (target_w, target_h)).get()
        else:
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        scale_x = frame.shape[1] / float(target_w)
        scale_y = frame.shape[0] / float(target_h)
        return resized, scale_x, scale_y

    def detect(self, frame):
        work, scale_x, scale_y = self._working_frame(frame)
        imgsz = max(self.config.DETECTION_RESOLUTION)
        results = self.model(
            work,
            imgsz=imgsz,
            verbose=False,
            conf=self.conf_thresh,
            classes=self.target_classes,
            half=self.use_half,
        )

        annotated_frame = frame.copy()
        detections = []
        for result in results:
            boxes = result.boxes
            for i, cls_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
                conf = float(boxes.conf[i].cpu())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                class_name = self.model.names[cls_id]
                color = (0, 0, 255) if class_name == "cell phone" else (0, 200, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{class_name} {conf:.2f}",
                    (x1, max(16, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                })

        return annotated_frame, detections
