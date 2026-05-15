import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO

# --- MODULE 1: Video Capture & Preprocessing ---
class CameraStream:
    """Multithreaded camera capture to ensure 0 latency (always returns the latest frame)."""
    def __init__(self, src=0, width=1920, height=1080):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed:
                time.sleep(0.05)
                
    def read(self):
        # Apply preprocessing here if needed (e.g., resize, denoise)
        if self.frame is not None:
            # Fast denoising (optional, disabled by default for performance)
            # frame = cv2.fastNlMeansDenoisingColored(self.frame, None, 10, 10, 7, 21)
            
            # Normalization (YOLO does this internally, but this is how you'd do it manually)
            # frame = cv2.normalize(self.frame, None, 0, 255, cv2.NORM_MINMAX)
            return self.grabbed, self.frame.copy()
        return self.grabbed, self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# --- MODULE 2: YOLOv8 Object Detection ---
class ExamDetector:
    def __init__(self, model_path="yolov8m.pt", conf_thresh=0.45):
        # Load YOLO model (will download if not present)
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        
        # Mapping COCO classes to our needs (until custom model is trained)
        # 0: person, 67: cell phone
        self.target_classes = [0, 67] 
        
    def detect(self, frame):
        """Runs inference and returns annotated frame + detection list."""
        # imgsz=1088 helps with small object detection (like phones) in 1080p
        results = self.model(frame, imgsz=1088, verbose=False, conf=self.conf_thresh, classes=self.target_classes)
        
        detections = []
        annotated_frame = results[0].plot()
        
        for result in results:
            boxes = result.boxes
            for i, cls_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
                    
                conf = float(boxes.conf[i].cpu())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                class_name = self.model.names[cls_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
                
        return annotated_frame, detections

# --- Testing the Modules ---
if __name__ == "__main__":
    print("[INFO] Starting video stream...")
    cam = CameraStream(src=0).start()
    
    print("[INFO] Loading YOLOv8 model...")
    # Change to your custom model path when ready
    detector = ExamDetector(model_path="yolov8m.pt", conf_thresh=0.30)
    
    # Warmup
    time.sleep(2.0)
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    print("[INFO] System running. Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
            
        start_time = time.perf_counter()
        
        # Run detection
        annotated_frame, detections = detector.detect(frame)
        
        # Calculate Latency & FPS
        latency_ms = (time.perf_counter() - start_time) * 1000
        fps_counter += 1
        if (time.time() - fps_start_time) > 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
            
        # Draw stats
        cv2.putText(annotated_frame, f"FPS: {fps} | Latency: {latency_ms:.1f}ms", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        # Log detections to console
        if detections:
            print(f"Detected: {[d['class'] for d in detections]} in {latency_ms:.1f}ms")
            
        cv2.imshow("Exam Proctoring - Module 1 & 2", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cam.stop()
    cv2.destroyAllWindows()
