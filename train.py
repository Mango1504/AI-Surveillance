"""
Training Pipeline for YOLO26 — train.py

STEP 1: Run collect_frames() to grab images from your camera
STEP 2: Label them on Roboflow (free) or use label_with_roboflow() below
STEP 3: Run train() with your dataset
STEP 4: Run swap_model() to plug the trained model into second.py

Run each step separately:
    python train.py --step 1   # collect frames
    python train.py --step 3   # train (after labelling)
    python train.py --step 4   # swap model in second.py
"""

import cv2
import os
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────
# CONFIG — edit these
# ──────────────────────────────────────────────
CAMERA_INDEX     = 0
DATASET_DIR      = r"C:/Users/write/Desktop/Phone/dataset"
FRAMES_TO_SAVE   = 500          # how many frames to collect (aim for 300-500+)
FRAME_INTERVAL   = 0.2          # seconds between saved frames (0.2 = 5 per second)
EPOCHS           = 100          # 50-100 for fine-tuning, 200+ for training from scratch
BATCH_SIZE       = 8            # lower if you get out-of-memory errors (try 4)
IMAGE_SIZE       = 640          # keep at 640 — YOLO26 default
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"   # get free at roboflow.com

# Classes you are training for
# Must match what you labelled in Roboflow
CLASS_NAMES = ["phone", "laptop"]


# ──────────────────────────────────────────────
# STEP 1 — Collect frames from your camera
# ──────────────────────────────────────────────
def collect_frames():
    """
    Saves raw frames from your camera into dataset/images/
    You then upload these to Roboflow to label them.

    Tips for good data:
    - Vary the positions of phones/laptops (different seats, angles)
    - Include frames with NO phones/laptops (negative samples)
    - Capture at different times of day (lighting changes)
    - Move the objects closer and further from the camera
    """
    save_dir = Path(DATASET_DIR) / "raw_frames"
    save_dir.mkdir(parents=True, exist_ok=True)

    cap   = cv2.VideoCapture(CAMERA_INDEX)
    count = 0
    last  = 0

    print(f"[COLLECT] Saving {FRAMES_TO_SAVE} frames to {save_dir}")
    print("  Press Q to stop early, SPACE to skip a frame\n")

    while count < FRAMES_TO_SAVE:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last >= FRAME_INTERVAL:
            filename = save_dir / f"frame_{count:05d}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
            last = now
            print(f"  Saved {count}/{FRAMES_TO_SAVE}", end="\r")

        # Show live preview
        cv2.putText(frame, f"Saved: {count}/{FRAMES_TO_SAVE}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collecting frames — Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[COLLECT] Done. {count} frames saved to:\n  {save_dir}")
    print("\nNEXT STEP:")
    print("  1. Go to https://roboflow.com and create a free account")
    print("  2. Create a new project → Object Detection")
    print(f"  3. Upload all images from: {save_dir}")
    print("  4. Draw bounding boxes around phones and laptops")
    print("  5. Generate a dataset version")
    print("  6. Export as 'YOLOv8 PyTorch' format (works with YOLO26)")
    print("  7. Copy the export snippet and paste your API key below\n")


# ──────────────────────────────────────────────
# STEP 2 — Download labelled dataset from Roboflow
# ──────────────────────────────────────────────
def download_dataset():
    """
    After labelling on Roboflow, download the dataset directly.
    Fill in your workspace, project, and version from the Roboflow export page.
    """
    from roboflow import Roboflow

    # ── Replace these with values from your Roboflow export page ──────
    WORKSPACE   = "your-workspace-name"
    PROJECT     = "your-project-name"
    VERSION     = 1
    # ──────────────────────────────────────────────────────────────────

    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(
        "yolov8",                            # YOLO26 uses the same format
        location=str(Path(DATASET_DIR) / "labelled")
    )
    print(f"\n[DOWNLOAD] Dataset saved to: {dataset.location}")
    print(f"           data.yaml is at:   {dataset.location}/data.yaml")
    return dataset.location


# ──────────────────────────────────────────────
# STEP 3 — Train
# ──────────────────────────────────────────────
def train(data_yaml: str = None):
    """
    Fine-tunes YOLO26n on your labelled dataset.
    Starting from the pretrained yolo26n.pt weights (transfer learning)
    means you need far less data and fewer epochs than training from scratch.

    After training, best weights are saved to:
        runs/detect/train/weights/best.pt
    """
    if data_yaml is None:
        # Try to find data.yaml automatically
        candidates = list(Path(DATASET_DIR).rglob("data.yaml"))
        if not candidates:
            print("[ERROR] data.yaml not found. Run --step 2 first, or pass --yaml path/to/data.yaml")
            return
        data_yaml = str(candidates[0])

    print(f"[TRAIN] Using dataset: {data_yaml}")
    print(f"[TRAIN] Epochs={EPOCHS}  Batch={BATCH_SIZE}  ImgSize={IMAGE_SIZE}")
    print(f"[TRAIN] Starting from pretrained yolo26n.pt (transfer learning)\n")

    model = YOLO("yolo26n.pt", task="detect")

    results = model.train(
        data        = data_yaml,
        epochs      = EPOCHS,
        batch       = BATCH_SIZE,
        imgsz       = IMAGE_SIZE,
        patience    = 20,           # stop early if no improvement for 20 epochs
        save        = True,
        plots       = True,         # saves confusion matrix, PR curve etc.
        augment     = True,         # random flips, crops, brightness — improves robustness
        degrees     = 10,           # random rotation ±10° (useful for angled cameras)
        hsv_h       = 0.015,        # hue augmentation
        hsv_s       = 0.7,          # saturation augmentation
        hsv_v       = 0.4,          # brightness augmentation
        project     = str(Path(DATASET_DIR) / "runs"),
        name        = "phone_laptop_yolo26n",
        verbose     = True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[TRAIN] Done!")
    print(f"        Best weights: {best_weights}")
    print(f"        mAP50:        {results.results_dict.get('metrics/mAP50(B)', 'see plots')}")
    print(f"\nNEXT STEP: Run  python train.py --step 4  to plug this into second.py")
    return str(best_weights)


# ──────────────────────────────────────────────
# STEP 4 — Validate the trained model
# ──────────────────────────────────────────────
def validate(weights_path: str = None, data_yaml: str = None):
    """
    Runs validation on your test set and prints mAP, precision, recall.
    mAP50 > 0.75 is good. > 0.85 is very good.
    """
    if weights_path is None:
        candidates = list(Path(DATASET_DIR).rglob("best.pt"))
        if not candidates:
            print("[ERROR] No best.pt found. Train first with --step 3")
            return
        weights_path = str(candidates[0])

    if data_yaml is None:
        candidates = list(Path(DATASET_DIR).rglob("data.yaml"))
        data_yaml   = str(candidates[0]) if candidates else None

    print(f"[VAL] Weights: {weights_path}")
    model   = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=IMAGE_SIZE, verbose=True)

    print(f"\n── Validation Results ─────────────────────")
    print(f"   mAP50:       {metrics.box.map50:.3f}")
    print(f"   mAP50-95:    {metrics.box.map:.3f}")
    print(f"   Precision:   {metrics.box.p.mean():.3f}")
    print(f"   Recall:      {metrics.box.r.mean():.3f}")
    print(f"───────────────────────────────────────────")
    print(f"\nInterpretation:")
    print(f"   mAP50 > 0.75  → good for surveillance use")
    print(f"   mAP50 > 0.85  → very good")
    print(f"   mAP50 < 0.60  → need more/better labelled data\n")


# ──────────────────────────────────────────────
# STEP 5 — Swap model into second.py
# ──────────────────────────────────────────────
def swap_model(weights_path: str = None):
    """
    Copies your best.pt to Desktop and prints the line to change in second.py.
    """
    if weights_path is None:
        candidates = list(Path(DATASET_DIR).rglob("best.pt"))
        if not candidates:
            print("[ERROR] No best.pt found. Train first with --step 3")
            return
        weights_path = str(max(candidates, key=os.path.getmtime))

    dest = r"C:/Users/write/Desktop/Phone/best.pt"
    import shutil
    shutil.copy2(weights_path, dest)

    print(f"[SWAP] Copied trained model to:\n   {dest}")
    print(f"\nIn second.py, change this line:")
    print(f'   model = YOLO("yolo26n.pt", task="detect", verbose=False)')
    print(f"to:")
    print(f'   model = YOLO(r"{dest}", task="detect", verbose=False)')
    print(f"\nThen restart second.py — it will now use your custom-trained model.")


# ──────────────────────────────────────────────
# LIVE TEST — preview the trained model
# ──────────────────────────────────────────────
def live_test(weights_path: str = None):
    """Quick live preview using your trained weights."""
    if weights_path is None:
        candidates = list(Path(DATASET_DIR).rglob("best.pt"))
        weights_path = str(max(candidates, key=os.path.getmtime)) if candidates else "yolo26n.pt"

    print(f"[TEST] Loading {weights_path}")
    model = YOLO(weights_path, task="detect")
    cap   = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results   = model(frame, conf=0.4, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Live test — trained model (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 Training Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="1=collect  2=download  3=train  4=validate  5=swap  6=live-test")
    parser.add_argument("--yaml",    type=str, help="Path to data.yaml (for steps 3 & 4)")
    parser.add_argument("--weights", type=str, help="Path to best.pt (for steps 4, 5, 6)")
    args = parser.parse_args()

    if args.step == 1:
        collect_frames()
    elif args.step == 2:
        download_dataset()
    elif args.step == 3:
        train(data_yaml=args.yaml)
    elif args.step == 4:
        validate(weights_path=args.weights, data_yaml=args.yaml)
    elif args.step == 5:
        swap_model(weights_path=args.weights)
    elif args.step == 6:
        live_test(weights_path=args.weights)
    else:
        print(__doc__)