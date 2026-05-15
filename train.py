"""
Training Pipeline for YOLO26 — train.py
Optimized for RTX 3060 Notebook (6GB VRAM)

Run each step separately:
    python train.py --step 1   # collect frames
    python train.py --step 3   # train (after labelling)
    python train.py --step 4   # validate
    python train.py --step 5   # swap model into second.py
"""

import cv2
import os
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────
# CONFIG — RTX 3060 Notebook (6GB VRAM)
# ──────────────────────────────────────────────
CAMERA_INDEX     = 0
DATASET_DIR      = r"C:/Users/write/Desktop/Phone/dataset"
FRAMES_TO_SAVE   = 500
FRAME_INTERVAL   = 0.2

EPOCHS           = 300           # good balance for fine-tuning
BATCH_SIZE       = 8             # safe for 6GB VRAM — drop to 4 if OOM error
IMAGE_SIZE       = 640           # YOLO26 default — keep at 640
WORKERS          = 4             # 3060 notebook: 4 workers is optimal
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"

CLASS_NAMES = ["phone", "laptop"]


# ──────────────────────────────────────────────
# STEP 1 — Collect frames from your camera
# ──────────────────────────────────────────────
def collect_frames():
    save_dir = Path(DATASET_DIR) / "raw_frames"
    save_dir.mkdir(parents=True, exist_ok=True)

    cap   = cv2.VideoCapture(CAMERA_INDEX)
    count = 0
    last  = 0

    print(f"[COLLECT] Saving {FRAMES_TO_SAVE} frames to {save_dir}")
    print("  Press Q to stop early\n")

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

        cv2.putText(frame, f"Saved: {count}/{FRAMES_TO_SAVE}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collecting frames — Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[COLLECT] Done. {count} frames saved to:\n  {save_dir}")


# ──────────────────────────────────────────────
# STEP 2 — Download dataset from Roboflow
# ──────────────────────────────────────────────
def download_dataset():
    from roboflow import Roboflow

    WORKSPACE = "your-workspace-name"
    PROJECT   = "your-project-name"
    VERSION   = 1

    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(
        "yolov8",
        location=str(Path(DATASET_DIR) / "labelled")
    )
    print(f"\n[DOWNLOAD] Dataset saved to: {dataset.location}")
    return dataset.location


# ──────────────────────────────────────────────
# STEP 3 — Train  (RTX 3060 Notebook optimized)
# ──────────────────────────────────────────────
def train(data_yaml: str = None):
    if data_yaml is None:
        candidates = list(Path(DATASET_DIR).rglob("data.yaml"))
        if not candidates:
            print("[ERROR] data.yaml not found. Pass --yaml path/to/data.yaml")
            return
        data_yaml = str(candidates[0])

    print(f"[TRAIN] Dataset  : {data_yaml}")
    print(f"[TRAIN] Epochs   : {EPOCHS}")
    print(f"[TRAIN] Batch    : {BATCH_SIZE}")
    print(f"[TRAIN] ImgSize  : {IMAGE_SIZE}")
    print(f"[TRAIN] Device   : RTX 3060 Notebook (cuda:0)")
    print(f"[TRAIN] FP16     : Enabled (half precision — saves VRAM)\n")

    model = YOLO("yolo26n.pt", task="detect")

    results = model.train(
        data        = data_yaml,
        epochs      = EPOCHS,
        batch       = BATCH_SIZE,
        imgsz       = IMAGE_SIZE,

        # ── RTX 3060 Notebook specific ──────────────
        device      = "0",          # force GPU cuda:0
        half        = True,         # FP16 — halves VRAM usage, same accuracy
        workers     = WORKERS,      # 4 workers optimal for laptop
        cache       = False,        # False = safer on 6GB VRAM
        amp         = True,         # automatic mixed precision — faster on 3060
        # ────────────────────────────────────────────

        patience    = 50,
        save        = True,
        plots       = True,
        augment     = True,
        degrees     = 10,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        project     = str(Path(DATASET_DIR) / "runs"),
        name        = "phone_laptop_yolo26n",
        verbose     = True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n[TRAIN] Done!")
    print(f"        Best weights : {best_weights}")
    print(f"        mAP50        : {results.results_dict.get('metrics/mAP50(B)', 'see plots')}")
    print(f"\nNEXT: python train.py --step 4  to validate")
    return str(best_weights)


# ──────────────────────────────────────────────
# STEP 4 — Validate
# ──────────────────────────────────────────────
def validate(weights_path: str = None, data_yaml: str = None):
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
    print(f"   mAP50 > 0.75 → good")
    print(f"   mAP50 > 0.85 → very good")
    print(f"   mAP50 < 0.60 → need more labelled data\n")


# ──────────────────────────────────────────────
# STEP 5 — Swap model into second.py
# ──────────────────────────────────────────────
def swap_model(weights_path: str = None):
    if weights_path is None:
        candidates = list(Path(DATASET_DIR).rglob("best.pt"))
        if not candidates:
            print("[ERROR] No best.pt found. Train first with --step 3")
            return
        weights_path = str(max(candidates, key=os.path.getmtime))

    dest = r"C:/Users/write/Desktop/Phone/best.pt"
    import shutil
    shutil.copy2(weights_path, dest)

    print(f"[SWAP] Copied to: {dest}")
    print(f"\nIn second.py change:")
    print(f'   model = YOLO("yolo26n.pt", task="detect", verbose=False)')
    print(f"to:")
    print(f'   model = YOLO(r"{dest}", task="detect", verbose=False)')


# ──────────────────────────────────────────────
# STEP 6 — Live test
# ──────────────────────────────────────────────
def live_test(weights_path: str = None):
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
        cv2.imshow("Live test — Q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 — RTX 3060 Notebook")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="1=collect 2=download 3=train 4=validate 5=swap 6=live-test")
    parser.add_argument("--yaml",    type=str, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, help="Path to best.pt")
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
