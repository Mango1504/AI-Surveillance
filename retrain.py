# clean_dataset.py
# Run this in: C:\Users\write\Desktop\AI Surveillance\
# Install first: pip install roboflow requests

from roboflow import Roboflow
import os
import json

# ── CONFIG ──────────────────────────────────────────────────────────────
API_KEY       = "YOUR_ROBOFLOW_API_KEY"   # get from roboflow.com → settings → API key
WORKSPACE     = "phonedetection-dyjez"
PROJECT       = "mobile-phone-detection-mtsje-xhoma-hjvql"
VERSION       = 1
JUNK_CLASSES  = ['6', 'undefined']        # classes to audit
# ────────────────────────────────────────────────────────────────────────

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

# Step 1 — Download dataset locally to inspect
print("Downloading dataset...")
dataset = version.download("yolov8", location="./audit_dataset")

# Step 2 — Scan all label files and report junk class stats
label_dirs = [
    "./audit_dataset/train/labels",
    "./audit_dataset/valid/labels",
    "./audit_dataset/test/labels",
]

CLASS_NAMES = ['6', 'phone', 'undefined']  # must match your data.yaml order
JUNK_IDS    = [CLASS_NAMES.index(c) for c in JUNK_CLASSES]  # [0, 2]

junk_files   = []
clean_files  = []
junk_counts  = {c: 0 for c in JUNK_CLASSES}

print("\nScanning labels...")
for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        continue
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(label_dir, fname)
        with open(fpath) as f:
            lines = f.readlines()

        has_junk  = False
        has_phone = False
        for line in lines:
            cls_id = int(line.strip().split()[0])
            if cls_id in JUNK_IDS:
                has_junk = True
                junk_counts[CLASS_NAMES[cls_id]] += 1
            if cls_id == 1:  # phone class
                has_phone = True

        if has_junk:
            junk_files.append(fpath)
        else:
            clean_files.append(fpath)

# Step 3 — Report
print(f"\n{'='*45}")
print(f"  Total label files scanned : {len(junk_files) + len(clean_files)}")
print(f"  Clean (phone only)        : {len(clean_files)}")
print(f"  Files with junk classes   : {len(junk_files)}")
for cls, count in junk_counts.items():
    print(f"    '{cls}' annotations found : {count}")
print(f"{'='*45}\n")

# Step 4 — Ask what to do
print("Options:")
print("  1 — Strip junk annotations but KEEP the image (if it also has 'phone')")
print("  2 — Delete junk annotation files entirely (removes image from training)")
print("  3 — Just show me the filenames, don't change anything")
choice = input("\nEnter 1, 2, or 3: ").strip()

if choice == "3":
    print("\nJunk files:")
    for f in junk_files:
        print(" ", f)

elif choice == "1":
    # Remove junk lines, keep phone lines
    fixed = 0
    removed = 0
    for fpath in junk_files:
        with open(fpath) as f:
            lines = f.readlines()
        clean_lines = [l for l in lines if int(l.strip().split()[0]) not in JUNK_IDS]
        if clean_lines:
            with open(fpath, "w") as f:
                f.writelines(clean_lines)
            fixed += 1
        else:
            # No phone annotations left — remove the file
            os.remove(fpath)
            # Also remove the corresponding image
            img_path = fpath.replace("/labels/", "/images/").replace(".txt", ".jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            removed += 1
    print(f"\nDone. Fixed: {fixed} files | Fully removed: {removed} files")

elif choice == "2":
    removed = 0
    for fpath in junk_files:
        os.remove(fpath)
        img_path = fpath.replace("/labels/", "/images/").replace(".txt", ".jpg")
        if os.path.exists(img_path):
            os.remove(img_path)
        removed += 1
    print(f"\nRemoved {removed} label+image pairs.")

# Step 5 — Update data.yaml to single class
if choice in ("1", "2"):
    yaml_path = "./audit_dataset/data.yaml"
    with open(yaml_path) as f:
        content = f.read()
    content = content.replace("nc: 3", "nc: 1")
    content = content.replace("names: ['6', 'phone', 'undefined']", "names: ['phone']")
    with open(yaml_path, "w") as f:
        f.write(content)
    print("\nUpdated data.yaml → nc: 1, names: ['phone']")
    print("\nNext step: re-run your train.py pointing to ./audit_dataset/data.yaml")
    print("Recommended: 150 epochs, patience=30")