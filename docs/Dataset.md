# Dataset and Models

The system relies on a combination of pre-trained and dynamically generated datasets.

## YOLOv8 Object Detection
- **Base Model**: Uses Ultralytics YOLOv8 (nano, small, or medium weights depending on hardware tier).
- **Dataset**: Pre-trained on the COCO dataset, which contains 80 classes.
- **Specialization**: The system specifically filters for Class `67` (Cell Phone) and Class `0` (Person). Future iterations will utilize custom-trained YOLO weights fine-tuned on specific exam hall objects like smartwatches, earpieces, and unauthorized cheat sheets.

## Biometric Identity Verification
- **Model**: DeepFace library, configured to use the `VGG-Face` model by default for high accuracy face representation.
- **Dataset**: Biometrics are **not** pre-trained. The system generates a transient session dataset:
  1. An `examinees.json` roster is provided at the start of a session.
  2. The `IdentityManager` parses an `applicants/` directory containing reference ID photos of students.
  3. Facial embeddings are generated on-the-fly and kept entirely in memory.
  4. Once the session ends, the memory is purged to comply with strict biometric privacy regulations.

## Vision-Language Model (VLM)
- **Model**: Ollama running LLaVA or LLaMA3-Vision.
- **Dataset**: These are massive multimodal foundation models. They are prompted dynamically with zero-shot spatial context (e.g., "Camera 1, Row 2 Col 3") and asked to describe the anomaly in a single sentence suitable for an incident report.
