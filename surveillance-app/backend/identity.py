import json
import os
import time

import cv2
import numpy as np

from config import get_config


DeepFace = None
_deepface_checked = False


def get_deepface():
    global DeepFace, _deepface_checked
    if _deepface_checked:
        return DeepFace
    _deepface_checked = True
    try:
        from deepface import DeepFace as _DeepFace
        DeepFace = _DeepFace
    except Exception as exc:
        DeepFace = None
        print(f"[WARNING] DeepFace unavailable ({exc}). Biometrics will be mocked.")
    return DeepFace


class IdentityManager:
    """Session-scoped identity matching via DeepFace, loaded lazily."""

    def __init__(self, examinees_json_path, image_dir_path, threshold=0.28, autoload=False):
        self.config = get_config()
        self.threshold = threshold
        self.known_embeddings = {}
        self.examinees_path = examinees_json_path
        self.image_dir_path = image_dir_path
        self.verify_cache = {}  # face_hash -> (student_id, expire_time)
        self.loaded = False
        if autoload:
            self._load_examinees()

    def _load_examinees(self):
        if self.loaded:
            return
        self.loaded = True
        if not os.path.exists(self.examinees_path):
            return

        deepface = get_deepface()
        if deepface is None:
            return

        with open(self.examinees_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        print("[IDENTITY] Generating session embeddings...")
        max_identities = self.config.EMBEDDING_CACHE_SIZE

        for idx, student in enumerate(data):
            if idx >= max_identities:
                print(f"[IDENTITY] Capped loaded identities at {max_identities} due to RAM budget.")
                break

            roll_number = student.get("roll_number")
            candidates = [
                os.path.join(self.image_dir_path, f"student_{roll_number}.jpg"),
                os.path.join(self.image_dir_path, f"{roll_number}.jpg"),
            ]
            img_path = next((path for path in candidates if roll_number and os.path.exists(path)), None)
            if not img_path:
                continue

            try:
                embeds = deepface.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)
                if embeds:
                    self.known_embeddings[str(roll_number)] = embeds[0]["embedding"]
            except Exception as exc:
                print(f"[IDENTITY] Error extracting embedding for {roll_number}: {exc}")

        print(f"[IDENTITY] Loaded {len(self.known_embeddings)} authorized identities.")

    def verify_face(self, face_crop):
        """Return student_id string if matched, 'Unknown' otherwise."""
        self._load_examinees()
        deepface = get_deepface()
        if deepface is None or not self.known_embeddings:
            return "Unknown"

        # Evict expired cache entries (30-second TTL)
        now = time.time()
        for key in list(self.verify_cache.keys()):
            student_id, expire_at = self.verify_cache[key]
            if now > expire_at:
                del self.verify_cache[key]

        try:
            temp_path = os.path.join(os.path.dirname(self.examinees_path), "_temp_face.jpg")
            cv2.imwrite(temp_path, face_crop)
            embeds = deepface.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=False)
            if not embeds:
                return "Unknown"

            face_embed = embeds[0]["embedding"]
            best_match = "Unknown"
            best_dist = float("inf")

            for student_id, known_embed in self.known_embeddings.items():
                dist = np.linalg.norm(np.array(face_embed) - np.array(known_embed))
                if dist < best_dist:
                    best_dist = dist
                    best_match = student_id

            if best_dist < self.threshold:
                self.verify_cache[id(face_crop)] = (best_match, now + 30)
                return best_match
        except Exception:
            return "Unknown"

        return "Unknown"

    def is_intruder(self, face_crop) -> bool:
        """Return True if the face is not in the session roster (unknown person)."""
        if not self.known_embeddings:
            return False  # No roster loaded — can't determine intruder status
        result = self.verify_face(face_crop)
        return result == "Unknown"

    def purge_data(self):
        self.known_embeddings.clear()
        self.verify_cache.clear()
        self.loaded = False
        print("[IDENTITY] All biometric embeddings purged from memory.")
