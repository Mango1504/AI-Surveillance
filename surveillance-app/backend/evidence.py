import os
import queue
import sqlite3
import threading
import time

import cv2
import numpy as np
from cryptography.fernet import Fernet

from config import get_config


class EvidenceCache:
    """Session evidence store. Existing tables are preserved; proctoring tables are migrated."""

    def __init__(self, key_path="secret.key"):
        self.config = get_config()
        self.db_path = self.config.DB_PATH
        self.key_path = key_path
        self.write_queue = queue.Queue()
        self.running = True
        self.db_lock = threading.Lock()  # Serialize ALL database operations
        self.conn_pool = []
        self.conn = self._connect()
        self._init_key()
        self._init_db()
        if self.config.profile.tier in {"HIGH", "ULTRA"}:
            self.conn_pool = [self._connect() for _ in range(4)]
        threading.Thread(target=self._writer_loop, daemon=True, name="DBWriter").start()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_key(self):
        if not os.path.exists(self.key_path):
            with open(self.key_path, "wb") as key_file:
                key_file.write(Fernet.generate_key())
        with open(self.key_path, "rb") as key_file:
            self.cipher = Fernet(key_file.read())

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                candidate_id TEXT,
                labels TEXT,
                report TEXT,
                frame_blob BLOB,
                flagged INTEGER,
                clip_path TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proctor_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL NOT NULL,
                ended_at REAL,
                tier TEXT,
                hardware_json TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proctor_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp REAL NOT NULL,
                camera_id TEXT,
                severity TEXT,
                event_type TEXT,
                candidate_id TEXT,
                score REAL,
                labels TEXT,
                clip_path TEXT,
                snapshot_path TEXT,
                details TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proctor_anomaly_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp REAL NOT NULL,
                candidate_id TEXT,
                movement_score REAL,
                gaze_score REAL,
                object_score REAL,
                aggregate_score REAL
            )
        """)
        # Identity DB — stores unauthorized persons detected during sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intruders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                camera_id TEXT,
                face_blob BLOB,
                confidence REAL,
                id_card_match TEXT,
                notes TEXT,
                adjudicated INTEGER DEFAULT 0,
                confirmed_intruder INTEGER DEFAULT NULL
            )
        """)
        self._ensure_column(cursor, "intruders", "adjudicated", "INTEGER")
        self._ensure_column(cursor, "intruders", "confirmed_intruder", "INTEGER")
        self._ensure_column(cursor, "incidents", "clip_path", "TEXT")
        self.conn.commit()

    def _ensure_column(self, cursor, table, column, column_type):
        columns = [row[1] for row in cursor.execute(f"PRAGMA table_info({table})")]
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def start_session(self):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO proctor_sessions (started_at, tier, hardware_json) VALUES (?, ?, ?)",
                (time.time(), self.config.profile.tier, str(self.config.profile)),
            )
            self.conn.commit()
            return cursor.lastrowid

    def end_session(self, session_id):
        with self.db_lock:
            self.conn.execute("UPDATE proctor_sessions SET ended_at = ? WHERE id = ?", (time.time(), session_id))
            self.conn.commit()

    def _frame_bytes(self, frame_data):
        if isinstance(frame_data, np.ndarray):
            quality = 70 if self.config.profile.tier == "LOW" else 85
            ok, buf = cv2.imencode(".jpg", frame_data, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes() if ok else b""
        if isinstance(frame_data, bytes):
            return frame_data
        return bytes(frame_data)

    def log_incident(self, candidate_id, labels, report, frame_data, flagged=1, clip_path=None):
        raw_bytes = self._frame_bytes(frame_data)
        encrypted_blob = self.cipher.encrypt(raw_bytes)
        payload = (time.time(), candidate_id, str(labels), report, encrypted_blob, flagged, clip_path)
        self.write_queue.put(("incident", payload))
        return None

    def log_proctor_event(
        self,
        session_id,
        camera_id,
        severity,
        event_type,
        candidate_id=None,
        score=0.0,
        labels=None,
        clip_path=None,
        snapshot_path=None,
        details=None,
    ):
        payload = (
            session_id,
            time.time(),
            str(camera_id),
            severity,
            event_type,
            candidate_id,
            score,
            str(labels or []),
            clip_path,
            snapshot_path,
            details,
        )
        self.write_queue.put(("proctor_event", payload))

    def log_anomaly_score(self, session_id, candidate_id, movement=0.0, gaze=0.0, objects=0.0):
        aggregate = max(movement, gaze, objects)
        payload = (session_id, time.time(), candidate_id, movement, gaze, objects, aggregate)
        self.write_queue.put(("anomaly_score", payload))

    def log_intruder(self, camera_id, face_frame, confidence=0.0, notes=""):
        """Store an unauthorized person's face crop in the identity DB."""
        raw_bytes = self._frame_bytes(face_frame)
        encrypted_blob = self.cipher.encrypt(raw_bytes) if raw_bytes else b""
        payload = (time.time(), str(camera_id), encrypted_blob, confidence, None, notes)
        self.write_queue.put(("intruder", payload))

    def get_all_intruders(self, limit=100):
        """Return stored intruder records (without decrypting face blobs)."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, camera_id, confidence, id_card_match, notes, adjudicated, confirmed_intruder
                FROM intruders
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [
                {"id": r[0], "timestamp": r[1], "camera_id": r[2],
                 "confidence": r[3], "id_card_match": r[4], "notes": r[5],
                 "adjudicated": bool(r[6]), "confirmed_intruder": r[7]}
                for r in cursor.fetchall()
            ]

    def adjudicate_intruder(self, intruder_id, confirmed: bool):
        """Admin decision: mark intruder as confirmed (True) or cleared (False)."""
        with self.db_lock:
            self.conn.execute(
                "UPDATE intruders SET adjudicated=1, confirmed_intruder=? WHERE id=?",
                (1 if confirmed else 0, intruder_id)
            )
            self.conn.commit()

    def _writer_loop(self):
        while self.running or not self.write_queue.empty():
            try:
                kind, payload = self.write_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                with self.db_lock:
                    cursor = self.conn.cursor()
                    if kind == "incident":
                        cursor.execute("""
                            INSERT INTO incidents (timestamp, candidate_id, labels, report, frame_blob, flagged, clip_path)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, payload)
                    elif kind == "proctor_event":
                        cursor.execute("""
                            INSERT INTO proctor_events (
                                session_id, timestamp, camera_id, severity, event_type, candidate_id,
                                score, labels, clip_path, snapshot_path, details
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, payload)
                    elif kind == "anomaly_score":
                        cursor.execute("""
                            INSERT INTO proctor_anomaly_scores (
                                session_id, timestamp, candidate_id, movement_score, gaze_score, object_score, aggregate_score
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, payload)
                    elif kind == "intruder":
                        cursor.execute("""
                            INSERT INTO intruders (timestamp, camera_id, face_blob, confidence, id_card_match, notes)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, payload)
                    self.conn.commit()
            except Exception as e:
                print(f"[EVIDENCE] DB write error ({kind}): {e}")
            self.write_queue.task_done()

    def get_all_flagged(self):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, candidate_id, labels, report, clip_path, flagged
                FROM incidents
                WHERE flagged = 1
                ORDER BY timestamp DESC
            """)
            return [
                {"id": r[0], "timestamp": r[1], "candidate_id": r[2], "labels": r[3],
                 "report": r[4], "clip_path": r[5], "flagged": r[6]}
                for r in cursor.fetchall()
            ]

    def get_all_incidents(self, limit: int = 200):
        """Return all incidents (flagged and non-flagged) ordered by most recent."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, candidate_id, labels, report, clip_path, flagged
                FROM incidents
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [
                {"id": r[0], "timestamp": r[1], "candidate_id": r[2], "labels": r[3],
                 "report": r[4], "clip_path": r[5], "flagged": r[6]}
                for r in cursor.fetchall()
            ]

    def insert_video_record(self, filename: str, filepath_url: str):
        """Insert a bare video file as an unflagged incident record (used by /videos/scan)."""
        import re
        ts = time.time()
        # Try to parse timestamp from filename: clip_0_YYYYMMDD_HHMMSS.mp4
        m = re.search(r'(\d{8})_(\d{6})', filename)
        if m:
            import calendar
            from datetime import datetime
            try:
                dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
                ts = calendar.timegm(dt.timetuple())  # treat as local → UTC epoch
            except Exception:
                pass
        payload = (ts, 'Unknown', filename, 'Recorded clip', None, 0, filepath_url)
        with self.db_lock:
            cursor = self.conn.cursor()
            # Only insert if not already tracked
            cursor.execute("SELECT id FROM incidents WHERE clip_path = ?", (filepath_url,))
            if not cursor.fetchone():
                cursor.execute("""
                    INSERT INTO incidents (timestamp, candidate_id, labels, report, frame_blob, flagged, clip_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, payload)
                self.conn.commit()
                return True
            return False

    def reset_session(self):
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM proctor_events")
            cursor.execute("DELETE FROM proctor_anomaly_scores")
            cursor.execute("DELETE FROM incidents")
            self.conn.commit()
            print("[EVIDENCE] Session reset. Proctoring tables cleared.")

    def delete_incidents(self, ids):
        if not ids:
            return
        with self.db_lock:
            cursor = self.conn.cursor()
            placeholders = ",".join("?" * len(ids))
            cursor.execute(f"DELETE FROM incidents WHERE id IN ({placeholders})", ids)
            self.conn.commit()

    def auto_purge(self):
        with self.db_lock:
            self.conn.execute("DELETE FROM incidents WHERE flagged = 0")
            self.conn.commit()

    def purge_non_flagged(self, videos_dir: str = None):
        """Delete all non-flagged incidents from DB and remove associated video files.

        Runs in a background thread to complete within the 5-minute window (E-04).
        """
        def _do_purge():
            with self.db_lock:
                cursor = self.conn.cursor()
                # Collect video paths for non-flagged incidents before deletion
                cursor.execute("SELECT clip_path FROM incidents WHERE flagged = 0")
                paths = [r[0] for r in cursor.fetchall() if r[0]]
                cursor.execute("DELETE FROM incidents WHERE flagged = 0")
                self.conn.commit()

            # Remove video files outside the DB lock to avoid blocking reads
            removed = 0
            for path in paths:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                        removed += 1
                    elif path and videos_dir:
                        # Try resolving relative basename
                        fpath = os.path.join(videos_dir, os.path.basename(path))
                        if os.path.exists(fpath):
                            os.remove(fpath)
                            removed += 1
                except Exception as e:
                    print(f"[PURGE] Could not remove {path}: {e}")

            print(f"[PURGE] Non-flagged purge complete. {len(paths)} DB rows, {removed} video files removed.")

        import threading
        t = threading.Thread(target=_do_purge, daemon=True, name="PurgeJob")
        t.start()
        return t

    def close(self):
        self.running = False
        self.write_queue.join()
        for conn in self.conn_pool:
            conn.close()
        self.conn.close()
