import os
import sqlite3

from config import DB_PATH, get_config


PROCTOR_TABLES = (
    """
    CREATE TABLE IF NOT EXISTS proctor_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at REAL NOT NULL,
        ended_at REAL,
        tier TEXT,
        hardware_json TEXT
    )
    """,
    """
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
    """,
    """
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
    """,
)


def migrate_database(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    for statement in PROCTOR_TABLES:
        cursor.execute(statement)
    conn.commit()
    conn.close()
    return db_path


def wire_modules():
    config = get_config()
    migrate_database(config.DB_PATH)
    return {
        "tier": config.profile.tier,
        "workers": config.NUM_DETECTION_WORKERS,
        "detect_every_n": config.DETECT_EVERY_N,
        "detection_resolution": config.DETECTION_RESOLUTION,
        "clip_resolution": config.CLIP_RESOLUTION,
        "db_path": config.DB_PATH,
    }


if __name__ == "__main__":
    info = wire_modules()
    print("[INTEGRATE] Proctoring database and adaptive config ready:")
    for key, value in info.items():
        print(f"  {key}: {value}")
