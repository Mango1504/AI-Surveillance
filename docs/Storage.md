# Storage and Data Management

The system implements a robust, privacy-first, edge-local storage mechanism. No data leaves the physical device without explicit administrator action.

## 1. Database (SQLite)
All metadata is stored in `evidence.db`, utilizing SQLite in Write-Ahead Log (WAL) mode to support highly concurrent reads/writes from the multi-threaded python backend.
- **Tables**:
  - `incidents`: Stores timestamp, location, candidate ID, VLM text reports, and file paths to associated video evidence.
  - `proctor_sessions`: Tracks start/end times and hardware configuration of an exam session.
  - `proctor_events` & `proctor_anomaly_scores`: Stores granular, rate-limited telemetry for auditing the Risk Engine's decision-making process.

## 2. Video Storage (`/videos`)
When an anomaly is confirmed, the `DiskWriter` module asynchronously flushes buffered frames to disk.
- **Format**: `.mp4` (or `.avi` fallback) encoded using `mp4v` or `MJPG` depending on the CPU core count.
- **Lifecycle**:
  - Clips are named contextually: `clip_0_YYYYMMDD_HHMMSS.mp4`.
  - The `EvidenceCache` includes an auto-purge utility. When a session ends, any video clips that were **not** flagged as confirmed breaches by the Risk Engine are securely deleted from the disk to minimize storage footprint and ensure privacy.

## 3. Ephemeral Biometrics
- Facial embeddings generated from applicant photos are stored purely in RAM within the `IdentityManager`. 
- They are never written to disk, and the `/purge-biometrics` endpoint hard-wipes them between sessions.
