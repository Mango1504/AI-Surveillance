# Future Scope

The AI Surveillance System lays the groundwork for a robust automated proctoring environment. Several enhancements are planned for future iterations:

1. **Multi-Camera Stitching**: Automatically combine feeds from multiple overlapping cameras to create a unified top-down view of the exam hall, tracking individuals as they move between camera zones.
2. **Advanced Behavioral Models**: Moving beyond simple gaze tracking to implement full skeleton/pose estimation (e.g., using MediaPipe) to detect subtle cheating behaviors like whispering, passing notes, or looking at hidden materials.
3. **Audio Analysis**: Integration of audio anomaly detection to flag speaking, whispering, or unusual noises during an exam session.
4. **Cloud Synchronization**: While currently designed as an edge-local system for privacy, an optional cloud sync module could push finalized, encrypted incident reports and clips to an AWS/Azure central repository for institutional auditing.
5. **Hardware Edge Acceleration**: Deploying tailored models using TensorRT specifically for NVIDIA Jetson Nano/Orin devices to create standalone, plug-and-play smart camera nodes.
6. **Automated Reporting via Email/SMS**: Real-time push notifications sent to administrators or head proctors when a high-severity "BREACH" alert is confirmed.
