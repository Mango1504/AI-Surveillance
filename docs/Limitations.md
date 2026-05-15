# Limitations

While the AI Surveillance System is highly capable, it currently operates with a few known limitations due to its edge-computing nature and reliance on computer vision.

1. **Hardware Dependencies**: Real-time multi-threaded AI inference (YOLOv8 + DeepFace) requires significant computational resources. Without a dedicated GPU (CUDA, MPS, or OpenCL), the system aggressively throttles frame rates to prevent CPU bottlenecking, which can delay alert generation.
2. **Environmental Constraints**: 
   - **Lighting**: Poor or highly variable lighting conditions significantly degrade facial recognition accuracy and object detection confidence.
   - **Occlusion**: Objects (like phones) hidden under desks, hands, or clothing cannot be detected. Faces partially covered by masks or hands may fail identity verification.
3. **Field of View**: A single camera has limited coverage. Large exam halls require multiple cameras, which currently scale linearly in hardware cost and processing power requirements.
4. **False Positives**: Despite the Risk Engine, objects visually similar to cell phones (like certain calculators or dark rectangular objects) may occasionally trigger false alerts.
5. **VLM Latency**: Generating human-readable reports via Ollama/LLaVA can introduce latency (3-10 seconds depending on hardware). While asynchronous, this means the text report arrives after the initial alert.
6. **Browser Codecs**: Native HTML5 `<video>` tags do not universally support OpenCV's optimized `mp4v` codec for live streaming. As a result, the system relies on MJPEG proxy streaming, which consumes more network bandwidth than compressed H.264 streams.
