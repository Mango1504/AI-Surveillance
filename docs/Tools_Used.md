# Tools Used

The system leverages a modern, full-stack technology suite, heavily emphasizing performance and machine learning integrations.

## Backend & AI

- **Python 3.8+**: Core backend runtime.
- **Ultralytics YOLOv8**: State-of-the-art, real-time object detection model.
- **DeepFace**: A lightweight facial recognition and attribute analysis framework for candidate verification.
- **Ollama (LLaVA/LLaMA3-Vision)**: Local Vision-Language Models for generating human-readable incident summaries.
- **OpenCV (`cv2`)**: Industry-standard library for image processing, frame manipulation, and video encoding/decoding.
- **Flask & Flask-SocketIO**: Web framework for serving REST APIs and real-time WebSocket communication.
- **SQLite 3**: Serverless database engine utilized for logging incidents and session metadata.
- **Cryptography**: Used for encrypting sensitive biometric and evidence data.

## Frontend

- **React 18**: UI library for building the interactive dashboard.
- **Vite**: Next-generation frontend tooling for extremely fast builds and development server.
- **Tailwind CSS**: Utility-first CSS framework used for designing the dark-themed proctor interface.
- **Zustand**: Minimalist, fast state-management library for React.
- **Axios**: Promise-based HTTP client for the browser.
- **Lucide React**: Beautiful and consistent icon set.

## Deployment & Hardware

- **CUDA / TensorRT / OpenVINO**: Hardware acceleration backends utilized by the AI models to maximize inference speed depending on available GPUs (NVIDIA, Apple Silicon, etc.).
