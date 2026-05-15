import requests
import cv2
import base64

class VLMReporter:
    """Generates natural language incident reports via Ollama LLaVA/LLaMA3-Vision."""
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="llava"):
        self.ollama_url = ollama_url
        self.model = model
        
    def generate_report(self, frame, detection_labels, camera_id=0, grid_row=None, grid_col=None, candidate_id="Unknown"):
        """Send frame + contextual metadata to Ollama to get a human-readable incident report."""
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Build location context string
        location = f"Camera {camera_id}"
        if grid_row is not None and grid_col is not None:
            location += f", Exam Hall Row {grid_row} Col {grid_col}"
        
        prompt = (
            f"{location}. Candidate ID: {candidate_id}. "
            f"Detected: {', '.join(detection_labels)}. "
            f"Describe the suspicious examination violation visible in this image in one clear sentence "
            f"for an invigilator. Include the candidate position and the specific violation observed."
        )
        
        try:
            response = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            }, timeout=10)
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"VLM Error: {response.status_code}"
        except Exception as e:
            return f"VLM unreachable: Please ensure Ollama is running and LLaVA model is pulled."
