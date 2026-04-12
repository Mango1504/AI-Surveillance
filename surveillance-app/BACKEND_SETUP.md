# Python Backend - CORS Configuration

To enable the React frontend to communicate with your Flask backend, add CORS (Cross-Origin Resource Sharing) support to your `second.py`.

## Required Modification to second.py

Add these lines at the top of `second.py` after the imports:

```python
from flask_cors import CORS
```

Then, after creating your Flask app (around line where `app = Flask(__name__)` is), add:

```python
app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})
```

## Installation

Install flask-cors:

```bash
pip install flask-cors
```

## Complete Modified Section

Here's what the top of your `second.py` should look like:

```python
"""
Phone Detection System — detector.py
Detects phones via YOLOv8, maps to grid cells, serves data via Flask API.
Android app connects to the Flask endpoints defined at the bottom.
"""

import cv2
import time
import os
import threading
import json
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS  # <-- ADD THIS

# ... rest of your code ...

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# ... rest of your Flask routes ...
```

## Testing the Connection

1. Start your Flask backend:
   ```bash
   python second.py
   ```

2. Start the React app:
   ```bash
   npm run dev
   ```

3. Open `http://localhost:3000` in your browser

4. Go to Dashboard → Feeds tab

5. The live stream should load if everything is configured correctly

## Notes

- The `origins` array can include production URLs when you deploy
- Change `http://localhost:3000` to match your actual frontend URL
- For production, use HTTPS and add your production domain
