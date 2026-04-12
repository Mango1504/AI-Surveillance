# Complete Setup Guide

## Project Overview

This is a complete React web application that integrates with your YOLOv8 phone detection Python backend. It provides real-time monitoring, alert notifications, video recording, and archival features for exam hall surveillance.

## System Requirements

- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

## Installation Steps

### 1. Install Node.js Dependencies

In the `surveillance-app` directory:

```bash
npm install
```

This installs all required packages:
- `react` & `react-dom` - React framework
- `react-router-dom` - Client-side routing
- `axios` - HTTP requests to backend
- `zustand` - State management
- `lucide-react` - UI icons
- `vite` - Build tool
- `tailwindcss` - CSS styling

### 2. Install Python Dependencies

```bash
pip install flask flask-cors opencv-python ultralytics torch torchvision
```

Or if you have a requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Prepare Backend

You have two options:

**Option A: Use Updated Backend (Recommended)**
```
Copy: second_CORS_enabled.py to your project folder
Use: python second_CORS_enabled.py
```

**Option B: Update Your Backend**
Add to your `second.py`:
```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})
```

## Running the Application

### Terminal 1: Start Backend

```bash
python second.py
```

Expected output:
```
[CAM] 1280x720  grid=3x4
[API] Running at http://0.0.0.0:5000
      /status     → JSON detection state
      /stream     → MJPEG live feed
      /snapshot   → single JPEG
      /grid-info  → grid dimensions
[CORS] Enabled for React frontend at http://localhost:3000
```

### Terminal 2: Start Frontend

```bash
npm run dev
```

Expected output:
```
  VITE v5.0.0  ready in 234 ms

  ➜  Local:   http://localhost:3000/
  ➜  press h + enter to show help
```

### Access Application

Open browser: **http://localhost:3000**

## Application Navigation

### 1. Home Page (/)
- System overview and features
- Information about YOLOv8 detection
- Feature showcase
- Links to login and admin panel

### 2. User Login (/login)
- Username: any value
- Password: any value (demo mode)
- Creates session with user privileges
- Access to Feeds, Alerts, Records

### 3. Admin Panel (/admin)
- Signup form to create admin accounts
- Admin login
- Full system access including monitoring
- Additional admin features

### 4. Dashboard (/dashboard)

**Feeds Tab:**
- Select exam hall (1-4)
- Real-time MJPEG stream
- Grid overlay showing cell positions
- Click to focus, X button to unfocus
- Detection annotations with bounding boxes
- Recording indicator (red dot)
- Live alert pulse animation
- Detection details cards

**Alerts Tab:**
- All phone detections with timestamps
- Row and column location
- Confidence percentage
- ExamHall number
- Play associated video recording
- Delete individual alerts
- Video player modal

**Records Tab:**
- Video archive browser
- Filter by exam hall
- Thumbnail previews
- Detailed metadata (Row, Col, Time, Date, Type)
- Play recording in modal
- Full video player with controls

## Features Explained

### Real-time Feed Monitoring
- MJPEG stream from `http://localhost:5000/stream`
- Grid overlay dividing frame into cells
- Grid cell labels (R1C1, R1C2, etc.)
- Updates every 40ms (~25 FPS)
- Click any feed to expand to fullscreen
- Close fullscreen with X button (top-right corner)

### Detection Alerts
- Automatically triggers when phone detected
- Shows grid position (Row, Column)
- Displays confidence percentage
- Stores timestamp
- Associates exam hall number
- Links to video recording
- Pulsing red alert indicator on feed
- Recording auto-starts for 5 seconds

### Grid System
- Divides video frame into 3×4 cells
- Cell-based location identification
- Configurable grid size (edit Python backend)
- Overlay shows R{row}C{col} labels
- Precise phone location for investigation

### Video Recording
- Auto-starts when phone detected
- Continues for 5 seconds after disappears
- Saved as AVI files with timestamp
- Stored in Python `SAVE_PATH` directory
- Associated with each alert
- Viewable in Records tab

### Admin Controls
- Separate login/signup interface
- Admin badge visible when logged in
- Full access to all features
- Future: user management, settings

### Responsive Design
- Mobile: stacked layout
- Tablet: 2-column grid
- Desktop: 3-column grid
- Navbar collapses on mobile
- Touch-friendly buttons and controls

## API Endpoints

All endpoints are on `http://localhost:5000`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/status` | GET | Current detection state (JSON) |
| `/stream` | GET | MJPEG live video feed |
| `/snapshot` | GET | Single JPEG frame |
| `/grid-info` | GET | Grid configuration |

### Status Response Example
```json
{
  "phone_detected": true,
  "timestamp": "2024-01-15T10:30:45Z",
  "recording": true,
  "frame_width": 1280,
  "frame_height": 720,
  "grid_rows": 3,
  "grid_cols": 4,
  "detections": [
    {
      "bbox": [100, 50, 200, 150],
      "center": [150, 100],
      "confidence": 0.95,
      "grid_row": 1,
      "grid_col": 2,
      "label": "Row 1, Col 2"
    }
  ]
}
```

## Configuration

### Backend Configuration (second.py)

```python
CAMERA_INDEX = 0              # 0 = webcam, use IP URL for IP camera
GRID_ROWS = 3                 # Grid rows (change to your preference)
GRID_COLS = 4                 # Grid columns
PHONE_CLASS_ID = 67           # YOLOv8 COCO class ID for phones
CONFIDENCE_MIN = 0.45         # Detection confidence threshold (0-1)
FLASK_HOST = "0.0.0.0"        # Bind to all interfaces
FLASK_PORT = 5000             # Flask server port
POST_DETECT_RECORD_SECS = 5   # Record duration after phone leaves
SAVE_PATH = r"C:/Users/write/Desktop/Phone"  # Video save location
```

### Frontend Configuration (vite.config.js)

```javascript
server: {
  port: 3000,                          // Frontend port
  proxy: {
    '/api': {
      target: 'http://localhost:5000', // Backend URL
      changeOrigin: true
    }
  }
}
```

### Frontend Configuration (src/services/api.js)

```javascript
const API_BASE_URL = 'http://localhost:5000'
```

## Customization Guide

### Change Grid Size
Edit `second.py`:
```python
GRID_ROWS = 4  # Change number of rows
GRID_COLS = 5  # Change number of columns
```

### Change Detection Confidence
Edit `second.py`:
```python
CONFIDENCE_MIN = 0.50  # Increase = fewer false positives
```

### Change Recording Duration
Edit `second.py`:
```python
POST_DETECT_RECORD_SECS = 10  # Change to 10 seconds
```

### Change Theme Colors
Edit `tailwind.config.js`:
```javascript
colors: {
  primary: '#1e40af',    // Blue
  secondary: '#0f172a',  // Dark
  accent: '#f59e0b',     // Orange
}
```

### Change API Polling Rate
Edit `src/hooks/useDetection.js`:
```javascript
const pollInterval = 2000  // Poll every 2 seconds (default 1 second)
```

## Troubleshooting

### Issue: "Cannot connect to backend"
**Causes**: Flask not running, wrong port, CORS not enabled
**Solution**: 
1. Check Flask running on port 5000
2. Verify CORS settings in backend
3. Test: `curl http://localhost:5000/status`

### Issue: "Feed not loading"
**Causes**: Camera not accessible, MJPEG endpoint down
**Solution**:
1. Check camera in Python backend
2. Test directly: `http://localhost:5000/stream`
3. Verify frame dimensions in backend output

### Issue: "Alerts not appearing"
**Causes**: No phone in frame, confidence too high
**Solution**:
1. Actually put a phone in front of camera
2. Lower `CONFIDENCE_MIN` in backend
3. Check `/status` endpoint returns `"phone_detected": true`

### Issue: "npm install fails"
**Solution**:
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Issue: "Port 3000/5000 already in use"
**Solution**:
```bash
# Find process using port
lsof -i :3000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or change ports in config
```

## Performance Optimization

1. **Reduce update frequency** (lower CPU usage):
   - Edit `useDetectionStatus()` polling interval

2. **Lower JPEG quality** (faster streaming):
   - Edit Python: `cv2.IMWRITE_JPEG_QUALITY, 60`

3. **Disable grid overlay** (if not needed):
   - Comment out grid rendering in `FeedView.jsx`

4. **Lazy load records** (faster page load):
   - Implement pagination in `RecordsView.jsx`

## Production Deployment

### For Small Deployment
```bash
npm run build
# Serves static files from dist/ folder
```

### For Cloud Deployment (AWS, Heroku, etc.)
1. Set `VITE_API_URL` environment variable
2. Update Python backend with production domain
3. Enable HTTPS
4. Add authentication (JWT, OAuth)
5. Use database instead of in-memory storage

## Security Notes

1. **Change default credentials** after first login
2. **Use strong passwords** for admin accounts
3. **Enable HTTPS** in production
4. **Restrict network access** to trusted IPs
5. **Add rate limiting** to backend endpoints
6. **Validate all inputs** on frontend and backend
7. **Use environment variables** for sensitive data

## File Structure

```
surveillance-app/
├── src/
│   ├── components/
│   │   ├── Navbar.jsx       # Navigation and user menu
│   │   ├── FeedView.jsx     # Live feeds with grid
│   │   ├── AlertsTab.jsx    # Alert notifications
│   │   └── RecordsView.jsx  # Video archive
│   ├── pages/
│   │   ├── Home.jsx         # Landing page
│   │   ├── Login.jsx        # User authentication
│   │   ├── AdminPanel.jsx   # Admin auth
│   │   └── Dashboard.jsx    # Main view
│   ├── context/
│   │   ├── authStore.js     # User state
│   │   └── alertStore.js    # Alerts state
│   ├── services/
│   │   └── api.js           # Backend client
│   ├── hooks/
│   │   └── useDetection.js  # Detection logic
│   ├── App.jsx              # Routing
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── index.html              # HTML template
├── package.json            # Dependencies
├── vite.config.js          # Build config
├── tailwind.config.js      # CSS config
├── README.md               # Full documentation
└── QUICK_START.md          # This guide
```

## Support & Help

**Python Backend Issues**: Check `BACKEND_SETUP.md`
**Frontend Issues**: Check `README.md`
**Quick Start**: See `QUICK_START.md`
**Verification**: Run `verify.sh` (Linux/Mac)

## Next Steps

1. ✅ Run both services
2. ✅ Access http://localhost:3000
3. ✅ Test with demo login
4. ✅ Place phone in front of camera
5. ✅ Check alerts appear
6. ✅ Play recorded video
7. ✅ Customize configuration
8. ✅ Deploy to production

---

**Happy monitoring! 🎥🔒**
