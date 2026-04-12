# Quick Start Guide - AI Surveillance System

Follow these steps to get the complete AI Surveillance System up and running.

## Step 1: Prepare the Python Backend

### Install Dependencies

```bash
pip install flask flask-cors opencv-python ultralytics torch torchvision
```

Or use the existing requirements if you have them.

### Update your Python backend

You have two options:

**Option A: Use the Updated Backend (Recommended)**
```bash
# Copy the CORS-enabled version to replace your original
copy second_CORS_enabled.py second.py
```

**Option B: Modify Your Existing Backend**
1. Add `from flask_cors import CORS` to imports
2. Add CORS configuration after `app = Flask(__name__)`:
   ```python
   CORS(app, resources={
       r"/*": {
           "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
           "methods": ["GET", "POST", "OPTIONS"],
           "allow_headers": ["Content-Type"],
           "supports_credentials": True
       }
   })
   ```

See `BACKEND_SETUP.md` for detailed instructions.

## Step 2: Start the Python Backend

Open a terminal in the surveillance-app folder and run:

```bash
python second.py
```

You should see output like:
```
[CAM] 1280x720  grid=3x4
[API] Running at http://0.0.0.0:5000
      /status     → JSON detection state
      /stream     → MJPEG live feed
      /snapshot   → single JPEG
      /grid-info  → grid dimensions

[CORS] Enabled for React frontend at http://localhost:3000
```

The backend is now running at `http://localhost:5000`

## Step 3: Start the React Frontend

Open a new terminal in the surveillance-app folder and run:

```bash
npm install
npm run dev
```

You should see output like:
```
  VITE v5.0.0  ready in 234 ms

  ➜  Local:   http://localhost:3000/
```

## Step 4: Access the Application

Open your browser and go to: **http://localhost:3000**

### Home Page
- View system overview and features
- Click "Get Started" to login

### User Login
- Go to `/login`
- Enter any username and password
- Click "Login"

### Admin Panel
- Go to `/admin`
- Create a new admin account via signup
- Or login with existing credentials

### Dashboard
- **Feeds Tab**: Watch live exam hall feeds with grid overlay
- **Alerts Tab**: View phone detections with video links
- **Records Tab**: Browse historical recordings

## Troubleshooting

### Feed Not Loading

**Error**: "Stream connection error"

**Solutions**:
1. Verify Flask backend is running
2. Check `http://localhost:5000/stream` loads in browser
3. Ensure camera is accessible
4. Check CORS is enabled in Flask

### Alerts Not Appearing

**Error**: Detections not showing in alerts tab

**Solutions**:
1. Verify phone is actually in frame
2. Check confidence threshold (default 0.45)
3. Ensure `phone_detected` is true in `/status` endpoint
4. Try lowering the confidence threshold

### Can't Connect to Backend

**Error**: Network error or timeout

**Solutions**:
1. Check Flask is running on port 5000
2. Try accessing directly: `http://localhost:5000/grid-info`
3. Check for firewall blocking
4. Verify no port conflicts

### npm install Fails

**Solutions**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -r node_modules package-lock.json
npm install
```

## Project Structure

```
surveillance-app/
├── src/
│   ├── components/          # React components
│   ├── pages/              # Page components
│   ├── context/            # State management (Zustand)
│   ├── services/           # API client
│   ├── hooks/              # Custom hooks
│   └── App.jsx             # Main app
├── second_CORS_enabled.py   # Updated backend with CORS
├── README.md               # Full documentation
├── BACKEND_SETUP.md        # Backend configuration guide
├── QUICK_START.md          # This file
└── package.json            # NPM dependencies
```

## Key Features

✅ **Real-time Feeds** - Live MJPEG stream from each exam hall
✅ **Grid-based Detection** - 3x4 grid for precise location
✅ **Instant Alerts** - Phone detections with row, column, confidence
✅ **Auto Recording** - Videos saved automatically for 5+ seconds
✅ **Video Archive** - Browse all recordings with metadata
✅ **Focus Mode** - Click feed to fullscreen, X button to close
✅ **Admin Controls** - Separate admin panel with signup
✅ **Responsive Design** - Works on desktop and tablets

## Configuration

### Change Grid Size
Edit Python backend:
```python
GRID_ROWS = 3  # Change to your preferred rows
GRID_COLS = 4  # Change to your preferred columns
```

### Change Confidence Threshold
Edit Python backend:
```python
CONFIDENCE_MIN = 0.45  # Lower = more detections, higher = fewer false positives
```

### Change Recording Duration
Edit Python backend:
```python
POST_DETECT_RECORD_SECS = 5  # Seconds after phone leaves frame
```

### Change API Port
Edit Python backend:
```python
FLASK_PORT = 5000  # Change if port is in use
```

Then update React `src/services/api.js`:
```javascript
const API_BASE_URL = 'http://localhost:5000'  // Update port here
```

## Development Tips

### View Frontend Errors
Open browser DevTools: `F12` or `Ctrl+Shift+I`

Look in the **Console** tab for JavaScript errors

### View Backend Logs
Check the terminal running Flask for API errors

### Hot Reload
The React app auto-reloads when you save files - no page refresh needed!

### Debug Feeds
Add to `FeedView.jsx` to see status data:
```javascript
console.log('Detection Status:', status)
```

## Performance Tips

1. **Reduce update frequency**: Edit `useDetectionStatus()` in `src/hooks/useDetection.js`
```javascript
const fetchStatus = useDetectionStatus(examHall, 2000) // 2 second poll
```

2. **Lower JPEG quality** in Python backend:
```python
cv2.IMWRITE_JPEG_QUALITY, 50  # Lower value = smaller file, faster
```

3. **Disable grid overlay** if not needed in `FeedView.jsx`

## Next Steps

1. **Connect Multiple Cameras**: Modify Python backend to loop through cameras
2. **Database Integration**: Store alerts in a database instead of memory
3. **Email Notifications**: Add email alerts for detections
4. **Deploy to Cloud**: Use AWS, Google Cloud, or Heroku
5. **Mobile App**: Create React Native app for mobile access

## Support

- **Backend Issues**: Check `BACKEND_SETUP.md`
- **Frontend Issues**: Check `README.md` 
- **Configuration**: Review Python config section in `second.py`

---

**Ready? Start with Step 1 above! 🚀**
