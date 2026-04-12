# AI Surveillance System - React Frontend

A modern, feature-rich React application for monitoring exam halls using AI-powered phone detection via YOLOv8.

## Features

✅ **Real-time Feed Monitoring** - Live MJPEG stream from Python backend
✅ **Grid-based Detection** - 3x4 grid system for precise location mapping
✅ **Instant Alerts** - Phone detections with row, column, and confidence levels
✅ **Video Recording** - Auto-recording when phones are detected
✅ **Admin Panel** - Signup and management interface
✅ **User Authentication** - Role-based access control
✅ **Alert Notifications** - Persistent alert storage with video links
✅ **Records Archive** - Browse historical recordings with metadata
✅ **Responsive Design** - Works on desktop and tablet devices

## Tech Stack

- **Frontend**: React 18 + Vite
- **Styling**: Tailwind CSS
- **Routing**: React Router v6
- **State Management**: Zustand
- **HTTP Client**: Axios
- **UI Components**: Lucide React Icons

## Prerequisites

Before running this React app, you need:

- Node.js 16+ and npm
- Python Flask backend running (see **Running the Backend** section)
- The backend should be accessible at `http://localhost:5000`

## Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure the backend URL** (if different from localhost:5000):
   Edit `src/services/api.js` and update `API_BASE_URL`

## Running the App

### Development Mode

```bash
npm run dev
```

The app will start at `http://localhost:3000`

### Production Build

```bash
npm run build
npm run preview
```

## Backend Integration

### Python Flask Backend Setup

Your `second.py` Flask app should have CORS enabled to communicate with the React frontend. Add this to your Python code:

```python
from flask_cors import CORS

# Add this after creating your Flask app
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
```

Install flask-cors if you don't have it:
```bash
pip install flask-cors
```

### Running Both Services

1. **Terminal 1 - Start the Python backend**:
   ```bash
   python second.py
   ```
   The Flask API will be available at `http://localhost:5000`

2. **Terminal 2 - Start the React frontend**:
   ```bash
   npm run dev
   ```
   The React app will be available at `http://localhost:3000`

## Project Structure

```
surveillance-app/
├── src/
│   ├── components/              # Reusable React components
│   │   ├── Navbar.jsx          # Navigation bar with alerts
│   │   ├── FeedView.jsx        # Live feed with grid overlay
│   │   ├── AlertsTab.jsx       # Alert notifications
│   │   └── RecordsView.jsx     # Video archive
│   ├── pages/                   # Page components
│   │   ├── Home.jsx            # Landing page
│   │   ├── Login.jsx           # User login
│   │   ├── AdminPanel.jsx      # Admin signup/login
│   │   └── Dashboard.jsx       # Main dashboard
│   ├── context/                 # State management
│   │   ├── authStore.js        # Authentication store
│   │   └── alertStore.js       # Alerts store
│   ├── services/                # API layer
│   │   └── api.js              # Backend API client
│   ├── hooks/                   # Custom React hooks
│   │   └── useDetection.js     # Detection data polling
│   ├── App.jsx                 # Main app component
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── public/                      # Static assets
├── package.json                # Dependencies
├── vite.config.js              # Vite configuration
├── tailwind.config.js          # Tailwind configuration
└── index.html                  # HTML template
```

## API Endpoints

The React app expects these endpoints from the Flask backend (from your `second.py`):

- `GET /status` - Returns current detection state with grid positions
- `GET /stream` - MJPEG stream endpoint
- `GET /snapshot` - Single JPEG frame
- `GET /grid-info` - Grid configuration (rows × cols)

## Authentication

### User Login
- Default credentials: any username with any password
- Accessible via `/login` route

### Admin Panel
- Access via `/admin` route
- Create a new admin account via signup form
- Admin status grants access to full dashboard

## Key Features Explained

### Feed View
- Click on any feed to open fullscreen mode
- X button on top-right closes fullscreen
- Grid overlay shows cell positions (R1C1, R1C2, etc.)
- Real-time recording indicator
- Alert pulse animation when phone detected

### Alerts Tab
- Shows all detected phones with:
  - Grid position (Row, Column)
  - Confidence percentage
  - Timestamp
  - ExamHall number
- Click "Video" button to play associated recording
- Delete alerts individually

### Records View
- Filter by exam hall
- Thumbnail preview with play button
- Full metadata (position, timestamp, confidence)
- Video player modal with detailed information
- Sortable by date

## Customization

### Changing Grid Size
Edit the grid configuration in `src/services/api.js` or update frontend to match your Python backend's grid size.

### Modifying Colors
Update the theme colors in `tailwind.config.js`:
```js
theme: {
  extend: {
    colors: {
      primary: '#1e40af',     // Main blue
      secondary: '#0f172a',   // Dark background
      accent: '#f59e0b',      // Orange accent
    }
  }
}
```

### Polling Interval
Edit `useDetectionStatus()` in `src/hooks/useDetection.js`:
```js
// Change the 1000 (milliseconds) to desired interval
const fetchInterval = 1000; // 1 second
```

## Troubleshooting

### "Connection error" on feed
- Ensure Flask backend is running on `http://localhost:5000`
- Check CORS is enabled in Flask app
- Verify the camera is accessible in Python code

### Alerts not appearing
- Check that Flask `/status` endpoint returns JSON with `phone_detected` field
- Verify confidence threshold in Python code matches detections

### Feed not loading
- Try accessing `http://localhost:5000/stream` directly in browser
- Check that camera index and RTSP URL are correct in Python code

## Performance Tips

1. **Reduce polling frequency** if CPU usage is high
2. **Lower JPEG quality** in Python backend for faster streaming
3. **Use grid overlay sparingly** - disable in production if needed
4. **Cache grid info** to reduce API calls

## Security Notes

- Change admin credentials immediately after first login
- Use HTTPS in production (configure Vite proxy or use reverse proxy)
- Implement proper authentication (JWT tokens, OAuth, etc.)
- Add rate limiting to Flask backend
- Validate all inputs on both frontend and backend
- Store sensitive data securely (use environment variables)

## Future Enhancements

- [ ] Multi-camera support
- [ ] Advanced analytics and reporting
- [ ] Machine learning model updates
- [ ] Mobile app (React Native)
- [ ] Cloud storage integration
- [ ] Email/SMS notifications
- [ ] Database integration for persistent storage
- [ ] Advanced search and filtering

## Environment Variables

Create a `.env` file for production settings:

```env
VITE_API_URL=http://your-backend.com:5000
VITE_APP_NAME=AI Surveillance System
```

Access in code: `import.meta.env.VITE_API_URL`

## Support & Documentation

For Flask backend issues, see `second.py` comments.
For React issues, check this README or review `src/services/api.js` integration.

## License

All rights reserved. Unauthorized access prohibited.

---

**Version**: 1.0.0
**Last Updated**: 2024
