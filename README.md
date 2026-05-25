# AI Surveillance System

An advanced, edge-deployed AI surveillance and automated proctoring system. It continuously monitors a camera feed to detect anomalies such as unauthorized cell phones, employing a multi-threaded Python backend and a real-time React dashboard.

## How to Run

### Method 1: The Simple Method (Recommended)
You can launch the entire system (both the AI backend and the React dashboard) with a single click.

1. Double-click the **`Start_All.bat`** file in this folder.
2. Two command prompt windows will open (one for the backend and one for the frontend).
3. The AI models will take approximately 10-15 seconds to load.
4. Your default web browser will automatically open to the dashboard at **http://localhost:3000**.

### Method 2: Manual Startup
If the `Start_All.bat` script does not work, you can start the components manually:

**Step 1: Start the AI Backend**
1. Open a terminal or command prompt in this directory.
2. Run the backend startup script:
   ```bash
   Run_Backend.bat
   ```
   *(This will activate the virtual environment and start the Python server on port 5000).*

**Step 2: Start the React Dashboard**
1. Open a **second** terminal or command prompt window.
2. Navigate to the frontend directory:
   ```bash
   cd surveillance-app
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open your browser and go to `http://localhost:3000` to view the surveillance dashboard.

## Documentation
Additional technical documentation, architectural details, and setup guides can be found in the `docs/` folder and `surveillance-app/README.md`.
