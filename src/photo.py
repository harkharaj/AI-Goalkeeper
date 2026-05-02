import cv2
import numpy as np
import time
import os
import threading
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

app = Flask(__name__)

# Directory to save images on the Pi
SAVE_FOLDER = "dataset_frames"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# State variables
take_photo_flag = False
latest_frame = None  # Holds the most recent camera frame
lock = threading.Lock()

# Initialize Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})  
picam2.configure(config)
picam2.set_controls({"ExposureTime": 10000})  
picam2.start()

# --- NEW: BACKGROUND CAMERA THREAD ---
def camera_thread():
    global latest_frame
    while True:
        # The camera hardware runs independently here
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1) # Mirror feed
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Safely update the global frame variable
        with lock:
            latest_frame = bgr_frame

# Start the camera thread immediately
threading.Thread(target=camera_thread, daemon=True).start()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Collector</title>
    <style>
        body { font-family: Arial; display: flex; flex-direction: column; align-items: center; background: #222; color: white;}
        img { border: 2px solid #fff; margin-top: 10px; width: 640px; }
        button { margin-top: 20px; padding: 15px 30px; font-size: 20px; background: #28a745; color: white; border: none; cursor: pointer; border-radius: 5px; }
        button:hover { background: #218838; }
        button:active { background: #1e7e34; transform: scale(0.98); }
    </style>
</head>
<body>
    <h2>Goalkeeper AI: Dataset Collector</h2>
    <button onclick="fetch('/trigger')">SNAP PHOTO</button>
    <br>
    <img src="/video_feed">
</body>
</html>
"""

def generate_frames():
    global take_photo_flag, latest_frame
    
    while True:
        # 1. Safely grab the newest frame from the background thread
        with lock:
            if latest_frame is None:
                continue
            current_frame = latest_frame.copy() # Only copy once!
            
        display_frame = current_frame
        
        # 2. Check if we need to save it
        if take_photo_flag:
            filename = os.path.join(SAVE_FOLDER, f"snap_{int(time.time()*1000)}.jpg")
            
            # Save the raw current_frame before drawing text on it
            cv2.imwrite(filename, current_frame)
            take_photo_flag = False
            
            # Draw on the display frame to confirm for the user
            cv2.putText(display_frame, "PHOTO SAVED!", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # 3. Encode and send to web
        _, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # 4. CRITICAL: Give the CPU and network a tiny rest (~30 FPS limit)
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/trigger')
def trigger():
    global take_photo_flag
    take_photo_flag = True
    return "OK"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
