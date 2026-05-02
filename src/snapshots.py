import cv2
import numpy as np
import time
import os
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

app = Flask(__name__)

# Directory to save images on the Pi
SAVE_FOLDER = "dataset_frames"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# State variables
app_state = "IDLE"
timer_start = 0
capture_frame_count = 0 # NEW: Counter for frames

# Initialize Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)}) 
picam2.configure(config)
picam2.set_controls({"ExposureTime": 10000}) 
picam2.start()

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
    </style>
</head>
<body>
    <h2>Goalkeeper AI: Dataset Collector</h2>
    <button onclick="fetch('/trigger')">START CAPTURE (3..2..1)</button>
    <br>
    <img src="/video_feed">
</body>
</html>
"""

def generate_frames():
    global app_state, timer_start, capture_frame_count
    
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1) # Mirror feed
        
        # Convert to BGR for proper colors
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Create copies: one to display text, one to save cleanly
        display_frame = bgr_frame.copy()
        clean_frame = bgr_frame.copy()

        if app_state == "COUNTDOWN":
            elapsed = time.time() - timer_start
            
            if elapsed < 1.0:
                text = "3"
            elif elapsed < 2.0:
                text = "2"
            elif elapsed < 3.0:
                text = "1"
            else:
                app_state = "CAPTURING"
                timer_start = time.time()
                capture_frame_count = 0 # Reset frame counter when capture starts
                text = "GO!"

            # Draw Countdown
            cv2.putText(display_frame, text, (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)

        elif app_state == "CAPTURING":
            elapsed = time.time() - timer_start
            
            if elapsed <= 2.0: # 2-second capture window
                capture_frame_count += 1
                
                # NEW: Only save if it is the 5th frame
                if capture_frame_count % 10 == 0:
                    filename = os.path.join(SAVE_FOLDER, f"snap_{int(time.time()*1000)}.jpg")
                    cv2.imwrite(filename, clean_frame)
                
                # Show recording indicator (shows on all frames so it doesn't flicker)
                cv2.putText(display_frame, "RECORDING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.circle(display_frame, (300, 20), 10, (0, 0, 255), -1)
            else:
                app_state = "IDLE"

        # Encode for web
        _, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/trigger')
def trigger():
    global app_state, timer_start
    if app_state == "IDLE":
        app_state = "COUNTDOWN"
        timer_start = time.time()
    return "OK"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
