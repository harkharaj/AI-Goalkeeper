import cv2
import time
import json
import threading
import numpy as np
from flask import Flask, Response, render_template_string, request
from picamera2 import Picamera2
from ultralytics import YOLO

# ============================================================
# CONFIGURATION — set these to match your real goal dimensions
# ============================================================
# Real-world X positions of your goal posts in whatever unit you want
# (e.g. cm, inches — just be consistent with your motor controller)
# Left post = 0, Right post = GOAL_WIDTH
GOAL_WIDTH_REAL  = 91       # e.g. 100cm wide goal
INFER_SIZE       = 160      # must match goalkeeper_ai.py
FRAME_W          = 320
FRAME_H          = 240

# ============================================================
# GLOBALS
# ============================================================
app          = Flask(__name__)
frame_lock   = threading.Lock()
output_frame = None

# Latest ONNX detection (pixel coords, bottom-left origin — same as goalkeeper_ai)
latest_cx    = None         # pixel X  (0–320)
latest_cy    = None         # pixel Y  (0–240, bottom-left origin)
latest_size  = None

# 4 calibration points collected by user
# Each entry: {"pixel_x": int, "real_x": float} — Y ignored for 1D motor
calib_points = []           # will hold up to 4 points

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Goalkeeper Calibration</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0a0a0a;
            color: #39ff14;
            font-family: 'Share Tech Mono', monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
            gap: 16px;
        }
        h1 { color: #fff; letter-spacing: 0.2em; font-size: 1rem; border-bottom: 1px solid #39ff14; padding-bottom: 8px; width: 900px; }
        .layout { display: flex; gap: 24px; align-items: flex-start; }
        .feed img { width: 640px; height: 480px; image-rendering: pixelated; border: 1px solid #333; display: block; }
        .feed p  { color: #555; font-size: 0.7rem; margin-top: 4px; }
        .panel {
            background: #111;
            border: 1px solid #222;
            border-radius: 6px;
            padding: 20px;
            width: 240px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .panel h2 { color: #fff; font-size: 0.85rem; border-bottom: 1px solid #333; padding-bottom: 6px; }
        .detection {
            background: #000;
            border: 1px solid #1a1a1a;
            padding: 10px;
            font-size: 0.8rem;
            border-radius: 4px;
        }
        .detection span { color: #fff; }
        label { font-size: 0.75rem; color: #888; }
        input[type=number] {
            width: 100%;
            background: #000;
            border: 1px solid #333;
            color: #39ff14;
            padding: 6px 8px;
            font-family: 'Share Tech Mono', monospace;
            font-size: 0.85rem;
            border-radius: 3px;
            margin-top: 4px;
        }
        button {
            width: 100%;
            padding: 9px;
            border: none;
            border-radius: 4px;
            font-family: 'Share Tech Mono', monospace;
            font-weight: bold;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .btn-capture { background: #007bff; color: #fff; }
        .btn-capture:hover { background: #0056b3; }
        .btn-clear   { background: #333; color: #aaa; }
        .btn-save    { background: #28a745; color: #fff; font-size: 0.9rem; padding: 12px; }
        .btn-save:hover { background: #1e7e34; }
        .points-list {
            background: #000;
            border: 1px solid #1a1a1a;
            padding: 10px;
            font-size: 0.72rem;
            border-radius: 4px;
            min-height: 80px;
            color: #aaa;
        }
        .points-list .pt { color: #39ff14; margin: 2px 0; }
        .msg { font-size: 0.75rem; color: #ff4444; min-height: 20px; }
        .msg.ok { color: #39ff14; }
        hr { border-color: #222; }
    </style>
</head>
<body>
    <h1>⬡ GOALKEEPER CALIBRATION — ONNX Detection</h1>

    <div class="layout">
        <div class="feed">
            <img src="/video_feed" alt="Loading...">
            <p>ORIGIN: BOTTOM-LEFT (0,0) &nbsp;|&nbsp; Same coordinate system as goalkeeper_ai.py</p>
        </div>

        <div class="panel">
            <h2>LIVE DETECTION</h2>
            <div class="detection">
                Pixel X: <span id="px">—</span><br>
                Pixel Y: <span id="py">—</span><br>
                Size:    <span id="sz">—</span>
            </div>

            <hr>
            <h2>CAPTURE POINT</h2>
            <p style="font-size:0.72rem; color:#666;">
                Place ball at a known real-world X position,
                wait for detection, enter the real X, then capture.
                Collect 2–4 points across the goal width.
            </p>
            <label>Real-world X at ball position:</label>
            <input type="number" id="real_x" placeholder="e.g. 50" step="0.1">
            <button class="btn-capture" onclick="capture()">CAPTURE POINT</button>
            <button class="btn-clear"   onclick="clearPts()">CLEAR ALL POINTS</button>

            <hr>
            <h2>CAPTURED POINTS</h2>
            <div class="points-list" id="pts_list">No points yet.</div>
            <div class="msg" id="msg"></div>

            <button class="btn-save" onclick="save()">SAVE CALIBRATION</button>
        </div>
    </div>

    <script>
        // Poll detection every 200ms
        setInterval(() => {
            fetch('/detection').then(r => r.json()).then(d => {
                document.getElementById('px').innerText = d.cx  !== null ? d.cx  : '—';
                document.getElementById('py').innerText = d.cy  !== null ? d.cy  : '—';
                document.getElementById('sz').innerText = d.size !== null ? d.size.toFixed(1) : '—';
            });
            fetch('/points').then(r => r.json()).then(pts => {
                const el = document.getElementById('pts_list');
                if (pts.length === 0) { el.innerHTML = 'No points yet.'; return; }
                el.innerHTML = pts.map((p, i) =>
                    `<div class="pt">[${i+1}] px=${p.pixel_x}  real=${p.real_x}</div>`
                ).join('');
            });
        }, 300);

        function capture() {
            const rx = parseFloat(document.getElementById('real_x').value);
            if (isNaN(rx)) { setMsg('Enter a real-world X value first.', false); return; }
            fetch('/capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({real_x: rx})
            }).then(r => r.json()).then(d => {
                setMsg(d.msg, d.ok);
            });
        }

        function clearPts() {
            fetch('/clear').then(() => setMsg('Points cleared.', true));
        }

        function save() {
            fetch('/save').then(r => r.json()).then(d => {
                setMsg(d.msg, d.ok);
            });
        }

        function setMsg(txt, ok) {
            const el = document.getElementById('msg');
            el.innerText = txt;
            el.className = ok ? 'msg ok' : 'msg';
        }
    </script>
</body>
</html>
"""

# ============================================================
# CAMERA + ONNX THREAD
# ============================================================
class CameraStream:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (FRAME_W, FRAME_H)},
            controls={"FrameDurationLimits": (16666, 16666)}
        )
        self.picam2.configure(config)
        self.picam2.set_controls({"ExposureTime": 8000})
        self.picam2.start()
        self.latest = None
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while True:
            self.latest = self.picam2.capture_array()

    def read(self):
        return self.latest


def detection_loop(cam):
    global output_frame, latest_cx, latest_cy, latest_size

    print("Loading ONNX model...")
    model = YOLO("best.onnx", task="detect")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results   = model.predict(source=frame_bgr, imgsz=INFER_SIZE, conf=0.5, verbose=False)

        detected = False
        for r in results:
            if len(r.boxes) == 0:
                continue

            best  = max(r.boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, best.xyxy[0])
            conf  = float(best.conf[0])

            ball_size  = ((x2 - x1) + (y2 - y1)) / 2
            cx         = int((x1 + x2) / 2)
            cy_opencv  = int((y1 + y2) / 2)
            cy         = FRAME_H - cy_opencv      # flip to bottom-left origin — matches goalkeeper_ai

            latest_cx   = cx
            latest_cy   = cy
            latest_size = ball_size

            # Draw box + crosshair
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(frame_bgr, (cx, cy_opencv), 4, (0, 255, 0), -1)

            # Pixel X label
            cv2.putText(frame_bgr, f"px={cx}  real_y={cy}  sz={ball_size:.1f}  c={conf:.2f}",
                        (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Vertical line at cx
            cv2.line(frame_bgr, (cx, 0), (cx, FRAME_H), (0, 200, 100), 1)

            detected = True
            break

        if not detected:
            latest_cx = latest_cy = latest_size = None

        # Draw captured calibration points as vertical lines
        for i, pt in enumerate(calib_points):
            px = pt['pixel_x']
            cv2.line(frame_bgr, (px, 0), (px, FRAME_H), (0, 80, 255), 1)
            cv2.putText(frame_bgr, f"P{i+1} real={pt['real_x']}",
                        (px + 3, 15 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 80, 255), 1)

        # Origin marker — bottom-left
        cv2.circle(frame_bgr, (0, FRAME_H - 1), 4, (255, 255, 0), -1)
        cv2.putText(frame_bgr, "(0,0)", (5, FRAME_H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        with frame_lock:
            output_frame = frame_bgr.copy()


# ============================================================
# FLASK ROUTES
# ============================================================
def gen_feed():
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            _, buf = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            data = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(gen_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection')
def detection():
    return {'cx': latest_cx, 'cy': latest_cy, 'size': latest_size}


@app.route('/points')
def points():
    return calib_points


@app.route('/capture', methods=['POST'])
def capture():
    data   = request.get_json()
    real_x = float(data['real_x'])

    if latest_cx is None:
        return {'ok': False, 'msg': 'No ball detected — place ball in view first.'}

    calib_points.append({'pixel_x': latest_cx, 'real_x': real_x})
    return {'ok': True, 'msg': f'Captured px={latest_cx} → real={real_x}'}


@app.route('/clear')
def clear():
    calib_points.clear()
    return {'ok': True}


@app.route('/save')
def save():
    if len(calib_points) < 2:
        return {'ok': False, 'msg': 'Need at least 2 points to calibrate.'}

    pixel_xs = np.array([p['pixel_x'] for p in calib_points], dtype=float)
    real_xs  = np.array([p['real_x']  for p in calib_points], dtype=float)

    # Fit a linear mapping: real_x = scale * pixel_x + offset
    # This is a 1D calibration — maps pixel X → real-world X
    coeffs = np.polyfit(pixel_xs, real_xs, 1)   # [scale, offset]
    scale  = float(coeffs[0])
    offset = float(coeffs[1])

    # Verify quality — compute residuals
    predicted = np.polyval(coeffs, pixel_xs)
    residuals = real_xs - predicted
    max_err   = float(np.max(np.abs(residuals)))

    # Save as JSON — easy to load in goalkeeper_ai
    calib = {
        'scale':       scale,
        'offset':      offset,
        'max_error':   max_err,
        'points_used': calib_points,
        'frame_width': FRAME_W,
        'note':        'real_x = scale * pixel_x + offset  (pixel_x from goalkeeper_ai, bottom-left origin)'
    }

    with open('calibration.json', 'w') as f:
        json.dump(calib, f, indent=2)

    print(f"\nCalibration saved:")
    print(f"  scale  = {scale:.4f}")
    print(f"  offset = {offset:.4f}")
    print(f"  max residual error = {max_err:.2f} real-world units")
    print(f"\nUsage in goalkeeper_ai.py:")
    print(f"  real_x = {scale:.4f} * last_pred_x + {offset:.4f}")

    return {
        'ok':  True,
        'msg': f'Saved! scale={scale:.4f} offset={offset:.4f} err={max_err:.2f}'
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Warming up camera...")
    cam = CameraStream()
    time.sleep(1.0)

    print("Starting ONNX detection thread...")
    t = threading.Thread(target=detection_loop, args=(cam,), daemon=True)
    t.start()

    print("\nCalibration dashboard → http://<PI_IP>:5000")
    print("\nHow to calibrate:")
    print("  1. Place ball at left post  → enter real X → Capture")
    print("  2. Place ball at right post → enter real X → Capture")
    print("  3. Optionally add 2 more points in between for accuracy")
    print("  4. Click SAVE CALIBRATION")
    print("  5. In goalkeeper_ai.py:  real_x = scale * last_pred_x + offset")

    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
