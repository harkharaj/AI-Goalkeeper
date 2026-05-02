import cv2
import time
import gc
import json
import threading
import queue
import numpy as np
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from ultralytics import YOLO
from collections import deque
import RPi.GPIO as GPIO

# ============================================================
# 1. USER SETTINGS — YOLO / TRACKING
# ============================================================
TRIGGER_SIZE = 18 # Ball size (px) TTI target 
TIME_TO_REACT = 0.1 # Seconds before impact to actuate 
INFER_SIZE = 160 # YOLO inference resolution 
MAX_MISSES = 3 # Frames without detection before reset 
HISTORY = 10 # Max history deque length 
HOME_TIMEOUT_S = 3.0 # Seconds after last detection before returning home

# --- Flight gate thresholds ---
MIN_SIZE_GROWTH  = 0.15     # px/frame growth to classify as thrown
ROLL_MIN_X_DISP  = 2        # px X must move to classify as rolling

# ============================================================
# 2. MOTOR SETTINGS
# ============================================================
STEP   = 18
DIR    = 17
ENABLE = 25

# Measured limits from centre (0 cm)
# Positive real_x = LEFT, Negative real_x = RIGHT  (matches calib sign convention)
STEPS_RIGHT_LIMIT    = 1400
DIST_RIGHT_LIMIT_CM  = 45.72

STEPS_LEFT_LIMIT     = 1650
DIST_LEFT_LIMIT_CM   = 48.77

STEPS_PER_CM_RIGHT   = STEPS_RIGHT_LIMIT / DIST_RIGHT_LIMIT_CM
STEPS_PER_CM_LEFT    = STEPS_LEFT_LIMIT  / DIST_LEFT_LIMIT_CM

# Speed ramp
DELAY_START_US = 3000
DELAY_MIN_US   =  300
RAMP_STEPS     =  150
DELAY_START    = DELAY_START_US / 1_000_000
DELAY_MIN      = DELAY_MIN_US   / 1_000_000

LEFT  = False   # CW
RIGHT = True    # CCW

# ============================================================
# 3. FLASK SETUP & GLOBALS
# ============================================================
app          = Flask(__name__)
output_frame = None
frame_lock   = threading.Lock()

# Motor command queue — tracking thread puts target_cm, motor thread consumes
# Sending None tells motor thread to shut down
motor_queue  = queue.Queue(maxsize=1)   # maxsize=1 — only keep latest command

# Tracks where the motor physically is right now (in cm from home)
current_pos_cm    = 0.0
current_pos_lock  = threading.Lock()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Goalkeeper AI</title>
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
            min-height: 100vh;
            padding: 24px;
            gap: 16px;
        }
        h1 {
            font-size: 1.1rem;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            color: #fff;
            border-bottom: 1px solid #39ff14;
            padding-bottom: 8px;
            width: 640px;
        }
        .frame-wrap {
            position: relative;
            width: 640px;
            height: 480px;
            border: 1px solid #222;
        }
        .frame-wrap img {
            width: 640px;
            height: 480px;
            display: block;
            image-rendering: pixelated;
        }
        .corner {
            position: absolute;
            width: 12px; height: 12px;
            border-color: #39ff14;
            border-style: solid;
        }
        .corner.tl { top:0; left:0;  border-width: 2px 0 0 2px; }
        .corner.tr { top:0; right:0; border-width: 2px 2px 0 0; }
        .corner.bl { bottom:0; left:0;  border-width: 0 0 2px 2px; }
        .corner.br { bottom:0; right:0; border-width: 0 2px 2px 0; }
        .status {
            width: 640px;
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: #555;
        }
        .status span { color: #39ff14; }
    </style>
</head>
<body>
    <h1>&#x2B21; Goalkeeper AI &mdash; Live Feed</h1>
    <div class="frame-wrap">
        <div class="corner tl"></div>
        <div class="corner tr"></div>
        <div class="corner bl"></div>
        <div class="corner br"></div>
        <img src="/video_feed" alt="Feed loading...">
    </div>
    <div class="status">
        <span>ORIGIN: BOTTOM-LEFT (0,0)</span>
        <span>TRIGGER SIZE: 16px</span>
        <span>INFER: 160px</span>
    </div>
</body>
</html>
"""

# ============================================================
# 4. MOTOR HELPERS
# ============================================================
def cm_to_steps(cm: float) -> tuple:
    """Convert signed cm to (steps, direction). Negative=RIGHT, Positive=LEFT."""
    if cm == 0:
        return 0, RIGHT
    if cm < 0:
        steps = round(abs(cm) * STEPS_PER_CM_RIGHT)
        steps = min(steps, STEPS_RIGHT_LIMIT)
        return steps, RIGHT
    else:
        steps = round(cm * STEPS_PER_CM_LEFT)
        steps = min(steps, STEPS_LEFT_LIMIT)
        return steps, LEFT


def ramp_delay(step_index: int, total_steps: int) -> float:
    """Trapezoidal ramp delay for smooth acceleration."""
    ramp = min(RAMP_STEPS, total_steps // 2)
    if step_index < ramp:
        t = step_index / ramp
        return DELAY_START + (DELAY_MIN - DELAY_START) * t
    elif step_index >= total_steps - ramp:
        t = (total_steps - step_index) / ramp
        return DELAY_START + (DELAY_MIN - DELAY_START) * t
    else:
        return DELAY_MIN


def _raw_move(steps: int, direction: bool):
    """Low-level: move exact steps in direction with ramp. Blocking."""
    GPIO.output(DIR, GPIO.HIGH if direction else GPIO.LOW)
    for i in range(steps):
        delay = ramp_delay(i, steps)
        GPIO.output(STEP, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        time.sleep(delay)


def move_to_cm(target_cm: float):
    """
    Move from current_pos_cm to target_cm.
    Updates current_pos_cm after move.
    Clamps to physical limits.
    """
    global current_pos_cm

    with current_pos_lock:
        delta_cm = target_cm - current_pos_cm

    if abs(delta_cm) < 0.5:   # ignore tiny moves < 0.5 cm
        return

    steps, direction = cm_to_steps(delta_cm)
    if steps == 0:
        return

    label = "RIGHT" if direction == RIGHT else "LEFT"
    print(f"[MOTOR] {current_pos_cm:+.1f} cm -> {target_cm:+.1f} cm  ({delta_cm:+.1f} cm, {steps} steps {label})")

    _raw_move(steps, direction)

    with current_pos_lock:
        current_pos_cm = target_cm


# ============================================================
# 5. MOTOR THREAD — consumes from queue, never blocks tracking
# ============================================================
def motor_thread_fn():
    """
    Dedicated thread for all motor movement.
    Tracking loop just drops a target_cm into the queue.
    If a new command arrives while moving, it's ignored until
    current move finishes (queue size = 1, old item replaced).
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([STEP, DIR, ENABLE], GPIO.OUT)
    GPIO.output(ENABLE, GPIO.LOW)
    print("[MOTOR] Thread started, GPIO ready.")

    try:
        while True:
            target_cm = motor_queue.get()    # blocks until command arrives
            if target_cm is None:            # shutdown signal
                break
            move_to_cm(target_cm)
    finally:
        GPIO.output(ENABLE, GPIO.HIGH)
        GPIO.cleanup()
        print("[MOTOR] GPIO cleaned up.")


def send_motor_command(target_cm: float):
    """
    Non-blocking: drop target_cm into queue.
    If queue is full (motor busy), discard old command and put new one.
    """
    try:
        motor_queue.get_nowait()   # drain old pending command if any
    except queue.Empty:
        pass
    try:
        motor_queue.put_nowait(target_cm)
    except queue.Full:
        pass


# ============================================================
# 6. FAST CAMERA STREAM
# ============================================================
class FastCameraStream:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (320, 240)},
            controls={"FrameDurationLimits": (16666, 16666)}
        )
        self.picam2.configure(config)
        self.picam2.set_controls({"ExposureTime": 8000})
        self.picam2.start()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.latest_frame = self.picam2.capture_array()

    def read(self):
        return self.latest_frame


# ============================================================
# 7. FLIGHT GATE
# ============================================================
def is_ball_in_flight(history):
    if len(history) < 3:
        return False, None

    sizes = [p[2] for p in history]
    xs    = [p[0] for p in history]

    n          = len(sizes)
    frames     = np.arange(n, dtype=float)
    size_slope = np.polyfit(frames, sizes, 1)[0]

    if size_slope > MIN_SIZE_GROWTH:
        return True, 'thrown'

    x_disp = abs(xs[-1] - xs[0])
    if x_disp > ROLL_MIN_X_DISP or size_slope > 0.05:
        return True, 'rolling'

    return False, None


# ============================================================
# 8. PREDICT X AT now + react_time
# ============================================================
def predict_x_at_future(history, react_time):
    if len(history) < 3:
        return None

    xs = np.array([p[0] for p in history], dtype=float)
    ts = np.array([p[3] for p in history], dtype=float)
    ts = ts - ts[0]

    coeffs   = np.polyfit(ts, xs, 1)
    t_future = ts[-1] + react_time
    pred_x   = int(np.clip(np.polyval(coeffs, t_future), 0, 320))
    return pred_x


# ============================================================
# 9. PREDICT TIME-TO-IMPACT
# ============================================================
def predict_time_to_impact(history, target_size):
    if len(history) < 3:
        return None

    sizes = np.array([p[2] for p in history], dtype=float)
    ts    = np.array([p[3] for p in history], dtype=float)
    ts    = ts - ts[0]

    coeffs = np.polyfit(ts, sizes, 1)
    slope  = coeffs[0]

    if slope <= 0.1:
        return None

    t_impact       = (target_size - coeffs[1]) / slope
    time_to_impact = t_impact - ts[-1]

    if time_to_impact <= 0 or time_to_impact > 3.0:
        return None

    return time_to_impact


# ============================================================
# 10. CORE YOLO TRACKER
# ============================================================
def tracking_loop(cam_stream, scale, offset):
    global output_frame

    print("Loading ONNX model...")
    model = YOLO("best.onnx", task="detect")

    history          = deque(maxlen=HISTORY)
    triggered        = False
    missed_frames    = 0
    frame_count      = 0
    last_pred_x      = None
    last_pred_x_time = None
    in_flight        = False
    flight_mode      = None
    in_flight_frames = 0

    last_detection_time = time.time()   # tracks when ball was last seen
    home_triggered      = False         # so we only send home command once

    while True:
        frame = cam_stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results  = model.predict(source=frame_bgr, imgsz=INFER_SIZE, conf=0.5, verbose=False)
        detected = False

        for r in results:
            if len(r.boxes) == 0:
                continue

            best_box        = max(r.boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            conf            = float(best_box.conf[0])

            ball_size = ((x2 - x1) + (y2 - y1)) / 2
            cx        = int((x1 + x2) / 2)
            cy_opencv = int((y1 + y2) / 2)
            cy        = 240 - cy_opencv

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame_bgr, f"Sz:{ball_size:.1f} c:{conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            history.append((cx, cy, ball_size, time.time()))
            last_detection_time = time.time()
            home_triggered      = False        # ball is back — reset home flag

            in_flight, flight_mode = is_ball_in_flight(history)

            if in_flight and not triggered:
                in_flight_frames += 1

                should_predict = (in_flight_frames == 3) or \
                                 (in_flight_frames > 3 and (in_flight_frames - 3) % 2 == 0)

                if should_predict:
                    pred_x = predict_x_at_future(history, TIME_TO_REACT)
                    if pred_x is not None:
                        last_pred_x      = pred_x
                        last_pred_x_time = time.time()

                tti = predict_time_to_impact(history, TRIGGER_SIZE)

                if tti is not None and tti <= TIME_TO_REACT and last_pred_x is not None:
                    triggered = True
                    real_x    = scale * last_pred_x + offset
                    print(f"\n>>> ACTUATE  pixel_x={last_pred_x}  real_x={real_x:.1f} cm  TTI={tti:.3f}s")
                    send_motor_command(real_x)

            elif not in_flight:
                cv2.putText(frame_bgr, "HELD / STATIC",
                            (4, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)

            detected = True
            missed_frames = 0
            break

        # ---- Return to home after HOME_TIMEOUT_S seconds of no detection ----
        if not detected:
            missed_frames += 1
            if missed_frames >= MAX_MISSES:
                history.clear()
                triggered        = False
                in_flight        = False
                flight_mode      = None
                in_flight_frames = 0

            time_since_last = time.time() - last_detection_time
            if time_since_last >= HOME_TIMEOUT_S and not home_triggered:
                home_triggered = True
                print(f"\n[HOME] No ball for {HOME_TIMEOUT_S:.0f}s — returning to home (0.0 cm)")
                send_motor_command(0.0)

        # Draw predicted X line — lingers 10 seconds
        if last_pred_x is not None and last_pred_x_time is not None:
            if time.time() - last_pred_x_time < 10.0:
                for yy in range(0, 240, 8):
                    cv2.line(frame_bgr,
                             (last_pred_x, yy),
                             (last_pred_x, min(yy + 4, 240)),
                             (0, 80, 255), 1)
                age   = time.time() - last_pred_x_time
                label = f"X={last_pred_x}  +{age:.1f}s"
                cv2.putText(frame_bgr, label,
                            (last_pred_x + 4, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
            else:
                last_pred_x      = None
                last_pred_x_time = None

        # Origin marker
        cv2.circle(frame_bgr, (0, 239), 4, (255, 255, 0), -1)
        cv2.putText(frame_bgr, "(0,0)", (5, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        # Motor position overlay
        with current_pos_lock:
            pos_txt = f"MOTOR: {current_pos_cm:+.1f} cm"
        cv2.putText(frame_bgr, pos_txt,
                    (200, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Status bar
        if triggered:
            state_txt = "TRIGGERED"
            state_col = (0, 255, 255)
        elif in_flight:
            state_txt = f"IN-FLIGHT [{flight_mode.upper()}]"
            state_col = (0, 255, 100)
        else:
            state_txt = "WATCHING"
            state_col = (180, 180, 180)
        cv2.putText(frame_bgr, state_txt,
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_col, 1)

        with frame_lock:
            output_frame = frame_bgr.copy()

        if frame_count % 300 == 0:
            gc.collect()


# ============================================================
# 11. FLASK ROUTES
# ============================================================
def generate_web_feed():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_web_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================
# 12. MAIN
# ============================================================
if __name__ == '__main__':

    # Load calibration once — zero overhead during tracking
    calib  = json.load(open('calibration.json'))
    scale  = calib['scale']
    offset = calib['offset']
    print(f"Calibration loaded: scale={scale:.4f}  offset={offset:.4f}")

    # Start motor thread first — GPIO setup happens inside
    mt = threading.Thread(target=motor_thread_fn, daemon=True)
    mt.start()
    time.sleep(0.5)   # give GPIO time to init

    print("Warming up camera...")
    cam = FastCameraStream()
    time.sleep(1.0)

    print("Starting tracker thread...")
    tt = threading.Thread(target=tracking_loop, args=(cam, scale, offset), daemon=True)
    tt.start()

    print("\nDashboard -> http://<PI_IP>:5000")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        motor_queue.put(None)   # clean shutdown signal to motor thread
        mt.join(timeout=3)
        print("Shutdown complete.")
