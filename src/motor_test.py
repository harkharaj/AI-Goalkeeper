import RPi.GPIO as GPIO
import time

# ─── Pin Definitions ──────────────────────────────────────────────────────────
STEP   = 18
DIR    = 17
ENABLE = 25

# ═════════════════════════════════════════════════════════════════════════════
#  CALIBRATION  —  measured limits from centre (0)
#  Right = negative, Left = positive
# ═════════════════════════════════════════════════════════════════════════════

STEPS_RIGHT_LIMIT =  1400       # steps from centre to right physical limit
DIST_RIGHT_LIMIT_CM = 45.72     # 1.5 ft = 45.72 cm

STEPS_LEFT_LIMIT  =  1650       # steps from centre to left physical limit
DIST_LEFT_LIMIT_CM  = 48.77     # 1.6 ft = 48.77 cm

# Derived: steps per cm for each side (they may differ slightly)
STEPS_PER_CM_RIGHT = STEPS_RIGHT_LIMIT / DIST_RIGHT_LIMIT_CM   # ~30.6 steps/cm
STEPS_PER_CM_LEFT  = STEPS_LEFT_LIMIT  / DIST_LEFT_LIMIT_CM    # ~33.8 steps/cm

# ═════════════════════════════════════════════════════════════════════════════
#  COMMAND  —  set the target position from centre in cm
#  Positive = LEFT, Negative = RIGHT
# ═════════════════════════════════════════════════════════════════════════════

TARGET_CM = -20.0    # ← change this   e.g. -20 = 20 cm right, +15 = 15 cm left

# ═════════════════════════════════════════════════════════════════════════════
#  SPEED RAMP  —  trapezoidal profile
#
#   DELAY_START  : initial (slow) pulse delay  — bigger = slower start
#   DELAY_MIN    : peak speed pulse delay       — smaller = faster cruise
#   RAMP_STEPS   : how many steps to ramp up (and same count to ramp down)
#
#  Timeline:  [ramp up over RAMP_STEPS] → [cruise at DELAY_MIN] → [ramp down]
#
#  Tips:
#   - If motor jerks at start  → increase DELAY_START
#   - If motor stalls at start → increase RAMP_STEPS or increase DELAY_START
#   - To go faster overall     → decrease DELAY_MIN (min ~0.0002)
#   - Smoother ramp            → increase RAMP_STEPS
# ═════════════════════════════════════════════════════════════════════════════

DELAY_START_US = 2000   # slow start delay  in microseconds  (bigger = slower start)
DELAY_MIN_US   =  200   # peak speed delay  in microseconds  (smaller = faster cruise)
RAMP_STEPS     =  200   # steps to ramp up / ramp down

# Convert to seconds for time.sleep (do not edit)
DELAY_START = DELAY_START_US / 1_000_000
DELAY_MIN   = DELAY_MIN_US   / 1_000_000

# ─── Setup ────────────────────────────────────────────────────────────────────
LEFT  = False   # CW
RIGHT = True    # CCW  (flip if your motor wiring is opposite)

GPIO.setmode(GPIO.BCM)
GPIO.setup([STEP, DIR, ENABLE], GPIO.OUT)
GPIO.output(ENABLE, GPIO.LOW)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def cm_to_steps(cm: float) -> tuple[int, bool]:
    """
    Convert a signed cm distance to (steps, direction).
    Negative cm → RIGHT, Positive cm → LEFT.
    Uses the correct steps/cm ratio for each side.
    """
    if cm == 0:
        return 0, RIGHT
    if cm < 0:                              # moving right
        steps = round(abs(cm) * STEPS_PER_CM_RIGHT)
        steps = min(steps, STEPS_RIGHT_LIMIT)
        return steps, RIGHT
    else:                                   # moving left
        steps = round(cm * STEPS_PER_CM_LEFT)
        steps = min(steps, STEPS_LEFT_LIMIT)
        return steps, LEFT


def ramp_delay(step_index: int, total_steps: int) -> float:
    """
    Return the pulse delay for a given step using a trapezoidal ramp.
    Linearly interpolates between DELAY_START and DELAY_MIN.
    """
    ramp = min(RAMP_STEPS, total_steps // 2)   # can't ramp more than half

    if step_index < ramp:
        # Ramp up: delay decreases from DELAY_START → DELAY_MIN
        t = step_index / ramp
        return DELAY_START + (DELAY_MIN - DELAY_START) * t

    elif step_index >= total_steps - ramp:
        # Ramp down: delay increases from DELAY_MIN → DELAY_START
        t = (total_steps - step_index) / ramp
        return DELAY_START + (DELAY_MIN - DELAY_START) * t

    else:
        # Cruise
        return DELAY_MIN


def move(steps: int, direction: bool):
    """Move `steps` steps in `direction` with trapezoidal speed ramp."""
    GPIO.output(DIR, GPIO.HIGH if direction else GPIO.LOW)
    label = "RIGHT (−)" if direction == RIGHT else "LEFT (+)"
    print(f"Moving {steps} steps → {label}")
    print(f"Ramp: {RAMP_STEPS} steps up/down | cruise: {DELAY_MIN_US} µs | start: {DELAY_START_US} µs")

    for i in range(steps):
        delay = ramp_delay(i, steps)
        GPIO.output(STEP, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        time.sleep(delay)
        print(f"  step {i+1:4d}/{steps}  |  delay {delay*1_000_000:.0f} µs", end="\r")

    print(f"  Done — {steps} steps completed.          ")


# ─── Main ─────────────────────────────────────────────────────────────────────
try:
    steps, direction = cm_to_steps(TARGET_CM)
    side = "RIGHT (−)" if TARGET_CM < 0 else "LEFT (+)"

    print(f"\nTarget : {TARGET_CM:+.1f} cm  →  {side}")
    print(f"Steps  : {steps}  (ratio used: "
          f"{STEPS_PER_CM_RIGHT:.1f} steps/cm right, "
          f"{STEPS_PER_CM_LEFT:.1f} steps/cm left)")

    if steps == 0:
        print("Already at centre — no movement.")
    else:
        move(steps, direction)

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    GPIO.output(ENABLE, GPIO.HIGH)
    GPIO.cleanup()
    print("GPIO cleaned up.")
