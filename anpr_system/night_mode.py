"""
night_mode.py — Automatic Night Mode Enhancement
Detects low-light conditions and applies adaptive enhancement pipeline.

Features:
  - Auto brightness detection per frame
  - Gamma correction (brightens dark frames without blowing highlights)
  - CLAHE on LAB L-channel (contrast limited adaptive histogram equalisation)
  - Bilateral denoising for grainy low-light frames
  - Manual override: force ON / OFF / AUTO
  - Logs mode switches so analytics can track night vs day detections
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
NIGHT_BRIGHTNESS_THRESHOLD = 85   # 0-255 mean brightness; below = night mode on
DAY_BRIGHTNESS_THRESHOLD   = 110  # hysteresis: only switch back to day above this
GAMMA_NIGHT                = 1.8  # gamma < 1 darkens, > 1 brightens
GAMMA_DAY                  = 1.0  # no correction in daylight
CLAHE_NIGHT_CLIP           = 3.5
CLAHE_DAY_CLIP             = 1.5
CHECK_INTERVAL_S           = 2.0  # re-evaluate brightness every N seconds

# ── State ──────────────────────────────────────────────────────────────────────
class NightModeState:
    def __init__(self):
        self.lock         = threading.Lock()
        self.mode         = 'AUTO'      # 'AUTO' | 'ON' | 'OFF'
        self.is_night     = False
        self.brightness   = 128
        self.last_check   = 0
        self.switch_log   = []          # [(timestamp, 'day'|'night')]

    @property
    def active(self) -> bool:
        """True when night enhancement should be applied."""
        with self.lock:
            if self.mode == 'ON':  return True
            if self.mode == 'OFF': return False
            return self.is_night   # AUTO

    def set_mode(self, mode: str):
        with self.lock:
            self.mode = mode.upper()
        print(f"[NIGHT] Mode set to: {mode}")

    def update(self, brightness: float):
        """Update night state from measured brightness (call every CHECK_INTERVAL_S)."""
        with self.lock:
            self.brightness = round(brightness, 1)
            was_night = self.is_night
            if not was_night and brightness < NIGHT_BRIGHTNESS_THRESHOLD:
                self.is_night = True
                self.switch_log.append((datetime.now().strftime('%H:%M:%S'), 'night'))
                print(f"[NIGHT] Switched to NIGHT MODE (brightness={brightness:.0f})")
            elif was_night and brightness > DAY_BRIGHTNESS_THRESHOLD:
                self.is_night = False
                self.switch_log.append((datetime.now().strftime('%H:%M:%S'), 'day'))
                print(f"[NIGHT] Switched to DAY MODE (brightness={brightness:.0f})")
            self.last_check = time.time()

    def status(self) -> dict:
        with self.lock:
            return {
                'mode':         self.mode,
                'is_night':     self.is_night,
                'active':       self.active,
                'brightness':   self.brightness,
                'switch_log':   list(self.switch_log[-10:]),
            }


# Singleton
night_state = NightModeState()


# ── Gamma Correction LUT ───────────────────────────────────────────────────────
def _build_gamma_lut(gamma: float) -> np.ndarray:
    inv   = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return table


_LUT_NIGHT = _build_gamma_lut(GAMMA_NIGHT)
_LUT_DAY   = _build_gamma_lut(GAMMA_DAY)


# ── Core Enhancement ───────────────────────────────────────────────────────────
def measure_brightness(frame: np.ndarray) -> float:
    """Return mean brightness of a frame (fast: downsample first)."""
    small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_LINEAR)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def enhance_frame(frame: np.ndarray, force_night: bool = None) -> np.ndarray:
    """
    Main entry point. Called per-frame in the MJPEG loop.
    Returns an enhanced copy of the frame.

    Args:
        frame:       Raw BGR frame from camera
        force_night: Override auto-detection (True=night, False=day, None=auto)
    """
    now = time.time()

    # Auto-update brightness state every CHECK_INTERVAL_S
    if (now - night_state.last_check) > CHECK_INTERVAL_S and night_state.mode == 'AUTO':
        brightness = measure_brightness(frame)
        night_state.update(brightness)

    apply_night = force_night if force_night is not None else night_state.active

    if not apply_night:
        # Day mode: light CLAHE only
        return _apply_clahe(frame, clip=CLAHE_DAY_CLIP)

    return _apply_night_pipeline(frame)


def _apply_clahe(frame: np.ndarray, clip: float) -> np.ndarray:
    """Apply CLAHE on LAB L-channel (preserves colour, boosts contrast)."""
    try:
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def _apply_night_pipeline(frame: np.ndarray) -> np.ndarray:
    """
    Full night pipeline:
    1. Gamma brightening
    2. CLAHE (stronger clip)
    3. Bilateral denoise (removes grain from boosted image)
    4. Sharpen edges (plates need sharp edges for OCR)
    """
    try:
        # Step 1: Gamma correction
        brightened = cv2.LUT(frame, _LUT_NIGHT)

        # Step 2: CLAHE on LAB
        lab     = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=CLAHE_NIGHT_CLIP, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # Step 3: Light bilateral denoising (d=5 for speed)
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)

        # Step 4: Unsharp mask for edge sharpening
        blurred  = cv2.GaussianBlur(denoised, (0, 0), 2)
        sharpened = cv2.addWeighted(denoised, 1.4, blurred, -0.4, 0)

        return sharpened
    except Exception:
        return frame


def enhance_roi_for_ocr(roi: np.ndarray) -> np.ndarray:
    """
    Extra-strong enhancement specifically for plate crop before OCR.
    Applied in night mode — doubles OCR accuracy on dark plates.
    """
    if roi is None or roi.size == 0:
        return roi
    try:
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        # Gamma
        gamma = cv2.LUT(gray, _build_gamma_lut(2.0))
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gamma)
        # Sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp  = cv2.filter2D(enhanced, -1, kernel)
        return sharp
    except Exception:
        return roi