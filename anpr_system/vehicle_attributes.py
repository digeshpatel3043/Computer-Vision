"""
vehicle_attributes.py — Vehicle Color & Type Detection
Detects dominant vehicle color and estimates vehicle class from frame context.

Color detection: HSV-based dominant color from vehicle body region (above plate).
Type estimation: Bounding box aspect ratio + relative size heuristic.

No extra model needed — runs on the cropped vehicle region from YOLO bbox.
"""

import cv2
import numpy as np
from typing import Optional


# ── Color Map (HSV ranges) ─────────────────────────────────────────────────────
# Each entry: (name, hex, hue_low, hue_high, sat_low, val_low)
COLOR_RANGES = [
    ('White',   '#FFFFFF', 0,   180, 0,   200),
    ('Black',   '#1a1a1a', 0,   180, 0,   50),
    ('Silver',  '#C0C0C0', 0,   180, 0,   130),
    ('Red',     '#E53935', 0,   10,  80,  80),
    ('Red',     '#E53935', 160, 180, 80,  80),
    ('Orange',  '#FF6D00', 10,  25,  80,  80),
    ('Yellow',  '#FFD600', 25,  35,  80,  80),
    ('Green',   '#43A047', 35,  85,  40,  40),
    ('Blue',    '#1E88E5', 85,  130, 40,  40),
    ('Purple',  '#8E24AA', 130, 160, 40,  40),
    ('Brown',   '#6D4C41', 10,  20,  40,  30),
    ('Grey',    '#9E9E9E', 0,   180, 0,   100),
]


def detect_vehicle_color(frame: np.ndarray, plate_bbox: dict) -> dict:
    """
    Detect dominant vehicle color using the region ABOVE the plate bbox.
    The body of the vehicle is typically in the upper portion of the frame.

    Args:
        frame:       Full BGR frame
        plate_bbox:  {'x', 'y', 'w', 'h'} of detected plate

    Returns:
        {'color': 'Blue', 'hex': '#1E88E5', 'confidence': 72}
    """
    if frame is None or plate_bbox is None:
        return {'color': 'Unknown', 'hex': '#888888', 'confidence': 0}

    h_f, w_f = frame.shape[:2]
    px, py, pw, ph = plate_bbox['x'], plate_bbox['y'], plate_bbox['w'], plate_bbox['h']

    # Sample the region above the plate (vehicle body)
    sample_h = min(int(ph * 4), py)          # up to 4× plate height above plate
    sample_w = int(pw * 2)                   # twice plate width
    sample_x = max(0, px - pw // 2)
    sample_y = max(0, py - sample_h)
    sample_x2 = min(w_f, sample_x + sample_w)

    if sample_h < 10 or (sample_x2 - sample_x) < 10:
        # Fallback: just use the whole frame top half
        roi = frame[:h_f//2, :]
    else:
        roi = frame[sample_y:py, sample_x:sample_x2]

    if roi.size == 0:
        return {'color': 'Unknown', 'hex': '#888888', 'confidence': 0}

    # Remove very dark (shadow) and very bright (reflection) pixels
    hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Exclude near-black pixels (val < 30) and oversaturated reflections (val > 240, sat < 20)
    mask = cv2.inRange(hsv,
                       np.array([0, 0, 30], dtype=np.uint8),
                       np.array([180, 255, 235], dtype=np.uint8))
    hsv_filtered = hsv[mask > 0]

    if len(hsv_filtered) < 50:
        hsv_filtered = hsv.reshape(-1, 3)

    # Count votes for each color range
    votes = {}
    for entry in COLOR_RANGES:
        name, hex_col, h_lo, h_hi, s_lo, v_lo = entry
        h_vals = hsv_filtered[:, 0]
        s_vals = hsv_filtered[:, 1]
        v_vals = hsv_filtered[:, 2]

        match = (
            (h_vals >= h_lo) & (h_vals <= h_hi) &
            (s_vals >= s_lo) &
            (v_vals >= v_lo)
        )
        count = int(np.sum(match))
        if count > 0:
            key = name
            if key not in votes or votes[key][0] < count:
                votes[key] = (count, hex_col)

    if not votes:
        return {'color': 'Unknown', 'hex': '#888888', 'confidence': 0}

    best       = max(votes.items(), key=lambda x: x[1][0])
    color_name = best[0]
    color_hex  = best[1][1]
    total_px   = len(hsv_filtered)
    confidence = min(99, int(best[1][0] / max(total_px, 1) * 100 * 2))

    return {
        'color':      color_name,
        'hex':        color_hex,
        'confidence': confidence,
    }


# ── Vehicle Type Estimation ────────────────────────────────────────────────────
def estimate_vehicle_type(frame: np.ndarray, plate_bbox: dict) -> str:
    """
    Estimate vehicle class from plate position + relative size in frame.
    Heuristic only — no second model required.

    Logic:
    - Plate very wide relative to frame → likely truck/bus
    - Plate very small + near top → motorcycle (far away)
    - Plate in bottom half + standard AR → car/suv
    - Plate high in frame + large → van/bus
    """
    if frame is None or plate_bbox is None:
        return 'Unknown'

    h_f, w_f = frame.shape[:2]
    pw, ph = plate_bbox['w'], plate_bbox['h']
    py     = plate_bbox['y']

    plate_area_ratio = (pw * ph) / (w_f * h_f)
    plate_width_ratio = pw / w_f
    plate_y_ratio     = py / h_f   # 0=top, 1=bottom

    if plate_width_ratio > 0.45:
        return 'Truck/Bus'
    if plate_area_ratio < 0.003:
        return 'Motorcycle'
    if plate_y_ratio < 0.35 and plate_width_ratio > 0.25:
        return 'Van'
    if plate_area_ratio > 0.04:
        return 'SUV'
    return 'Car'


# ── Combined Analysis ──────────────────────────────────────────────────────────
def analyze_vehicle(frame: np.ndarray, plate_bbox: dict) -> dict:
    """
    Run both color and type detection.
    Returns dict merged into detection result.
    """
    color_info = detect_vehicle_color(frame, plate_bbox)
    vtype_hint = estimate_vehicle_type(frame, plate_bbox)

    return {
        'detected_color':      color_info['color'],
        'detected_color_hex':  color_info['hex'],
        'color_confidence':    color_info['confidence'],
        'detected_type_hint':  vtype_hint,
    }
