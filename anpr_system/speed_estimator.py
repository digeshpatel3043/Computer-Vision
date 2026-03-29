"""
speed_estimator.py — Vehicle Speed Estimation
Estimates speed between two horizontal detection zones in the frame.

How it works:
  - Frame is split into Zone A (top 40%) and Zone B (bottom 40%)
  - When a plate is confirmed in Zone A, timestamp is recorded
  - When same plate appears in Zone B, time delta is measured
  - Speed = known_distance_meters / time_delta_seconds → km/h

Configuration:
  ZONE_DISTANCE_M: Real-world distance between Zone A centre and Zone B centre.
  Measure this physically at your camera installation.
  Default: 5.0m (typical driveway length).

The speed estimate is approximate — camera angle and lens distortion affect accuracy.
It is good enough for "fast / normal / slow" classification and overspeed alerts.
"""

import time
import threading
from datetime import datetime
from typing import Optional, Dict

# ── Config ─────────────────────────────────────────────────────────────────────
ZONE_DISTANCE_M   = 5.0    # metres between Zone A and Zone B midpoints
OVERSPEED_KMH     = 20.0   # alert threshold (km/h) — typical parking lot limit
ENTRY_EXPIRY_S    = 10.0   # forget Zone A timestamp after N seconds (vehicle left)

# Zone boundaries as fraction of frame height
ZONE_A_Y_RANGE = (0.10, 0.40)   # top zone
ZONE_B_Y_RANGE = (0.60, 0.90)   # bottom zone


# ── State ──────────────────────────────────────────────────────────────────────
class SpeedEstimator:
    def __init__(self):
        self.lock         = threading.Lock()
        self._zone_a: Dict[str, float] = {}   # plate → timestamp when seen in Zone A
        self._results: Dict[str, dict] = {}   # plate → latest speed result
        self._history: list = []              # last 50 speed readings

    def get_plate_zone(self, plate_bbox: dict, frame_height: int) -> Optional[str]:
        """Return 'A', 'B', or None based on plate centre Y position."""
        cy_frac = (plate_bbox['y'] + plate_bbox['h'] / 2) / frame_height
        if ZONE_A_Y_RANGE[0] <= cy_frac <= ZONE_A_Y_RANGE[1]:
            return 'A'
        if ZONE_B_Y_RANGE[0] <= cy_frac <= ZONE_B_Y_RANGE[1]:
            return 'B'
        return None

    def update(self, plate: str, plate_bbox: dict, frame_height: int) -> Optional[dict]:
        """
        Called each time a plate is detected.
        Returns speed result dict if speed could be calculated, else None.
        """
        if not plate or plate in ('READING...', 'UNREADABLE'):
            return None

        zone = self.get_plate_zone(plate_bbox, frame_height)
        now  = time.time()

        with self.lock:
            # Clean up expired Zone A entries
            expired = [p for p, t in self._zone_a.items() if now - t > ENTRY_EXPIRY_S]
            for p in expired:
                del self._zone_a[p]

            if zone == 'A':
                # Record entry time into Zone A
                if plate not in self._zone_a:
                    self._zone_a[plate] = now
                    print(f"[SPEED] {plate} entered Zone A")
                return None

            elif zone == 'B' and plate in self._zone_a:
                # Calculate speed
                elapsed = now - self._zone_a[plate]
                if elapsed < 0.1:
                    return None   # impossibly fast → bad detection

                speed_ms  = ZONE_DISTANCE_M / elapsed
                speed_kmh = round(speed_ms * 3.6, 1)
                del self._zone_a[plate]

                result = {
                    'plate':      plate,
                    'speed_kmh':  speed_kmh,
                    'elapsed_s':  round(elapsed, 2),
                    'overspeed':  speed_kmh > OVERSPEED_KMH,
                    'timestamp':  datetime.now().strftime('%H:%M:%S'),
                }
                self._results[plate] = result
                self._history.insert(0, result)
                if len(self._history) > 50:
                    self._history.pop()

                tag = '🚨 OVERSPEED' if result['overspeed'] else '✓ OK'
                print(f"[SPEED] {plate}: {speed_kmh} km/h {tag}")
                return result

        return None

    def get_latest(self, plate: str) -> Optional[dict]:
        with self.lock:
            return self._results.get(plate)

    def get_history(self) -> list:
        with self.lock:
            return list(self._history)

    def get_zone_a_count(self) -> int:
        with self.lock:
            return len(self._zone_a)

    def status(self) -> dict:
        with self.lock:
            return {
                'plates_in_zone_a':  len(self._zone_a),
                'total_readings':    len(self._history),
                'overspeed_count':   sum(1 for r in self._history if r['overspeed']),
                'recent':            list(self._history[:5]),
                'zone_distance_m':   ZONE_DISTANCE_M,
                'overspeed_limit':   OVERSPEED_KMH,
            }

    def draw_zones(self, frame) -> None:
        """Draw zone overlays onto frame (in-place)."""
        import cv2
        h, w = frame.shape[:2]
        # Zone A — blue dashed line
        y_a = int(h * (ZONE_A_Y_RANGE[0] + ZONE_A_Y_RANGE[1]) / 2)
        cv2.line(frame, (0, y_a), (w, y_a), (255, 200, 0), 1)
        cv2.putText(frame, 'ZONE A', (5, y_a - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        # Zone B — orange dashed line
        y_b = int(h * (ZONE_B_Y_RANGE[0] + ZONE_B_Y_RANGE[1]) / 2)
        cv2.line(frame, (0, y_b), (w, y_b), (0, 180, 255), 1)
        cv2.putText(frame, 'ZONE B', (5, y_b - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)


# Singleton
speed_estimator = SpeedEstimator()
