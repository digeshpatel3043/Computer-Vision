"""
live_detection.py — Real-Time Camera Detection
Optimised for speed: frame-skipping, imgsz=320, async OCR worker.
"""

import cv2
import numpy as np
import base64
import re
import threading
import time
from datetime import datetime

from yolo_model import load_model
from ocr_module import read_plate
from utils import fuzzy_lookup, assign_slot, get_all_slots
from database import log_entry, add_alert
from night_mode import enhance_frame, night_state, enhance_roi_for_ocr
from vehicle_attributes import analyze_vehicle
from speed_estimator import speed_estimator

# ── Tuning ─────────────────────────────────────────────────────────────────────
FRAME_SKIP    = 3       # run YOLO every Nth frame → 66% fewer calls
DETECTION_COOLDOWN = 1.5  # seconds between OCR submissions
RESULT_TTL    = 15.0    # seconds to keep last result visible in overlay


# ── YOLO Detect ────────────────────────────────────────────────────────────────
_frame_counter = 0
_last_plates   = []


def detect_plates_fast(frame) -> list:
    """
    Fast YOLO detection for live stream.
    - imgsz=320 (2× faster than 640)
    - Skips every FRAME_SKIP-1 frames
    - Returns cached result on skipped frames
    """
    global _frame_counter, _last_plates
    _frame_counter += 1

    if _frame_counter % FRAME_SKIP != 0:
        return _last_plates

    model = load_model()
    try:
        results = model(frame, conf=0.20, iou=0.45, verbose=False, imgsz=320)
        plates  = []
        for r in results:
            for xyxy, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, xyxy)
                bw, bh = x2-x1, y2-y1
                if bw < 20 or bh < 10:
                    continue
                ar = bw / bh if bh > 0 else 0
                if not (1.2 < ar < 7.5):
                    continue
                plates.append({
                    'x': x1, 'y': y1, 'w': bw, 'h': bh,
                    'method': 'yolo', 'yolo_conf': float(conf),
                })
        _last_plates = sorted(plates, key=lambda p: -p['yolo_conf'])[:3]
        return _last_plates
    except Exception as e:
        print(f"[LIVE YOLO ERROR] {e}")
        return []


# ── Async OCR Worker ───────────────────────────────────────────────────────────
class OCRWorker:
    """
    Background thread that processes plate crops asynchronously.
    Never blocks the MJPEG stream loop.
    """
    def __init__(self):
        self.lock         = threading.Lock()
        self.pending      = None
        self.busy         = False
        self.raw_reading  = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, roi, frame, plate_bbox):
        with self.lock:
            self.pending = (roi.copy(), frame.copy(), dict(plate_bbox))

    @property
    def is_busy(self):
        with self.lock:
            return self.busy

    def _loop(self):
        while True:
            job = None
            with self.lock:
                if self.pending and not self.busy:
                    job          = self.pending
                    self.pending = None
                    self.busy    = True
            if job:
                try:
                    self._process(*job)
                except Exception as e:
                    print(f"[OCR WORKER ERROR] {e}")
                finally:
                    with self.lock:
                        self.busy = False
            time.sleep(0.05)

    def _process(self, roi, frame, p):
        # Night-mode enhanced ROI for better OCR in dark conditions
        if night_state.active:
            roi = enhance_roi_for_ocr(roi)

        ocr_text, ocr_conf, ocr_raw = read_plate(roi, fast=True)

        # Partial result for immediate UI feedback
        with self.lock:
            self.raw_reading = {
                'plate':      ocr_text or ocr_raw[:15],
                'confidence': ocr_conf,
                'status':     'READING',
                'timestamp':  datetime.now().strftime('%H:%M:%S'),
            }

        lookup  = ocr_raw if ocr_raw else ocr_text
        info, matched, dist = fuzzy_lookup(lookup)

        if matched:
            ocr_text = matched
        elif not ocr_text and ocr_raw:
            ocr_text = re.sub(r'[^A-Z0-9]', '', ocr_raw)[:12]
        elif not ocr_text:
            ocr_text = 'READING...'

        # Vehicle color + type detection
        attrs = analyze_vehicle(frame, p)

        # Speed estimation
        speed_result = speed_estimator.update(ocr_text, p, frame.shape[0])

        auth   = bool(info and info['category'] != 'Blacklisted')
        action = 'ENTRY' if auth else 'DENIED'
        slot   = assign_slot(ocr_text) if auth else None

        fuzzy_penalty = dist * 5 if dist < 999 else 30
        conf_total    = max(10, round(min(99, ocr_conf*0.7 + (40 if auth else 20) - fuzzy_penalty), 1))

        # Encode crops (lower quality for speed)
        _, buf  = cv2.imencode('.jpg', roi,   [cv2.IMWRITE_JPEG_QUALITY, 70])
        crop_b64 = base64.b64encode(buf).decode()
        _, buf2 = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        frame_b64 = base64.b64encode(buf2).decode()

        result = {
            'plate':      ocr_text,
            'ocr_raw':    ocr_raw or ocr_text,
            'ocr_conf':   round(ocr_conf, 1),
            'conf':       conf_total,
            'authorized': auth,
            'action':     action,
            'owner':      info['owner']    if info else 'Unknown',
            'vtype':      info['vtype']    if info else attrs['detected_type_hint'],
            'flat':       info['flat']     if info else '—',
            'category':   info['category'] if info else 'Unregistered',
            'slot':       slot or '—',
            'match_dist': dist,
            'bbox':       p,
            'crop_b64':   crop_b64,
            'frame_b64':  frame_b64,
            'registered': bool(info),
            'timestamp':  datetime.now().strftime('%H:%M:%S'),
            'night_mode': night_state.active,
            'speed':      speed_result,
            **attrs,
        }

        log_entry(
            plate=ocr_text, action=action,
            confidence=conf_total, ocr_confidence=ocr_conf,
            slot=slot or '—', ocr_raw=ocr_raw or ocr_text,
            vtype=result['vtype'], plate_crop_b64=crop_b64, image_b64=frame_b64
        )

        if not auth:
            add_alert(
                'Unknown Plate' if not info else 'Blacklisted',
                ocr_text,
                f"Plate {ocr_text} denied (conf:{ocr_conf:.0f}%, dist:{dist})"
            )

        if speed_result and speed_result['overspeed']:
            add_alert('Overspeed', ocr_text,
                      f"{ocr_text} at {speed_result['speed_kmh']} km/h (limit {speed_estimator.OVERSPEED_KMH} km/h)", 'high')

        cam_state.current_result = result
        cam_state.last_det_time  = time.time()
        print(f"[LIVE] {ocr_text} | {action} | {attrs['detected_color']} {result['vtype']} | conf={conf_total}%")


# ── Camera State ───────────────────────────────────────────────────────────────
class CameraState:
    def __init__(self):
        self.cap            = None
        self.running        = False
        self.lock           = threading.Lock()
        self.latest_frame   = None
        self.frame_count    = 0
        self.last_det_time  = 0
        self.current_result = None
        self.mode           = 'idle'

    def start(self, camera_index: int = 0) -> bool:
        self.stop()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[CAM] Cannot open camera index {camera_index}")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always latest frame

        self.cap     = cap
        self.running = True
        self.mode    = 'live'
        threading.Thread(target=self._capture_loop, daemon=True).start()
        print(f"[CAM] Started camera index {camera_index}")
        return True

    def _capture_loop(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
            time.sleep(0.033)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mode = 'idle'


# ── Singletons (shared with app.py) ───────────────────────────────────────────
cam_state  = CameraState()
ocr_worker = OCRWorker()


# ── Frame Drawing ──────────────────────────────────────────────────────────────
def draw_overlay(frame, plates, detection_result=None) -> np.ndarray:
    """Annotate frame with bounding boxes, plate text, and banner."""
    out  = frame.copy()
    h_f, w_f = frame.shape[:2]

    for i, p in enumerate(plates):
        x, y, bw, bh = p['x'], p['y'], p['w'], p['h']
        col = (0, 175, 245)
        if detection_result and i == 0:
            col = (0, 210, 60) if detection_result.get('authorized') else (0, 50, 230)

        cv2.rectangle(out, (x, y), (x+bw, y+bh), col, 2)

        if detection_result and i == 0:
            lbl = f"{detection_result.get('plate','?')} [{detection_result.get('ocr_conf',0):.0f}%]"
            (tw, txh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            by2 = max(y - txh - 10, 0)
            cv2.rectangle(out, (x, by2), (x+tw+8, by2+txh+8), col, -1)
            cv2.putText(out, lbl, (x+4, by2+txh+2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
        else:
            cv2.putText(out, 'READING...', (x, max(y-6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 136), 2)

    # Top banner
    if detection_result and detection_result.get('plate') not in [None, 'READING...']:
        auth  = detection_result.get('authorized', False)
        col_b = (0, 180, 60) if auth else (0, 40, 220)
        cv2.rectangle(out, (0, 0), (w_f, 60), (0, 0, 0), -1)
        cv2.rectangle(out, (0, 0), (w_f, 60), col_b, 2)
        status = f"{'ALLOWED' if auth else 'DENIED'} | {detection_result.get('plate','')} | {detection_result.get('owner','—')}"
        cv2.putText(out, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_b, 2)
        detail = f"Type:{detection_result.get('vtype','—')}  Flat:{detection_result.get('flat','—')}  Conf:{detection_result.get('conf',0):.0f}%"
        cv2.putText(out, detail, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)

    # Bottom bar
    cv2.rectangle(out, (0, h_f-28), (w_f, h_f), (0, 0, 0), -1)
    cv2.putText(out, f"ANPR LIVE | {datetime.now().strftime('%H:%M:%S')} | Plates:{len(plates)}",
                (6, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 100), 1)
    return out


# ── MJPEG Stream Generator ─────────────────────────────────────────────────────
def generate_mjpeg():
    """
    Yield MJPEG frames for /video_feed endpoint.
    Multi-frame confirmation: plate must appear in 2+ consecutive frames before OCR.
    """
    consecutive = 0
    last_area   = 0

    while True:
        frame = cam_state.get_frame()

        if frame is None:
            # Standby screen
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, 'ANPR SYSTEM — Click "Start Camera"',
                        (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 80), 2)
            cv2.putText(blank, datetime.now().strftime('%Y-%m-%d  %H:%M:%S'),
                        (160, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,60,60), 1)
            _, j = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + j.tobytes() + b'\r\n'
            time.sleep(0.05)
            continue

        now = time.time()

        # Night mode adaptive enhancement (auto / manual override)
        enhanced = enhance_frame(frame)

        plates = detect_plates_fast(enhanced)

        # Multi-frame confirmation
        if plates:
            area = plates[0]['w'] * plates[0]['h']
            consecutive = consecutive + 1 if last_area > 0 and abs(area - last_area) / last_area < 0.30 else 1
            last_area   = area
        else:
            consecutive = 0
            last_area   = 0

        # Submit OCR when confirmed + cooldown elapsed + worker free
        if (plates and consecutive >= 2 and
                (now - cam_state.last_det_time) > DETECTION_COOLDOWN and
                not ocr_worker.is_busy):

            cam_state.last_det_time = now
            p = plates[0]
            x, y, bw, bh = p['x'], p['y'], p['w'], p['h']
            pad_x = int(bw * 0.25)
            pad_y = int(bh * 0.40)
            roi   = enhanced[
                max(0, y-pad_y):min(enhanced.shape[0], y+bh+pad_y),
                max(0, x-pad_x):min(enhanced.shape[1], x+bw+pad_x)
            ]
            if roi.size > 0:
                ocr_worker.submit(roi, enhanced, p)

        # Show result while fresh
        current  = cam_state.current_result
        active   = current and (now - cam_state.last_det_time) < RESULT_TTL
        annotated = draw_overlay(enhanced, plates, current if active else None)

        # Draw speed estimation zones
        speed_estimator.draw_zones(annotated)

        # Night mode indicator
        if night_state.active:
            cv2.putText(annotated, '🌙 NIGHT MODE', (4, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Scanning indicator
        if ocr_worker.is_busy and plates:
            h_f, w_f = annotated.shape[:2]
            cv2.rectangle(annotated, (0, h_f-20), (w_f, h_f), (0,0,0), -1)
            cv2.putText(annotated, 'SCANNING PLATE...',
                        (10, h_f-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        time.sleep(0.033)