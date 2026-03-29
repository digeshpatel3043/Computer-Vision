"""
multi_camera.py — Multi-Camera Manager
Run up to 4 cameras simultaneously, each with its own:
  - Capture thread
  - OCR worker
  - Detection state
  - MJPEG stream endpoint

Each camera gets an ID (0-3). The UI can display all 4 streams
or switch between them.

Usage in app.py:
    from multi_camera import multi_cam
    multi_cam.start_camera(0)
    multi_cam.start_camera(1)

Stream endpoints:  /video_feed/0  /video_feed/1  etc.
"""

import cv2
import numpy as np
import base64
import re
import threading
import time
from datetime import datetime
from typing import Dict, Optional

from yolo_model import load_model
from ocr_module import read_plate
from utils import fuzzy_lookup, assign_slot
from database import log_entry, add_alert
from night_mode import enhance_frame
from vehicle_attributes import analyze_vehicle
from speed_estimator import speed_estimator

# ── Per-camera tuning ──────────────────────────────────────────────────────────
FRAME_SKIP        = 3
DETECTION_COOLDOWN = 2.0
RESULT_TTL        = 15.0
MAX_CAMERAS       = 4


# ── Single Camera Instance ─────────────────────────────────────────────────────
class CameraInstance:
    def __init__(self, cam_id: int):
        self.cam_id         = cam_id
        self.name           = f'Camera {cam_id}'
        self.cap            = None
        self.running        = False
        self.lock           = threading.Lock()
        self.latest_frame   = None
        self.frame_count    = 0
        self.last_det_time  = 0
        self.current_result = None
        self.mode           = 'idle'
        self._frame_counter = 0
        self._last_plates   = []
        self._consecutive   = 0
        self._last_area     = 0
        self.ocr_worker     = _OCRWorker(cam_id)

    # ── Camera Control ────────────────────────────────────────────────────────
    def start(self, index: int = None, rtsp_url: str = None) -> bool:
        self.stop()

        # RTSP URL takes priority (IP cameras, dashcams etc.)
        if rtsp_url:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                print(f"[CAM-{self.cam_id}] Cannot open RTSP: {rtsp_url}")
                return False
        else:
            src = index if index is not None else self.cam_id

            # ── Windows: try DirectShow first to avoid black-frame delay ────
            import platform
            cap = None
            if platform.system() == 'Windows':
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                    try:
                        t = cv2.VideoCapture(src, backend)
                        if t.isOpened():
                            cap = t
                            print(f"[CAM-{self.cam_id}] Opened with backend {backend}")
                            break
                        t.release()
                    except Exception:
                        continue
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(src)

            if not cap.isOpened():
                print(f"[CAM-{self.cam_id}] Cannot open camera index {src}")
                return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          25)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # Discard first 5 frames (Windows cameras emit black frames on open)
        for _ in range(5):
            cap.read()

        self.cap     = cap
        self.running = True
        self.mode    = 'live'
        threading.Thread(target=self._capture_loop, daemon=True).start()
        print(f"[CAM-{self.cam_id}] Started → index={index}")
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mode = 'idle'
        print(f"[CAM-{self.cam_id}] Stopped")

    def _capture_loop(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.frame_count += 1
            time.sleep(0.033)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    # ── Detection ─────────────────────────────────────────────────────────────
    def _detect_plates(self, frame) -> list:
        self._frame_counter += 1
        if self._frame_counter % FRAME_SKIP != 0:
            return self._last_plates
        model = load_model()
        try:
            results = model(frame, conf=0.20, iou=0.45, verbose=False, imgsz=320)
            plates  = []
            for r in results:
                for xyxy, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, xyxy)
                    bw, bh = x2-x1, y2-y1
                    if bw < 20 or bh < 10: continue
                    ar = bw / bh if bh > 0 else 0
                    if not (1.2 < ar < 7.5): continue
                    plates.append({'x': x1, 'y': y1, 'w': bw, 'h': bh,
                                   'method': 'yolo', 'yolo_conf': float(conf)})
            self._last_plates = sorted(plates, key=lambda p: -p['yolo_conf'])[:3]
            return self._last_plates
        except Exception as e:
            print(f"[CAM-{self.cam_id} YOLO] {e}")
            return []

    # ── MJPEG Stream ──────────────────────────────────────────────────────────
    def generate_stream(self):
        """Generator for MJPEG stream of this camera."""
        self._consecutive = 0
        self._last_area   = 0

        while True:
            frame = self.get_frame()

            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                for row in range(0, 480, 40):
                    cv2.line(blank, (0, row), (640, row), (18, 22, 28), 1)
                for col in range(0, 640, 40):
                    cv2.line(blank, (col, 0), (col, 480), (18, 22, 28), 1)
                cv2.putText(blank, f'CAMERA {self.cam_id} — NOT STARTED',
                            (90, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 70), 2)
                cv2.putText(blank, 'Click Start in Multi-Cam panel',
                            (155, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 60), 1)
                cv2.putText(blank, f'CAM {self.cam_id}  |  {datetime.now().strftime("%H:%M:%S")}',
                            (230, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 40, 50), 1)
                _, j = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + j.tobytes() + b'\r\n'
                time.sleep(0.1)
                continue

            now      = time.time()
            enhanced = enhance_frame(frame)
            plates   = self._detect_plates(enhanced)

            # Multi-frame confirmation
            if plates:
                area = plates[0]['w'] * plates[0]['h']
                same = self._last_area > 0 and abs(area - self._last_area) / self._last_area < 0.30
                self._consecutive = self._consecutive + 1 if same else 1
                self._last_area   = area
            else:
                self._consecutive = 0
                self._last_area   = 0

            if (plates and self._consecutive >= 2 and
                    (now - self.last_det_time) > DETECTION_COOLDOWN and
                    not self.ocr_worker.is_busy):
                self.last_det_time = now
                p = plates[0]
                x, y, bw, bh = p['x'], p['y'], p['w'], p['h']
                pad_x = int(bw * 0.25)
                pad_y = int(bh * 0.40)
                roi   = enhanced[
                    max(0, y-pad_y):min(enhanced.shape[0], y+bh+pad_y),
                    max(0, x-pad_x):min(enhanced.shape[1], x+bw+pad_x)
                ]
                if roi.size > 0:
                    self.ocr_worker.submit(roi, enhanced, p, self)

            current  = self.current_result
            active   = current and (now - self.last_det_time) < RESULT_TTL
            annotated = self._draw_overlay(enhanced, plates, current if active else None)

            # Camera label
            cv2.putText(annotated, f'CAM {self.cam_id} | {self.name}',
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            time.sleep(0.033)

    def _draw_overlay(self, frame, plates, result=None) -> np.ndarray:
        out = frame.copy()
        h_f, w_f = frame.shape[:2]
        for i, p in enumerate(plates):
            x, y, bw, bh = p['x'], p['y'], p['w'], p['h']
            col = (0,175,245)
            if result and i == 0:
                col = (0,210,60) if result.get('authorized') else (0,50,230)
            cv2.rectangle(out, (x,y), (x+bw,y+bh), col, 2)
            if result and i == 0:
                lbl = f"{result.get('plate','?')} [{result.get('ocr_conf',0):.0f}%]"
                cv2.putText(out, lbl, (x, max(y-6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if result and result.get('plate') not in [None, 'READING...']:
            auth  = result.get('authorized', False)
            col_b = (0,180,60) if auth else (0,40,220)
            cv2.rectangle(out, (0, h_f-24), (w_f, h_f), (0,0,0), -1)
            label = f"{'ALLOWED' if auth else 'DENIED'} | {result.get('plate','')} | {result.get('owner','—')}"
            cv2.putText(out, label, (6, h_f-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_b, 1)
        return out

    def status(self) -> dict:
        return {
            'cam_id':   self.cam_id,
            'name':     self.name,
            'mode':     self.mode,
            'running':  self.running,
            'frames':   self.frame_count,
            'result':   self.current_result,
        }


# ── Per-Camera OCR Worker ──────────────────────────────────────────────────────
class _OCRWorker:
    def __init__(self, cam_id: int):
        self.cam_id  = cam_id
        self.lock    = threading.Lock()
        self.pending = None
        self.busy    = False
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, roi, frame, p, cam: CameraInstance):
        with self.lock:
            self.pending = (roi.copy(), frame.copy(), dict(p), cam)

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
                    print(f"[OCR-CAM{self.cam_id}] {e}")
                finally:
                    with self.lock:
                        self.busy = False
            time.sleep(0.05)

    def _process(self, roi, frame, p, cam: CameraInstance):
        ocr_text, ocr_conf, ocr_raw = read_plate(roi, fast=True)
        lookup = ocr_raw if ocr_raw else ocr_text
        info, matched, dist = fuzzy_lookup(lookup)

        if matched:
            ocr_text = matched
        elif not ocr_text:
            ocr_text = re.sub(r'[^A-Z0-9]', '', ocr_raw or '')[:12] or 'READING...'

        # Vehicle attributes (color + type hint)
        attrs = analyze_vehicle(frame, p)

        # Speed estimation
        speed_result = speed_estimator.update(ocr_text, p, frame.shape[0])

        auth   = bool(info and info['category'] != 'Blacklisted')
        action = 'ENTRY' if auth else 'DENIED'
        slot   = assign_slot(ocr_text) if auth else None

        fuzzy_penalty = dist * 5 if dist < 999 else 30
        conf_total    = max(10, round(min(99, ocr_conf*0.7 + (40 if auth else 20) - fuzzy_penalty), 1))

        _, buf   = cv2.imencode('.jpg', roi,   [cv2.IMWRITE_JPEG_QUALITY, 70])
        crop_b64 = base64.b64encode(buf).decode()
        _, buf2  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        frame_b64 = base64.b64encode(buf2).decode()

        result = {
            'plate':      ocr_text,
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
            'camera':     cam.cam_id,
            **attrs,
            'speed':      speed_result,
        }

        gate_name = f'Gate {chr(65 + cam.cam_id)}'  # Gate A, B, C, D
        log_entry(plate=ocr_text, action=action, gate=gate_name,
                  confidence=conf_total, ocr_confidence=ocr_conf,
                  slot=slot or '—', ocr_raw=ocr_raw or ocr_text,
                  vtype=result['vtype'], plate_crop_b64=crop_b64, image_b64=frame_b64)

        if not auth:
            add_alert('Unknown Plate' if not info else 'Blacklisted', ocr_text,
                      f"[CAM-{cam.cam_id}] {ocr_text} denied")

        if speed_result and speed_result['overspeed']:
            add_alert('Overspeed', ocr_text,
                      f"{ocr_text} travelling at {speed_result['speed_kmh']} km/h (limit {speed_estimator.OVERSPEED_KMH})", 'high')

        cam.current_result = result
        cam.last_det_time  = time.time()
        print(f"[CAM-{cam.cam_id}] {ocr_text} | {action} | {attrs['detected_color']} {result['vtype']}")


# ── Multi-Camera Manager ───────────────────────────────────────────────────────
class MultiCameraManager:
    def __init__(self):
        self.cameras: Dict[int, CameraInstance] = {}
        self.lock = threading.Lock()

    def start_camera(self, cam_id: int, index: int = None, rtsp_url: str = None) -> bool:
        with self.lock:
            if cam_id not in self.cameras:
                self.cameras[cam_id] = CameraInstance(cam_id)
            cam = self.cameras[cam_id]
        src = index if index is not None else cam_id
        return cam.start(index=src, rtsp_url=rtsp_url)

    def stop_camera(self, cam_id: int):
        with self.lock:
            cam = self.cameras.get(cam_id)
        if cam:
            cam.stop()

    def stop_all(self):
        with self.lock:
            cams = list(self.cameras.values())
        for cam in cams:
            cam.stop()

    def get_stream(self, cam_id: int):
        with self.lock:
            if cam_id not in self.cameras:
                self.cameras[cam_id] = CameraInstance(cam_id)
            return self.cameras[cam_id].generate_stream()

    def get_last_result(self, cam_id: int) -> Optional[dict]:
        with self.lock:
            cam = self.cameras.get(cam_id)
        return cam.current_result if cam else None

    def set_camera_name(self, cam_id: int, name: str):
        with self.lock:
            if cam_id in self.cameras:
                self.cameras[cam_id].name = name

    def all_status(self) -> list:
        with self.lock:
            return [cam.status() for cam in self.cameras.values()]


# Singleton
multi_cam = MultiCameraManager()