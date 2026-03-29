"""
manual_detection.py — High-Accuracy Manual Image Detection
Accepts uploaded images, runs 10-pass YOLO+contour detection at imgsz=640.
More accurate than live detection — no frame skipping, no shortcuts.
"""

import cv2
import numpy as np
import base64
import re
import time
from datetime import datetime
from itertools import combinations as _combs
from typing import Optional

from yolo_model import load_model
from ocr_module import read_plate
from utils import fuzzy_lookup, assign_slot, make_access_decision
from database import log_entry, add_alert


# ── YOLO Result Filter ─────────────────────────────────────────────────────────
def _filter_yolo(results, min_conf=0.10) -> list:
    plates = []
    for r in results:
        for xyxy, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if float(conf) < min_conf:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            bw, bh = x2 - x1, y2 - y1
            if bw < 15 or bh < 8:
                continue
            ar = bw / bh if bh > 0 else 0
            if not (1.0 < ar < 9.0):
                continue
            plates.append({
                'x': x1, 'y': y1, 'w': bw, 'h': bh,
                'method': 'yolo', 'yolo_conf': float(conf),
            })
    return sorted(plates, key=lambda p: -p['yolo_conf'])[:3]


# ── Auto-crop: remove UI sidebars from browser screenshots ────────────────────
def _auto_crop(frame) -> tuple:
    h, w    = frame.shape[:2]
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    col_std = np.std(gray, axis=0)
    row_std = np.std(gray, axis=1)

    left = next((x for x in range(min(w//3, 500)) if col_std[x] > 20), 0)
    top  = next((y for y in range(min(h//4, 200)) if row_std[y] > 20), 0)
    right  = next((w-x for x in range(1, min(w//3, 500)) if col_std[w-x] > 20), w)
    bottom = next((h-y for y in range(1, min(h//4, 200)) if row_std[h-y] > 20), h)

    crop_w = right - left
    crop_h = bottom - top
    if crop_w < w * 0.95 or crop_h < h * 0.95:
        print(f"[AUTOCROP] {w}x{h} → {crop_w}x{crop_h}")
        return frame[top:bottom, left:right], left, top
    return frame, 0, 0


# ── Contour-Based Fallback ─────────────────────────────────────────────────────
def _contour_detect(img, y_offset=0, scale=1) -> list:
    h, w    = img.shape[:2]
    img_area = w * h
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(enhanced, 9, 17, 17)
    all_cands = []

    # Strategy A: Sobel edges
    sx = cv2.Sobel(bilateral, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(bilateral, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for kw in [40, 25, 15]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 4))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 30 or bh < 6:
                continue
            ar = bw / bh if bh > 0 else 0
            if not (2.5 < ar < 9.5):
                continue
            area = bw * bh
            if area < img_area * 0.0003 or area > img_area * 0.20:
                continue
            cx, cy = x + bw/2, y + bh/2
            score  = area * min(ar/4.0, 2.0) * (1 + 1.0 - abs(cx/w - 0.5)*1.5) * (1.5 if 0.25 < cy/h < 0.88 else 0.4)
            all_cands.append({
                'x': int(x/scale), 'y': int(y/scale) + y_offset,
                'w': int(bw/scale), 'h': int(bh/scale),
                'method': f'contour_sobel', 'yolo_conf': 0.28,
                '_score': score,
            })

    # Strategy B: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=13, C=3
    )
    for kw in [35, 20]:
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 4))
        closed_a = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_h)
        contours, _ = cv2.findContours(closed_a, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 30 or bh < 6:
                continue
            ar = bw / bh if bh > 0 else 0
            if not (2.5 < ar < 9.5):
                continue
            area = bw * bh
            if area < img_area * 0.0003 or area > img_area * 0.20:
                continue
            cx, cy = x + bw/2, y + bh/2
            score  = area * min(ar/4.0, 2.0) * (1 + 1.0 - abs(cx/w - 0.5)*1.5) * (1.5 if 0.25 < cy/h < 0.88 else 0.4)
            all_cands.append({
                'x': int(x/scale), 'y': int(y/scale) + y_offset,
                'w': int(bw/scale), 'h': int(bh/scale),
                'method': 'contour_adapt', 'yolo_conf': 0.26,
                '_score': score,
            })

    if not all_cands:
        return []

    # Deduplicate by IoU > 0.25
    all_cands.sort(key=lambda c: -c['_score'])
    final = []
    for cand in all_cands:
        overlap = any(
            (lambda ix1, iy1, ix2, iy2:
             (ix2 > ix1 and iy2 > iy1 and
              (ix2-ix1)*(iy2-iy1) / max(1, cand['w']*cand['h'] + kept['w']*kept['h'] - (ix2-ix1)*(iy2-iy1)) > 0.25)
             )(max(cand['x'], kept['x']), max(cand['y'], kept['y']),
               min(cand['x']+cand['w'], kept['x']+kept['w']),
               min(cand['y']+cand['h'], kept['y']+kept['h']))
            for kept in final
        )
        if not overlap:
            final.append(cand)
        if len(final) >= 4:
            break
    return final


# ── 10-Pass Detection Pipeline ─────────────────────────────────────────────────
def multi_pass_detect(frame) -> list:
    """
    Try up to 10 detection strategies until a plate is found.
    imgsz=640 throughout — matches training resolution for best accuracy.
    """
    model  = load_model()
    h, w   = frame.shape[:2]
    y_bot  = int(h * 0.40)
    y_top  = int(h * 0.60)
    bottom = frame[y_bot:, :]
    top    = frame[:y_top, :]

    def _yolo(img, conf=0.20):
        try:
            return _filter_yolo(model(img, conf=conf, iou=0.45, verbose=False, imgsz=640), conf)
        except Exception as e:
            print(f"[YOLO ERR] {e}")
            return []

    # Pass 1: YOLO full, conf=0.20
    if plates := _yolo(frame, 0.20):
        print(f"[DETECT] Pass 1 (YOLO full 0.20): {len(plates)}")
        return plates

    # Pass 2: YOLO full, conf=0.10
    if plates := _yolo(frame, 0.10):
        print(f"[DETECT] Pass 2 (YOLO full 0.10): {len(plates)}")
        return plates

    # Pass 3: YOLO on auto-cropped (removes sidebar/chrome)
    cropped, cx_off, cy_off = _auto_crop(frame)
    if cropped.shape != frame.shape:
        plates = _yolo(cropped, 0.10)
        if plates:
            for p in plates:
                p['x'] += cx_off; p['y'] += cy_off
            print(f"[DETECT] Pass 3 (YOLO auto-crop): {len(plates)}")
            return plates

    # Pass 4: YOLO top 60%
    plates = _yolo(top, 0.10)
    if plates:
        print(f"[DETECT] Pass 4 (YOLO top): {len(plates)}")
        return plates

    # Pass 5: YOLO bottom 60%
    plates = _yolo(bottom, 0.10)
    if plates:
        for p in plates: p['y'] += y_bot
        print(f"[DETECT] Pass 5 (YOLO bottom): {len(plates)}")
        return plates

    # Pass 6: YOLO 2× upscale
    up2    = cv2.resize(frame, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    plates = _yolo(up2, 0.10)
    if plates:
        for p in plates:
            p['x'] //= 2; p['y'] //= 2
            p['w'] = max(1, p['w']//2); p['h'] = max(1, p['h']//2)
        print(f"[DETECT] Pass 6 (YOLO 2x): {len(plates)}")
        return plates

    # Pass 7: Contour full
    if plates := _contour_detect(frame):
        print(f"[DETECT] Pass 7 (contour full): {len(plates)}")
        return plates

    # Pass 8: Contour 4× upscale
    up4f   = cv2.resize(frame, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    if plates := _contour_detect(up4f, scale=4):
        print(f"[DETECT] Pass 8 (contour 4x): {len(plates)}")
        return plates

    # Pass 9: Contour 4× bottom
    up4b   = cv2.resize(bottom, (w*4, bottom.shape[0]*4), interpolation=cv2.INTER_CUBIC)
    if plates := _contour_detect(up4b, y_offset=y_bot, scale=4):
        print(f"[DETECT] Pass 9 (contour 4x bottom): {len(plates)}")
        return plates

    # Pass 10: YOLO 4× upscale
    up4    = cv2.resize(frame, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    plates = _yolo(up4, 0.10)
    if plates:
        for p in plates:
            p['x'] //= 4; p['y'] //= 4
            p['w'] = max(1, p['w']//4); p['h'] = max(1, p['h']//4)
        print(f"[DETECT] Pass 10 (YOLO 4x): {len(plates)}")
        return plates

    print("[DETECT] All 10 passes exhausted — no plate found")
    return []


# ── Main Entry Point ───────────────────────────────────────────────────────────
def process_image(image_b64: str) -> tuple[dict, int]:
    """
    Full high-accuracy pipeline for uploaded images.
    Returns (result_dict, http_status_code)
    """
    try:
        # Decode image
        raw = image_b64.split(',', 1)[1] if image_b64.startswith('data:') else image_b64
        nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {'error': 'Invalid image data', 'success': False}, 400

        # Smart resize — preserve AR, cap longest side at 1280px
        h0, w0 = frame.shape[:2]
        if max(h0, w0) > 1280:
            s     = 1280 / max(h0, w0)
            frame = cv2.resize(frame, (int(w0*s), int(h0*s)), interpolation=cv2.INTER_AREA)
        print(f"[MANUAL] Input {w0}x{h0} → {frame.shape[1]}x{frame.shape[0]}")

        # Detect plates
        plates = multi_pass_detect(frame)

        if not plates:
            return {
                'success':  False,
                'message':  'No plate detected. Try: closer crop, better lighting, or less blur.',
                'detections': 0,
                'annotated_frame': None,
            }, 200

        results     = []
        annotated   = frame.copy()

        for idx, p in enumerate(plates[:3]):
            x, y, bw, bh = p['x'], p['y'], p['w'], p['h']

            # Generous padding for OCR
            pad_x = int(bw * 0.20)
            pad_y = int(bh * 0.35)
            roi   = frame[
                max(0, y-pad_y):min(frame.shape[0], y+bh+pad_y),
                max(0, x-pad_x):min(frame.shape[1], x+bw+pad_x)
            ]

            if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 15:
                continue

            # OCR
            ocr_text, ocr_conf, ocr_raw = read_plate(roi, fast=False)

            # Fuzzy match
            info, matched, dist = fuzzy_lookup(ocr_raw if ocr_raw else ocr_text)
            if matched:
                ocr_text = matched
            elif not ocr_text:
                ocr_text = re.sub(r'[^A-Z0-9]', '', ocr_raw)[:12] or 'UNREADABLE'

            # Access decision
            decision = make_access_decision(ocr_text, ocr_conf, dist)

            # Crop + threshold images for UI
            _, buf = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
            crop_b64 = base64.b64encode(buf).decode()

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
            _, thresh = cv2.threshold(
                cv2.resize(gray_roi, (300, 80)), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            _, tbuf = cv2.imencode('.jpg', thresh, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thresh_b64 = base64.b64encode(tbuf).decode()

            # Log
            log_entry(
                plate=ocr_text, action=decision['action'],
                confidence=decision['conf'], ocr_confidence=ocr_conf,
                slot=decision['slot'], ocr_raw=ocr_raw or ocr_text,
                vtype=decision['vtype'], plate_crop_b64=crop_b64
            )

            if not decision['authorized']:
                add_alert(
                    'Unknown Plate' if not decision['registered'] else 'Blacklisted',
                    ocr_text,
                    f"Plate {ocr_text} denied (conf:{ocr_conf:.0f}%, dist:{dist})"
                )

            # Draw on annotated frame
            col = (0, 210, 60) if decision['authorized'] else (0, 50, 230)
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), col, 3)
            cv2.putText(annotated, ocr_text, (x, max(y-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            results.append({
                **decision,
                'ocr_raw':      ocr_raw or ocr_text,
                'crop_b64':     crop_b64,
                'thresh_b64':   thresh_b64,
                'bbox':         p,
                'timestamp':    datetime.now().strftime('%H:%M:%S'),
                'xPct':         round(p['x'] / frame.shape[1] * 100, 1),
                'yPct':         round(p['y'] / frame.shape[0] * 100, 1),
            })

        if not results:
            return {'success': False, 'message': 'Plates found but OCR failed.', 'detections': 0}, 200

        _, abuf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(abuf).decode()

        return {
            'success':         True,
            'detections':      len(results),
            'results':         results,
            'best_result':     results[0],
            'annotated_frame': annotated_b64,
            'image_size':      {'w': frame.shape[1], 'h': frame.shape[0]},
            'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, 200

    except Exception as e:
        print(f"[MANUAL ERROR] {e}")
        import traceback; traceback.print_exc()
        return {'error': str(e), 'success': False}, 500
