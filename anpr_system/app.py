"""
app.py — Main Flask Application  ← THIS IS THE FILE YOU RUN
All API routes + startup.

Run:  python app.py
Open: http://localhost:5000
"""
import analytics
print(analytics.__file__)
import os
import threading
import time
from datetime import datetime
from flask import Flask, Response, request, jsonify, render_template

import database as db
from analytics import (
    ensure_analytics_columns,
    get_overview,
    get_hourly_heatmap,
    get_daily_trend,
    get_peak_hour_prediction,
    get_ocr_accuracy_trend,
    detect_anomalies,
    get_speed_analytics,
    get_color_stats,
    get_night_day_breakdown,
    get_action_breakdown,
    get_category_breakdown,
    get_vehicle_type_breakdown,
    get_top_plates,
    get_frequent_denials,
    get_gate_activity,
    get_ocr_confidence_distribution
)
from live_detection import cam_state, ocr_worker, generate_mjpeg
from manual_detection import process_image
from utils import assign_slot, free_slot, get_all_slots, fuzzy_lookup
from night_mode import night_state
from speed_estimator import speed_estimator
from multi_camera import multi_cam
from database import (
    init_db, register_vehicle, get_all_vehicles, delete_vehicle,
    get_logs, get_alerts, resolve_alert, get_stats, add_alert, log_entry,
    cache_invalidate, repair_db, check_vehicle
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder='templates', static_folder='static')


# ── UI ─────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception:
        path = os.path.join(BASE_DIR, 'templates', 'index.html')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return (
            f"<h2>templates/index.html not found</h2>"
            f"<p>BASE_DIR: <code>{BASE_DIR}</code></p>"
            f"<p>Files: <code>{os.listdir(BASE_DIR)}</code></p>"
        ), 404


# ── LIVE CAMERA ────────────────────────────────────────────────────────────────
@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    d     = request.get_json() or {}
    index = int(d.get('index', 0))
    cam_state.start(index)   # always returns True now — opens in background thread
    return jsonify({'ok': True, 'mode': cam_state.mode,
                    'message': f'Camera {index} opening — live feed appears in ~3 seconds'})

@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    cam_state.stop()
    return jsonify({'ok': True})

@app.route('/api/camera/status')
def camera_status():
    r = cam_state.current_result
    now = time.time()
    return jsonify({
        'mode':        cam_state.mode,
        'running':     cam_state.running,
        'result':      r if r and (now-cam_state.last_det_time)<15 else None,
        'age_seconds': round(now-cam_state.last_det_time,1) if cam_state.last_det_time else None,
        'ocr_busy':    ocr_worker.is_busy,
    })

@app.route('/api/camera/last')
def camera_last():
    r = cam_state.current_result
    now = time.time()
    if r:
        return jsonify({**r, 'fresh': True, 'age_seconds': round(now-cam_state.last_det_time,1)})
    return jsonify({'fresh': False, 'mode': cam_state.mode, 'plate': None})

@app.route('/api/camera/raw_reading')
def camera_raw_reading():
    raw = ocr_worker.raw_reading
    cur = cam_state.current_result
    now = time.time()
    return jsonify({
        'raw_reading':  raw,
        'final_result': cur if cur and (now-cam_state.last_det_time)<15 else None,
        'timestamp':    datetime.now().strftime('%H:%M:%S'),
    })

# Camera finder helper
@app.route('/api/camera/find', methods=['GET'])
def camera_find():
    """Test which camera indices are available on this machine."""
    import cv2, platform
    available = []
    for i in range(6):
        try:
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                available.append({'index': i, 'works': ret})
                cap.release()
        except Exception:
            pass
    return jsonify({'available_cameras': available,
                    'platform': platform.system(),
                    'tip': 'Use the index where works=true'})


# ── MANUAL DETECTION ───────────────────────────────────────────────────────────
@app.route('/api/manual_capture', methods=['POST'])
def manual_capture():
    d         = request.get_json() or {}
    image_b64 = d.get('image', '')
    if not image_b64:
        return jsonify({'error': 'No image provided', 'success': False}), 400

    result_box = [None, None]
    exc_box    = [None]

    def _run():
        try:
            result_box[0], result_box[1] = process_image(image_b64)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        return jsonify({'success': False, 'error': 'Detection timed out (30s).'}), 200
    if exc_box[0]:
        return jsonify({'success': False, 'error': str(exc_box[0])}), 500
    return jsonify(result_box[0]), result_box[1]


# ── VEHICLES ───────────────────────────────────────────────────────────────────
@app.route('/api/vehicles')
def api_vehicles():
    return jsonify(get_all_vehicles())

@app.route('/api/register_vehicle', methods=['POST'])
def api_register_vehicle():
    d     = request.get_json() or {}
    plate = d.get('plate','').upper().strip()
    if not plate:
        return jsonify({'error': 'plate required'}), 400
    return jsonify(register_vehicle(
        plate=plate, owner=d.get('owner','Unknown'),
        vtype=d.get('vtype','Car'), flat=d.get('flat','—'),
        category=d.get('category','Resident'), image_b64=d.get('image_b64','')
    ))

@app.route('/api/vehicles/<int:vid>', methods=['DELETE'])
def api_delete_vehicle(vid):
    delete_vehicle(vid)
    return jsonify({'ok': True})


# ── ALLOW ENTRY (Manual Override) ──────────────────────────────────────────────
@app.route('/api/allow_entry', methods=['POST'])
def allow_entry():
    d        = request.get_json() or {}
    plate    = d.get('plate','').upper().strip()
    if not plate:
        return jsonify({'error': 'plate required'}), 400

    owner    = d.get('owner','Unknown')
    vtype    = d.get('vtype','Car')
    flat     = d.get('flat','—')
    category = d.get('category','Resident')
    crop_b64 = d.get('crop_b64','')

    info = check_vehicle(plate)
    if not info:
        register_vehicle(plate, owner, vtype, flat, category)
        info = {'plate':plate,'owner':owner,'vtype':vtype,'flat':flat,'category':category}

    slot = assign_slot(plate)
    log_entry(plate=plate, action='ENTRY', confidence=99, ocr_confidence=99,
              slot=slot or '—', ocr_raw=plate, vtype=info.get('vtype',vtype),
              plate_crop_b64=crop_b64)

    return jsonify({
        'ok': True, 'plate': plate, 'slot': slot or '—',
        'owner': info.get('owner',owner), 'vtype': info.get('vtype',vtype),
        'flat': info.get('flat',flat), 'category': info.get('category',category),
        'message': f"Entry allowed for {plate}. Slot {slot or 'N/A'} assigned.",
        'timestamp': datetime.now().strftime('%H:%M:%S'),
    })


# ── EXIT ───────────────────────────────────────────────────────────────────────
@app.route('/api/exit_vehicle', methods=['POST'])
def api_exit_vehicle():
    d     = request.get_json() or {}
    plate = d.get('plate','').upper().strip()
    if not plate: return jsonify({'error':'plate required'}),400
    slot  = free_slot(plate)
    log_entry(plate=plate, action='EXIT', slot=slot or '—')
    return jsonify({'ok':True,'plate':plate,'slot':slot,'message':f'{plate} exited. Slot {slot} freed.'})


# ── LOGS / SLOTS / ALERTS / STATS ──────────────────────────────────────────────
@app.route('/api/logs')
def api_logs():
    limit = request.args.get('limit',50,type=int)
    return jsonify(get_logs(limit))

@app.route('/api/slots')
def api_slots():
    slots = get_all_slots()
    return jsonify([{'id':k,'plate':v,'status':'occupied' if v else 'free'} for k,v in sorted(slots.items())])

@app.route('/api/alerts')
def api_alerts():
    return jsonify(get_alerts())

@app.route('/api/alerts/<int:aid>/resolve', methods=['POST'])
def api_resolve(aid):
    resolve_alert(aid)
    return jsonify({'ok': True})

@app.route('/api/stats')
def api_stats():
    stats = get_stats()
    slots = get_all_slots()
    from yolo_model import is_loaded as yl
    from ocr_module  import is_loaded as ol
    return jsonify({
        **stats,
        'slots_free':     sum(1 for v in slots.values() if v is None),
        'slots_occupied': sum(1 for v in slots.values() if v is not None),
        'slots_total':    len(slots),
        'camera_mode':    cam_state.mode,
        'yolo_loaded':    yl(),
        'ocr_loaded':     ol(),
    })


# ── ANALYTICS ──────────────────────────────────────────────────────────────────
@app.route('/api/analytics')
def analytics_api():
    try:
        return {
            "overview": get_overview(),
            "hourly": get_hourly_heatmap(),
            "daily": get_daily_trend(),
            "prediction": get_peak_hour_prediction(),
            "ocr_trend": get_ocr_accuracy_trend(),
            "anomalies": detect_anomalies(),
            "speed": get_speed_analytics(),
            "colors": get_color_stats(),
            "day_night": get_night_day_breakdown(),
            "actions": get_action_breakdown(),
            "categories": get_category_breakdown(),
            "vehicle_types": get_vehicle_type_breakdown(),
            "top_plates": get_top_plates(),
            "frequent_denials": get_frequent_denials(),
            "gates": get_gate_activity(),
            "ocr_distribution": get_ocr_confidence_distribution()
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/api/anomalies')
def api_anomalies():
    from analytics import detect_anomalies
    return jsonify(detect_anomalies())


# ── NIGHT MODE ─────────────────────────────────────────────────────────────────
@app.route('/api/night_mode', methods=['GET'])
def get_night_mode():
    return jsonify(night_state.status())

@app.route('/api/night_mode', methods=['POST'])
def set_night_mode():
    d    = request.get_json() or {}
    mode = d.get('mode','AUTO').upper()
    if mode not in ('AUTO','ON','OFF'):
        return jsonify({'error':'mode must be AUTO, ON or OFF'}),400
    night_state.set_mode(mode)
    return jsonify({'ok':True,**night_state.status()})


# ── MULTI-CAMERA ───────────────────────────────────────────────────────────────
@app.route('/video_feed/<int:cam_id>')
def video_feed_multi(cam_id):
    return Response(multi_cam.get_stream(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras')
def api_cameras():
    return jsonify(multi_cam.all_status())

@app.route('/api/cameras/<int:cam_id>/start', methods=['POST'])
def api_cam_start(cam_id):
    d   = request.get_json() or {}
    idx = d.get('index', cam_id)
    ok  = multi_cam.start_camera(cam_id, index=idx, rtsp_url=d.get('rtsp_url'))
    if d.get('name'): multi_cam.set_camera_name(cam_id, d['name'])
    return jsonify({'ok': ok, 'cam_id': cam_id})

@app.route('/api/cameras/<int:cam_id>/stop', methods=['POST'])
def api_cam_stop(cam_id):
    multi_cam.stop_camera(cam_id)
    return jsonify({'ok': True})

@app.route('/api/cameras/<int:cam_id>/last')
def api_cam_last(cam_id):
    return jsonify(multi_cam.get_last_result(cam_id) or {'fresh': False, 'cam_id': cam_id})


# ── SPEED ──────────────────────────────────────────────────────────────────────
@app.route('/api/speed/status')
def api_speed_status():
    return jsonify(speed_estimator.status())

@app.route('/api/speed/history')
def api_speed_history():
    return jsonify(speed_estimator.get_history())

@app.route('/api/speed/config', methods=['POST'])
def api_speed_config():
    import speed_estimator as se
    d = request.get_json() or {}
    if 'distance_m'  in d: se.ZONE_DISTANCE_M = float(d['distance_m'])
    if 'limit_kmh'   in d: se.OVERSPEED_KMH   = float(d['limit_kmh'])
    return jsonify({'ok':True,'distance_m':se.ZONE_DISTANCE_M,'limit_kmh':se.OVERSPEED_KMH})


# ── DB TOOLS ───────────────────────────────────────────────────────────────────
@app.route('/api/db/fix', methods=['POST'])
def api_db_fix():
    return jsonify({'ok':True,'fixes':repair_db()})

@app.route('/api/db/cleanup', methods=['POST'])
def api_db_cleanup():
    return jsonify({'ok':True,'fixes':repair_db()})

@app.route('/api/db/test_match', methods=['POST'])
def api_test_match():
    d   = request.get_json() or {}
    ocr = d.get('ocr','').upper().strip()
    if not ocr: return jsonify({'error':'ocr required'}),400
    info,matched,dist = fuzzy_lookup(ocr)
    return jsonify({'input':ocr,'matched_plate':matched,'distance':dist,'info':info,
                    'authorized':bool(info and info.get('category')!='Blacklisted'),
                    'verdict':'MATCH' if matched and dist<=3 else 'NO MATCH'})


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from yolo_model import load_model
    from ocr_module  import get_reader

    print('='*65)
    print('  🔥 ANPR SYSTEM v2 — YOLOv8 + EasyOCR')
    print('='*65)

    init_db()
    ensure_analytics_columns()

    try:    load_model(); yolo_ok = True
    except Exception as e: yolo_ok = False; print(f'[✗] YOLO: {e}')

    try:    get_reader();  ocr_ok  = True
    except Exception as e: ocr_ok  = False; print(f'[✗] OCR: {e}')

    print(f"[{'✓' if yolo_ok else '✗'}] YOLOv8   {'loaded' if yolo_ok else 'NOT loaded'}")
    print(f"[{'✓' if ocr_ok  else '✗'}] EasyOCR  {'loaded' if ocr_ok  else 'NOT loaded'}")
    print()
    print('  🌐 Open: http://localhost:5001')
    print('='*65)
    print('  CAMERA TIPS:')
    print('  • If camera shows black → visit http://localhost:5000/api/camera/find')
    print('  • This shows which camera indices work on your machine')
    print('  • Try Cam 0, Cam 1, or Cam 2 in the UI dropdown')
    print('='*65)
    print('  ROUTES:')
    print('  GET  /video_feed              — MJPEG live stream')
    print('  POST /api/camera/start        — Start camera {"index": 0}')
    print('  GET  /api/camera/find         — Find available camera indices')
    print('  POST /api/manual_capture      — Upload image & detect')
    print('  POST /api/allow_entry         — Manual override')
    print('  POST /api/exit_vehicle        — Exit & free slot')
    print('  GET  /api/analytics           — Full analytics data')
    print('='*65)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)