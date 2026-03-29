# ANPR System — YOLOv8 + EasyOCR
**Smart Vehicle Entry System | Modular Flask Architecture**

---

## 📁 Project Structure

```
anpr_system/
├── app.py               ← Main Flask app (all API routes)
├── yolo_model.py        ← YOLOv8 model loader (singleton)
├── ocr_module.py        ← EasyOCR processing
├── live_detection.py    ← Real-time camera detection (speed-optimised)
├── manual_detection.py  ← Image upload detection (high-accuracy)
├── database.py          ← All SQLite DB operations
├── utils.py             ← Fuzzy matching, parking slots, access control
├── templates/
│   └── index.html       ← Complete Web UI
├── requirements.txt
├── best.pt              ← YOUR trained YOLOv8 model (place here)
└── anpr.db              ← Auto-created on first run
```

---

## ⚡ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model
Copy your trained `best.pt` into the project root:
```
anpr_system/best.pt
```
The system also checks:
- `runs/detect/train2/weights/best.pt`
- `yolov8_plate.pt`

### 3. Run
```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/video_feed` | MJPEG live camera stream |
| POST | `/api/camera/start` | Start camera `{"index": 0}` |
| POST | `/api/camera/stop` | Stop camera |
| GET | `/api/camera/last` | Latest detection result |
| POST | `/api/manual_capture` | Upload image & detect `{"image": "base64..."}` |
| POST | `/api/register_vehicle` | Register vehicle |
| POST | `/api/allow_entry` | Manual override — allow denied vehicle |
| POST | `/api/exit_vehicle` | Exit & free parking slot |
| GET | `/api/logs` | Detection log |
| GET | `/api/vehicles` | All registered vehicles |
| GET | `/api/slots` | Parking slot status |
| GET | `/api/stats` | System statistics |
| GET | `/api/alerts` | Security alerts |
| POST | `/api/db/fix` | Repair database |
| POST | `/api/db/test_match` | Test fuzzy match `{"ocr": "DL1CS8739"}` |

---

## 🔧 Key Design Decisions

### Live Detection (Speed Mode)
- `imgsz=320` — ~40% faster YOLO inference
- Frame skipping every 3rd frame — 66% fewer YOLO calls
- `decoder='greedy'` EasyOCR — ~60% faster
- Multi-frame confirmation (2+ consecutive hits before OCR)
- Async OCR worker thread — never blocks MJPEG stream
- CLAHE pre-enhancement per frame

### Manual Detection (Accuracy Mode)
- `imgsz=640` — matches training resolution
- **10-pass detection pipeline:**
  1. YOLO full (conf=0.20)
  2. YOLO full (conf=0.10)
  3. YOLO auto-cropped (removes UI sidebars)
  4. YOLO top 60%
  5. YOLO bottom 60%
  6. YOLO 2× upscale
  7. Contour (Sobel + adaptive threshold)
  8. Contour 4× upscale
  9. Contour 4× upscale bottom
  10. YOLO 4× upscale (conf=0.10)
- Smart aspect-ratio-preserving resize (max 1280px)
- 30s background thread with timeout

### Fuzzy Plate Matching
- Exact match first (always hits DB directly for new plates)
- OCR confusion table (O/0, I/1, B/8, S/5 etc.)
- Levenshtein distance ≤ 3 for match
- Token sliding-window substrings
- 1.5s time budget on expensive removal variants

### Access Control
- Registered + not Blacklisted → **ALLOWED**
- Not registered → **DENIED** (show Allow Entry button)
- Blacklisted → **DENIED**

---

## 🎯 Manual Override Flow

1. Image uploaded → plate detected as **DENIED**
2. UI shows **"Allow Entry"** button with owner/flat fields
3. Click → POST `/api/allow_entry`
4. System registers vehicle, assigns slot, logs ENTRY
5. Next detection shows **ALLOWED**

---

## 🗄️ Database Schema

**vehicles** — registered vehicles  
**logs** — all detection events (ENTRY / DENIED / EXIT)  
**alerts** — unresolved security alerts  

---

## ⚙️ CPU Optimisations (i3-friendly)

| Optimisation | Impact |
|---|---|
| `imgsz=320` live | ~40% faster YOLO |
| Frame skip N=3 | 66% fewer YOLO calls |
| `decoder='greedy'` | ~60% faster EasyOCR |
| `INTER_LINEAR` resize | ~3× faster than CUBIC |
| In-memory plate cache | Zero repeated DB reads |
| `CAP_PROP_BUFFERSIZE=1` | No stale frame lag |
| Async OCR thread | 0ms MJPEG blocking |
