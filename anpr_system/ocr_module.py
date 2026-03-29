"""
ocr_module.py — OCR Processing via EasyOCR
Handles plate text extraction with multi-variant fallback strategy.
"""

import cv2
import re
import numpy as np

# ── EasyOCR Reader (loaded once) ──────────────────────────────────────────────
_reader = None


def get_reader():
    global _reader
    if _reader is not None:
        return _reader
    import easyocr
    try:
        print("[OCR] Loading EasyOCR with quantization (CPU-optimised)...")
        _reader = easyocr.Reader(['en'], gpu=False, quantize=True)
        print("[✓] EasyOCR loaded (quantized)")
    except Exception:
        _reader = easyocr.Reader(['en'], gpu=False)
        print("[✓] EasyOCR loaded (standard CPU)")
    return _reader


def is_loaded() -> bool:
    return _reader is not None


# ── Text Cleaning ──────────────────────────────────────────────────────────────
def clean_plate_text(raw: str) -> str:
    """Remove non-alphanumeric chars and uppercase."""
    return re.sub(r'[^A-Z0-9]', '', raw.upper())


# ── Main OCR Entry Point ───────────────────────────────────────────────────────
def read_plate(roi, fast: bool = False) -> tuple[str, float, str]:
    """
    Run OCR on a cropped plate ROI.

    Args:
        roi:  BGR or grayscale numpy array (the plate crop)
        fast: If True, use greedy decoder only (for live stream speed)

    Returns:
        (best_text, confidence_pct, raw_all)
        - best_text:     cleaned plate string
        - confidence_pct: 0–100 float
        - raw_all:       space-joined raw OCR tokens (for fuzzy matching)
    """
    reader = get_reader()

    if roi is None or roi.size == 0:
        return '', 0, ''

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

        if gray.shape[0] < 5 or gray.shape[1] < 10:
            return '', 0, ''

        best_text, best_conf, best_raw = '', 0.0, ''

        # ── Method A: Fixed 300×100 resize (proven working path) ─────────────
        fixed = cv2.resize(gray, (300, 100), interpolation=cv2.INTER_CUBIC)
        for variant in [fixed, cv2.bitwise_not(fixed)]:
            try:
                results = reader.readtext(variant, detail=1, paragraph=False)
                for _, txt, conf in results:
                    clean = clean_plate_text(txt)
                    if len(clean) >= 4 and conf > best_conf:
                        best_conf = conf
                        best_text = clean
                        best_raw  = clean
            except Exception:
                pass

        if best_text:
            print(f"[OCR-A] 300x100: '{best_text}' conf={int(best_conf*100)}%")

        # ── Method B: CLAHE + scale-up + denoise (enhanced path) ─────────────
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        h, w     = enhanced.shape[:2]

        if h < 100:
            scale  = max(3.0, 100 / h)
            interp = cv2.INTER_LANCZOS4 if h < 30 else cv2.INTER_CUBIC
            enhanced = cv2.resize(
                enhanced, (int(w * scale), int(h * scale)), interpolation=interp
            )
            enhanced = cv2.fastNlMeansDenoising(
                enhanced, h=8, templateWindowSize=7, searchWindowSize=21
            )

        # Choose orientation: dark-on-light vs light-on-dark
        mean_bright = np.mean(enhanced)
        variants = (
            [cv2.bitwise_not(enhanced), enhanced]
            if mean_bright < 127
            else [enhanced, cv2.bitwise_not(enhanced)]
        )
        # Add sharpened variant
        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        variants.append(cv2.filter2D(enhanced, -1, kernel_sharp))

        decoder = 'greedy'  # always greedy — fastest, still accurate for plates
        for variant in variants:
            try:
                results = reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    batch_size=1,
                    width_ths=0.7,
                    decoder=decoder,
                )
                texts, confs = [], []
                for _, txt, conf in results:
                    clean = clean_plate_text(txt)
                    if len(clean) >= 2:
                        texts.append(clean)
                        confs.append(conf)

                if texts:
                    idx = int(np.argmax(confs))
                    if confs[idx] > best_conf:
                        best_conf = confs[idx]
                        best_text = texts[idx]
                        best_raw  = ' '.join(texts)
            except Exception:
                continue

        conf_pct = round(best_conf * 100, 1)
        if best_text:
            print(f"[OCR] Final: '{best_text}' conf={conf_pct}% raw='{best_raw}'")
        else:
            print(f"[OCR] No text found in ROI {gray.shape}")

        return best_text, conf_pct, best_raw

    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return '', 0, ''
