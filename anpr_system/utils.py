"""
utils.py — Helper Functions
Fuzzy plate matching, text preprocessing, parking slot management.
"""

import re
import time
import threading
from itertools import combinations as _combs
from typing import Optional, Tuple

from database import check_vehicle, check_vehicle_cached, cache_warm, get_conn

# ── OCR Common Confusions ──────────────────────────────────────────────────────
OCR_CONFUSIONS = {
    'Z': '2', '2': 'Z',
    'O': '0', '0': 'O',
    'I': '1', '1': 'I',
    'B': '8', '8': 'B',
    'S': '5', '5': 'S',
    'G': '6', '6': 'G',
    'Q': '0', 'D': '0',
    'N': 'M', 'M': 'N',
    '3': '8', '8': '3',
    'A': '4', '4': 'A',
}


# ── Text Cleaning ──────────────────────────────────────────────────────────────
def clean_plate(raw: str) -> str:
    """Strip all non-alphanumeric chars and uppercase."""
    return re.sub(r'[^A-Z0-9]', '', raw.upper())


def preprocess_roi(roi):
    """
    Preprocessing pipeline for OCR:
    - Grayscale
    - CLAHE
    - Resize to minimum height if too small
    Returns a grayscale numpy array.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    h, w = enhanced.shape[:2]
    if h < 60:
        scale = max(3.0, 60 / h)
        enhanced = cv2.resize(
            enhanced,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LANCZOS4 if h < 30 else cv2.INTER_CUBIC
        )
    return enhanced


# ── Levenshtein Distance ───────────────────────────────────────────────────────
def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
        prev = curr
    return prev[-1]


# ── Fuzzy Plate Lookup ─────────────────────────────────────────────────────────
def fuzzy_lookup(ocr_raw: str) -> Tuple[Optional[dict], Optional[str], int]:
    """
    Smart fuzzy lookup with OCR confusion handling.
    Tries exact match first, then edit-distance with confusion substitutions.

    Returns:
        (vehicle_info | None, matched_plate | None, edit_distance)
    """
    if not ocr_raw:
        return None, None, 999

    conn = get_conn()
    all_plates = [r[0] for r in conn.execute('SELECT plate FROM vehicles').fetchall()]
    conn.close()

    if not all_plates:
        return None, None, 999

    clean = clean_plate(ocr_raw)

    # Exact match — always direct DB (new plates might not be cached yet)
    info = check_vehicle(clean)
    if info:
        cache_warm(clean, info)
        print(f"[FUZZY] Exact match: {clean}")
        return info, clean, 0

    # Build candidate strings from OCR tokens
    tokens = re.findall(r'[A-Z0-9]{2,}', ocr_raw.upper())
    if not tokens:
        return None, None, 999

    candidates = set()
    for tok in tokens:
        candidates.add(tok)
        # Sliding window substrings
        for s in range(min(4, max(0, len(tok)-7))):
            if tok[s].isalpha() and s+1 < len(tok) and tok[s+1].isalpha():
                for e in range(s+7, min(s+13, len(tok)+1)):
                    candidates.add(tok[s:e])

    # Join adjacent tokens
    for i in range(len(tokens)-1):
        joined = tokens[i] + tokens[i+1]
        candidates.add(joined)
        for s in range(min(3, max(0, len(joined)-8))):
            if joined[s].isalpha() and s+1 < len(joined) and joined[s+1].isalpha():
                candidates.add(joined[s:s+12])

    # Confusion substitutions
    extra = set()
    for cand in list(candidates)[:30]:
        for i, ch in enumerate(cand):
            if ch in OCR_CONFUSIONS:
                extra.add(cand[:i] + OCR_CONFUSIONS[ch] + cand[i+1:])
    candidates.update(extra)

    # Find best match by Levenshtein
    best_plate, best_dist, best_cand = None, 999, None
    for cand in candidates:
        if len(cand) < 5:
            continue
        for reg in all_plates:
            d = levenshtein(cand, reg)
            if d < best_dist:
                best_dist, best_plate, best_cand = d, reg, cand

    # Removal variants — with 1.5s time budget
    if best_dist > 1:
        t0   = time.time()
        top5 = sorted(candidates, key=lambda c: min(levenshtein(c, p) for p in all_plates))[:6]
        for cand in top5:
            if time.time() - t0 > 1.5:
                break
            for n in range(1, 4):
                for pos in _combs(range(len(cand)), n):
                    new = ''.join(ch for i, ch in enumerate(cand) if i not in pos)
                    if len(new) < 7:
                        continue
                    for reg in all_plates:
                        d = levenshtein(new, reg)
                        if d < best_dist:
                            best_dist, best_plate, best_cand = d, reg, new

    if best_dist <= 3:
        info = check_vehicle(best_plate)
        if info:
            cache_warm(best_plate, info)
        print(f"[FUZZY] {clean[:20]} → {best_cand} → {best_plate} (dist={best_dist})")
        return info, best_plate, best_dist

    print(f"[FUZZY] No match: {clean[:20]} → closest={best_plate} dist={best_dist}")
    return None, None, best_dist


# ── Parking Slot Manager ───────────────────────────────────────────────────────
_SLOTS: dict = {
    f'{zone}-{str(i).zfill(2)}': None
    for zone in ['A', 'B', 'C']
    for i in range(1, 6)
}
_slots_lock = threading.Lock()


def get_free_slot() -> Optional[str]:
    with _slots_lock:
        return next((s for s, p in _SLOTS.items() if p is None), None)


def assign_slot(plate: str) -> Optional[str]:
    with _slots_lock:
        # If already parked, return existing slot
        for s, p in _SLOTS.items():
            if p == plate:
                return s
        # Assign first free
        slot = next((s for s, p in _SLOTS.items() if p is None), None)
        if slot:
            _SLOTS[slot] = plate
            print(f"[SLOT] Assigned {slot} → {plate}")
        return slot


def free_slot(plate: str) -> Optional[str]:
    with _slots_lock:
        for s, p in _SLOTS.items():
            if p == plate:
                _SLOTS[s] = None
                print(f"[SLOT] Freed {s} (was {plate})")
                return s
    return None


def get_all_slots() -> dict:
    with _slots_lock:
        return dict(_SLOTS)


# ── Access Decision ────────────────────────────────────────────────────────────
def make_access_decision(plate: str, ocr_conf: float,
                          match_dist: int = 0) -> dict:
    """
    Unified access control logic.
    Returns full decision dict used by both live and manual detection.
    """
    from database import check_vehicle

    info    = check_vehicle(plate)
    allowed = bool(info and info['category'] != 'Blacklisted')
    action  = 'ENTRY' if allowed else 'DENIED'
    slot    = assign_slot(plate) if allowed else None

    fuzzy_penalty = match_dist * 5 if match_dist < 999 else 30
    conf_total    = max(10, round(min(99, ocr_conf * 0.7 + (40 if allowed else 20) - fuzzy_penalty), 1))

    return {
        'plate':      plate,
        'owner':      info['owner']    if info else 'Unknown',
        'vtype':      info['vtype']    if info else 'Unknown',
        'flat':       info['flat']     if info else '—',
        'category':   info['category'] if info else 'Unregistered',
        'authorized': allowed,
        'action':     action,
        'status':     'ALLOWED' if allowed else 'DENIED',
        'slot':       slot or '—',
        'conf':       conf_total,
        'ocr_conf':   round(ocr_conf, 1),
        'match_dist': match_dist,
        'registered': bool(info),
    }
