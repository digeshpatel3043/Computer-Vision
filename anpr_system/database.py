"""
database.py — All SQLite Database Operations
Tables: vehicles, logs, alerts
"""

import sqlite3
import threading
from datetime import datetime
from typing import Optional, Dict, List

# ── Config ─────────────────────────────────────────────────────────────────────
import os
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anpr.db')

# ── In-memory plate cache (avoids repeated DB hits on hot path) ───────────────
_cache: Dict[str, Optional[dict]] = {}
_cache_lock = threading.Lock()


# ── Connection Helper ──────────────────────────────────────────────────────────
def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


# ── Cache Helpers ──────────────────────────────────────────────────────────────
def cache_warm(plate: str, info: dict):
    """Store a registration in cache immediately after INSERT/UPDATE."""
    with _cache_lock:
        _cache[plate] = info


def cache_invalidate(plate: str = None):
    """Clear one plate or the entire cache (e.g. after delete)."""
    with _cache_lock:
        if plate:
            _cache.pop(plate, None)
        else:
            _cache.clear()


# ── Init ───────────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables and seed sample data."""
    conn = get_conn()
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS vehicles (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        plate           TEXT UNIQUE NOT NULL,
        owner           TEXT DEFAULT 'Unknown',
        vtype           TEXT DEFAULT 'Car',
        flat            TEXT DEFAULT '—',
        category        TEXT DEFAULT 'Resident',
        image_b64       TEXT DEFAULT '',
        registered_at   TEXT
    );

    CREATE TABLE IF NOT EXISTS logs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        plate           TEXT,
        ocr_raw         TEXT,
        vtype           TEXT,
        action          TEXT,
        gate            TEXT DEFAULT 'Gate A',
        confidence      REAL DEFAULT 0,
        ocr_confidence  REAL DEFAULT 0,
        slot            TEXT DEFAULT '—',
        image_b64       TEXT DEFAULT '',
        plate_crop_b64  TEXT DEFAULT '',
        timestamp       TEXT
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        type        TEXT,
        plate       TEXT,
        message     TEXT,
        severity    TEXT DEFAULT 'high',
        resolved    INTEGER DEFAULT 0,
        timestamp   TEXT
    );
    ''')

    # Seed demo vehicles
    seed = [
        ('DL8CAU6883', 'Arjun Mehta',   'SUV',       'A-101', 'Resident'),
        ('DL9CAU5573', 'Priya Sharma',  'Van',        'A-102', 'Resident'),
        ('DL12CN8660', 'Rohit Gupta',   'Car',        'B-201', 'Resident'),
        ('DL4CAM9066', 'Sneha Patel',   'Hatchback',  'B-202', 'Resident'),
        ('DL12CK8643', 'Vikram Singh',  'SUV',        'C-301', 'Resident'),
        ('DL10CH1252', 'Kavita Nair',   'Car',        'C-302', 'Resident'),
        ('DL2CH8634',  'Suresh Iyer',   'Sedan',      'D-401', 'Resident'),
        ('DL1CS8739',  'Test Vehicle',  'SUV',        'A-103', 'Resident'),
        ('MH12AB1234', 'Blacklisted X', 'Car',        'N/A',   'Blacklisted'),
    ]
    for plate, owner, vtype, flat, category in seed:
        try:
            conn.execute(
                'INSERT INTO vehicles(plate,owner,vtype,flat,category,registered_at) VALUES(?,?,?,?,?,?)',
                (plate, owner, vtype, flat, category, datetime.now().isoformat())
            )
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    print("[✓] Database initialised")


# ── Vehicle CRUD ───────────────────────────────────────────────────────────────
def check_vehicle(plate: str) -> Optional[dict]:
    """
    Direct DB lookup (no cache).
    Use check_vehicle_cached() in performance-critical paths.
    """
    conn = get_conn()
    row = conn.execute(
        'SELECT plate,owner,vtype,flat,category FROM vehicles WHERE plate=?', (plate,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {'plate': row[0], 'owner': row[1], 'vtype': row[2],
            'flat': row[3], 'category': row[4]}


def check_vehicle_cached(plate: str) -> Optional[dict]:
    """Cache-first lookup. Never caches None (avoids stale miss after register)."""
    with _cache_lock:
        if plate in _cache:
            return _cache[plate]
    result = check_vehicle(plate)
    if result is not None:
        with _cache_lock:
            _cache[plate] = result
    return result


def register_vehicle(plate: str, owner: str = 'Unknown', vtype: str = 'Car',
                     flat: str = '—', category: str = 'Resident',
                     image_b64: str = '') -> dict:
    """Insert or update a vehicle record. Returns the vehicle dict."""
    conn = get_conn()
    try:
        conn.execute(
            'INSERT INTO vehicles(plate,owner,vtype,flat,category,image_b64,registered_at) VALUES(?,?,?,?,?,?,?)',
            (plate, owner, vtype, flat, category, image_b64, datetime.now().isoformat())
        )
        msg = f'{plate} registered.'
    except sqlite3.IntegrityError:
        conn.execute(
            'UPDATE vehicles SET owner=?,vtype=?,flat=?,category=?,image_b64=? WHERE plate=?',
            (owner, vtype, flat, category, image_b64, plate)
        )
        msg = f'{plate} updated.'
    conn.commit()
    conn.close()

    info = {'plate': plate, 'owner': owner, 'vtype': vtype, 'flat': flat, 'category': category}
    cache_invalidate(plate)
    cache_warm(plate, info)
    return {'ok': True, 'message': msg, 'plate': plate}


def delete_vehicle(vehicle_id: int) -> bool:
    conn = get_conn()
    conn.execute('DELETE FROM vehicles WHERE id=?', (vehicle_id,))
    conn.commit()
    conn.close()
    cache_invalidate()
    return True


def get_all_vehicles() -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        'SELECT id,plate,owner,vtype,flat,category,image_b64,registered_at FROM vehicles'
    ).fetchall()
    conn.close()
    return [{'id': r[0], 'plate': r[1], 'owner': r[2], 'vtype': r[3],
             'flat': r[4], 'category': r[5], 'image_b64': r[6],
             'registered_at': r[7]} for r in rows]


# ── Access Control ─────────────────────────────────────────────────────────────
def evaluate_access(plate: str) -> dict:
    """
    Core access logic.
    Returns: {plate, owner, status ('ALLOWED'|'DENIED'), category, vtype, flat}
    """
    info = check_vehicle(plate)

    if info is None:
        return {
            'plate':    plate,
            'owner':    'Unknown',
            'status':   'DENIED',
            'reason':   'not_registered',
            'category': 'Unregistered',
            'vtype':    'Unknown',
            'flat':     '—',
        }

    if info['category'] == 'Blacklisted':
        return {**info, 'status': 'DENIED', 'reason': 'blacklisted'}

    return {**info, 'status': 'ALLOWED', 'reason': 'registered'}


# ── Log ────────────────────────────────────────────────────────────────────────
def log_entry(plate: str, action: str, gate: str = 'Gate A',
              confidence: float = 0, ocr_confidence: float = 0,
              slot: str = '—', ocr_raw: str = '', vtype: str = '',
              image_b64: str = '', plate_crop_b64: str = '') -> int:
    """Insert a detection event into logs. Returns the new row id."""
    conn = get_conn()
    cur = conn.execute(
        '''INSERT INTO logs
           (plate,ocr_raw,vtype,action,gate,confidence,ocr_confidence,
            slot,image_b64,plate_crop_b64,timestamp)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
        (plate, ocr_raw, vtype, action, gate,
         confidence, ocr_confidence, slot,
         image_b64, plate_crop_b64,
         datetime.now().strftime('%H:%M:%S'))
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_logs(limit: int = 50) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        '''SELECT id,plate,ocr_raw,vtype,action,gate,confidence,ocr_confidence,
                  slot,image_b64,plate_crop_b64,timestamp
           FROM logs ORDER BY id DESC LIMIT ?''', (limit,)
    ).fetchall()
    conn.close()
    return [{'id': r[0], 'plate': r[1], 'ocr_raw': r[2], 'vtype': r[3],
             'action': r[4], 'gate': r[5], 'confidence': r[6],
             'ocr_confidence': r[7], 'slot': r[8], 'image_b64': r[9],
             'plate_crop_b64': r[10], 'timestamp': r[11]} for r in rows]


# ── Alerts ─────────────────────────────────────────────────────────────────────
def add_alert(alert_type: str, plate: str, message: str, severity: str = 'high') -> int:
    conn = get_conn()
    cur  = conn.execute(
        'INSERT INTO alerts(type,plate,message,severity,timestamp) VALUES(?,?,?,?,?)',
        (alert_type, plate, message, severity, datetime.now().strftime('%H:%M:%S'))
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_alerts(limit: int = 30) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        'SELECT id,type,plate,message,severity,resolved,timestamp FROM alerts ORDER BY id DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    return [{'id': r[0], 'type': r[1], 'plate': r[2], 'message': r[3],
             'severity': r[4], 'resolved': bool(r[5]), 'timestamp': r[6]} for r in rows]


def resolve_alert(alert_id: int) -> bool:
    conn = get_conn()
    conn.execute('UPDATE alerts SET resolved=1 WHERE id=?', (alert_id,))
    conn.commit()
    conn.close()
    return True


# ── Stats ──────────────────────────────────────────────────────────────────────
def get_stats() -> dict:
    conn = get_conn()
    total   = conn.execute('SELECT COUNT(*) FROM logs').fetchone()[0]
    entries = conn.execute("SELECT COUNT(*) FROM logs WHERE action='ENTRY'").fetchone()[0]
    denied  = conn.execute("SELECT COUNT(*) FROM logs WHERE action='DENIED'").fetchone()[0]
    exits   = conn.execute("SELECT COUNT(*) FROM logs WHERE action='EXIT'").fetchone()[0]
    alerts  = conn.execute('SELECT COUNT(*) FROM alerts WHERE resolved=0').fetchone()[0]
    reg     = conn.execute('SELECT COUNT(*) FROM vehicles').fetchone()[0]
    conn.close()
    return {
        'total_events':  total,
        'entries':       entries,
        'exits':         exits,
        'denied':        denied,
        'active_alerts': alerts,
        'registered':    reg,
    }


# ── DB Repair ──────────────────────────────────────────────────────────────────
def repair_db() -> List[str]:
    """Fix NULLs, duplicates, and invalid plates. Returns list of fixes applied."""
    fixes = []
    conn  = get_conn()

    # Fix NULL columns
    for col, default in [('owner','Unknown'),('vtype','Car'),('flat','—'),('category','Resident')]:
        n = conn.execute(
            f"UPDATE vehicles SET {col}=? WHERE {col} IS NULL OR {col}=''", (default,)
        ).rowcount
        if n:
            fixes.append(f"Fixed {n} NULL {col}")

    # Remove invalid plates
    rows    = conn.execute('SELECT id, plate FROM vehicles').fetchall()
    removed = []
    for vid, plate in rows:
        if len(plate) < 5 or not any(c.isdigit() for c in plate) or not any(c.isalpha() for c in plate):
            conn.execute('DELETE FROM vehicles WHERE id=?', (vid,))
            removed.append(plate)
    if removed:
        fixes.append(f"Removed invalid: {removed}")

    # Deduplicate
    dupes = conn.execute(
        'SELECT plate, COUNT(*) FROM vehicles GROUP BY plate HAVING COUNT(*) > 1'
    ).fetchall()
    for plate, cnt in dupes:
        ids = [r[0] for r in conn.execute(
            'SELECT id FROM vehicles WHERE plate=? ORDER BY id DESC', (plate,)
        ).fetchall()]
        for old_id in ids[1:]:
            conn.execute('DELETE FROM vehicles WHERE id=?', (old_id,))
        fixes.append(f"Deduped {cnt-1} copies of {plate}")

    conn.commit()
    conn.close()
    cache_invalidate()
    return fixes
