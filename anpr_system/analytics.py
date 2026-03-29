"""
analytics.py — ANPR Full Analytics Engine (v2)
New features over v1:
  - Real per-day trends (log_date column, schema migration)
  - Peak hour prediction (weighted 7-day moving average)
  - OCR accuracy trend per day
  - Anomaly detection (double entry, long stay, rapid repeat, high denial, ghost plates)
  - Speed analytics from speed_estimator
  - Vehicle color breakdown
  - Night vs day detection split
  - Hourly heatmap with band labels
"""

import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from database import DB_PATH


def _conn():
    return sqlite3.connect(DB_PATH)


# ── Schema Migration ───────────────────────────────────────────────────────────
def ensure_analytics_columns():
    """Add analytics columns to logs table if missing. Safe to call on every startup."""
    conn     = _conn()
    existing = [row[1] for row in conn.execute("PRAGMA table_info(logs)").fetchall()]

    migrations = {
        'log_date':       "ALTER TABLE logs ADD COLUMN log_date TEXT DEFAULT ''",
        'log_hour':       "ALTER TABLE logs ADD COLUMN log_hour INTEGER DEFAULT -1",
        'detected_color': "ALTER TABLE logs ADD COLUMN detected_color TEXT DEFAULT ''",
        'speed_kmh':      "ALTER TABLE logs ADD COLUMN speed_kmh REAL DEFAULT 0",
    }
    for col, sql in migrations.items():
        if col not in existing:
            conn.execute(sql)
            print(f"[DB-MIGRATE] Added column: logs.{col}")

    # Back-fill log_date with today for old records
    today = datetime.now().strftime('%Y-%m-%d')
    conn.execute("UPDATE logs SET log_date=? WHERE log_date IS NULL OR log_date=''", (today,))
    conn.execute(
        "UPDATE logs SET log_hour=CAST(substr(timestamp,1,2) AS INTEGER) "
        "WHERE log_hour=-1 AND timestamp IS NOT NULL AND length(timestamp)>=2"
    )
    conn.commit()
    conn.close()
    print("[Analytics] Schema OK")


# ── Overview KPIs ──────────────────────────────────────────────────────────────
def get_overview() -> dict:
    c     = _conn()
    now   = datetime.now()
    today = now.strftime('%Y-%m-%d')
    yest  = (now - timedelta(days=1)).strftime('%Y-%m-%d')

    total   = c.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    entries = c.execute("SELECT COUNT(*) FROM logs WHERE action='ENTRY'").fetchone()[0]
    denied  = c.execute("SELECT COUNT(*) FROM logs WHERE action='DENIED'").fetchone()[0]
    exits   = c.execute("SELECT COUNT(*) FROM logs WHERE action='EXIT'").fetchone()[0]

    today_total   = c.execute("SELECT COUNT(*) FROM logs WHERE log_date=?", (today,)).fetchone()[0]
    today_entries = c.execute("SELECT COUNT(*) FROM logs WHERE action='ENTRY'  AND log_date=?", (today,)).fetchone()[0]
    today_denied  = c.execute("SELECT COUNT(*) FROM logs WHERE action='DENIED' AND log_date=?", (today,)).fetchone()[0]
    unique_today  = c.execute("SELECT COUNT(DISTINCT plate) FROM logs WHERE log_date=?", (today,)).fetchone()[0]

    registered  = c.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
    blacklisted = c.execute("SELECT COUNT(*) FROM vehicles WHERE category='Blacklisted'").fetchone()[0]
    alerts_open = c.execute("SELECT COUNT(*) FROM alerts WHERE resolved=0").fetchone()[0]

    # 7-day average
    since7    = (now - timedelta(days=7)).strftime('%Y-%m-%d')
    count7    = c.execute("SELECT COUNT(*) FROM logs WHERE log_date>=?", (since7,)).fetchone()[0]
    avg7      = round(count7 / 7, 1)

    # OCR avg confidence today
    ocr_avg = c.execute(
        "SELECT AVG(ocr_confidence) FROM logs WHERE log_date=? AND ocr_confidence>0", (today,)
    ).fetchone()[0] or 0

    c.close()
    approval = round(entries / max(entries + denied, 1) * 100, 1)

    return {
        'total': total, 'entries': entries, 'denied': denied, 'exits': exits,
        'today_total': today_total, 'today_entries': today_entries,
        'today_denied': today_denied, 'unique_today': unique_today,
        'registered': registered, 'blacklisted': blacklisted,
        'alerts_open': alerts_open, 'approval_rate': approval,
        'avg_per_day': avg7, 'ocr_avg_today': round(ocr_avg, 1),
    }


# ── Real Daily Trend ───────────────────────────────────────────────────────────
def get_daily_trend(days: int = 14) -> list:
    c   = _conn()
    now = datetime.now()
    out = []
    for i in range(days - 1, -1, -1):
        d     = now - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        e = c.execute("SELECT COUNT(*) FROM logs WHERE action='ENTRY'  AND log_date=?", (d_str,)).fetchone()[0]
        n = c.execute("SELECT COUNT(*) FROM logs WHERE action='DENIED' AND log_date=?", (d_str,)).fetchone()[0]
        x = c.execute("SELECT COUNT(*) FROM logs WHERE action='EXIT'   AND log_date=?", (d_str,)).fetchone()[0]
        out.append({'date': d.strftime('%b %d'), 'date_raw': d_str, 'entries': e, 'denied': n, 'exits': x})
    c.close()
    return out


# ── Hourly Heatmap ─────────────────────────────────────────────────────────────
def get_hourly_heatmap(days: int = 7) -> list:
    c     = _conn()
    since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    rows  = c.execute(
        "SELECT log_hour, COUNT(*) FROM logs WHERE log_date>=? AND log_hour>=0 GROUP BY log_hour",
        (since,)
    ).fetchall()
    c.close()

    hourly = {r[0]: r[1] for r in rows}
    bands  = {(6,9):'morning_peak',(10,16):'daytime',(17,20):'evening_peak',(21,23):'night'}
    result = []
    for h in range(24):
        band = 'late_night'
        for (lo, hi), b in bands.items():
            if lo <= h <= hi:
                band = b; break
        result.append({
            'hour': h, 'label': f'{h:02d}:00',
            'count': hourly.get(h, 0),
            'avg':   round(hourly.get(h, 0) / days, 1),
            'band':  band,
        })
    return result


# ── Peak Hour Prediction ───────────────────────────────────────────────────────
def get_peak_hour_prediction() -> dict:
    """Weighted moving average over last 7 days to predict busiest remaining hours today."""
    c   = _conn()
    now = datetime.now()
    weights = [1, 1, 1, 2, 2, 3, 4]  # day-7 → day-1

    hourly_by_day = defaultdict(list)
    for i in range(7, 0, -1):
        d_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
        rows  = c.execute(
            "SELECT log_hour, COUNT(*) FROM logs WHERE log_date=? AND log_hour>=0 GROUP BY log_hour",
            (d_str,)
        ).fetchall()
        by_h = {r[0]: r[1] for r in rows}
        for h in range(24):
            hourly_by_day[h].append(by_h.get(h, 0))
    c.close()

    predicted = {}
    for h in range(24):
        counts = hourly_by_day[h]
        if len(counts) < 7:
            counts = [0] * (7 - len(counts)) + counts
        wma = sum(cv * w for cv, w in zip(counts, weights)) / sum(weights)
        predicted[h] = round(wma, 1)

    future = [(h, predicted[h]) for h in range(24) if h > now.hour]
    future.sort(key=lambda x: -x[1])

    return {
        'predicted_by_hour': [{'hour': h, 'label': f'{h:02d}:00', 'predicted': predicted[h]} for h in range(24)],
        'next_peak_hour':    f'{future[0][0]:02d}:00' if future else '—',
        'next_peak_count':   future[0][1] if future else 0,
        'top_3_peaks':       [{'hour': f'{h:02d}:00', 'predicted': v} for h, v in future[:3]],
    }


# ── OCR Accuracy Trend ─────────────────────────────────────────────────────────
def get_ocr_accuracy_trend(days: int = 14) -> list:
    c   = _conn()
    now = datetime.now()
    out = []
    for i in range(days - 1, -1, -1):
        d_str = (now - timedelta(days=i)).strftime('%Y-%m-%d')
        label = (now - timedelta(days=i)).strftime('%b %d')
        row   = c.execute(
            "SELECT AVG(ocr_confidence), COUNT(*) FROM logs WHERE log_date=? AND ocr_confidence>0",
            (d_str,)
        ).fetchone()
        avg_conf = round(row[0] or 0, 1)
        count    = row[1] or 0
        high     = c.execute(
            "SELECT COUNT(*) FROM logs WHERE log_date=? AND ocr_confidence>60", (d_str,)
        ).fetchone()[0]
        out.append({
            'date': label, 'date_raw': d_str,
            'avg_conf': avg_conf, 'count': count,
            'high_rate': round(high / max(count, 1) * 100, 1),
        })
    c.close()
    return out


# ── Anomaly Detection ──────────────────────────────────────────────────────────
def detect_anomalies() -> dict:
    c   = _conn()
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    yest  = (now - timedelta(days=1)).strftime('%Y-%m-%d')

    # 1. Double Entry
    double_entries = []
    plates = c.execute(
        "SELECT DISTINCT plate FROM logs WHERE action IN ('ENTRY','EXIT')"
    ).fetchall()
    for (plate,) in plates:
        events   = c.execute(
            "SELECT action FROM logs WHERE plate=? AND action IN ('ENTRY','EXIT') ORDER BY id", (plate,)
        ).fetchall()
        last_act = None
        for (act,) in events:
            if act == 'ENTRY' and last_act == 'ENTRY':
                double_entries.append({'plate': plate, 'issue': 'ENTRY logged twice without EXIT', 'severity': 'medium'})
                break
            last_act = act

    # 2. Long Stay (entered but no exit in last 24h)
    long_stays = []
    entered = c.execute(
        "SELECT DISTINCT plate FROM logs WHERE action='ENTRY' AND log_date IN (?,?)", (today, yest)
    ).fetchall()
    for (plate,) in entered:
        has_exit = c.execute(
            "SELECT COUNT(*) FROM logs WHERE plate=? AND action='EXIT' AND log_date IN (?,?)",
            (plate, today, yest)
        ).fetchone()[0]
        if not has_exit:
            last_e = c.execute(
                "SELECT timestamp, log_date FROM logs WHERE plate=? AND action='ENTRY' ORDER BY id DESC LIMIT 1",
                (plate,)
            ).fetchone()
            if last_e:
                long_stays.append({'plate': plate, 'issue': f"No exit since {last_e[1]} {last_e[0]}", 'severity': 'low'})

    # 3. Rapid Repeat (same plate 3+ times in 1 hour today)
    rapid_repeats = []
    rows = c.execute(
        "SELECT plate, log_hour, COUNT(*) as cnt FROM logs WHERE log_date=? "
        "GROUP BY plate, log_hour HAVING cnt>=3", (today,)
    ).fetchall()
    for plate, hour, cnt in rows:
        rapid_repeats.append({'plate': plate, 'issue': f"Seen {cnt}× in {hour:02d}:00 — possible loop/glitch", 'severity': 'low'})

    # 4. High Denial Plates (denied 4+ times in last 24h)
    high_denial = []
    rows = c.execute(
        "SELECT plate, COUNT(*) as cnt FROM logs WHERE action='DENIED' AND log_date>=? "
        "GROUP BY plate HAVING cnt>=4 ORDER BY cnt DESC LIMIT 10", (yest,)
    ).fetchall()
    for plate, cnt in rows:
        high_denial.append({'plate': plate, 'issue': f"Denied {cnt}× in 24h — possible tailgater", 'severity': 'high'})

    # 5. Ghost Plates (unregistered, seen 3+)
    ghost_plates = []
    rows = c.execute(
        """SELECT l.plate, COUNT(*) as cnt FROM logs l
           LEFT JOIN vehicles v ON l.plate=v.plate
           WHERE v.plate IS NULL AND l.plate NOT IN ('READING...','UNREADABLE','')
           GROUP BY l.plate HAVING cnt>=3 ORDER BY cnt DESC LIMIT 10"""
    ).fetchall()
    for plate, cnt in rows:
        ghost_plates.append({'plate': plate, 'issue': f"Unregistered, seen {cnt}× — register or blacklist", 'severity': 'medium'})

    c.close()

    all_anomalies = (
        [{'type': 'Double Entry', **a} for a in double_entries] +
        [{'type': 'Long Stay',    **a} for a in long_stays[:10]] +
        [{'type': 'Rapid Repeat', **a} for a in rapid_repeats[:5]] +
        [{'type': 'High Denial',  **a} for a in high_denial] +
        [{'type': 'Ghost Plate',  **a} for a in ghost_plates]
    )

    return {
        'total': len(all_anomalies),
        'double_entries': double_entries,
        'long_stays':     long_stays[:10],
        'rapid_repeats':  rapid_repeats[:5],
        'high_denial':    high_denial,
        'ghost_plates':   ghost_plates,
        'all':            all_anomalies[:30],
    }


# ── Speed Analytics ────────────────────────────────────────────────────────────
def get_speed_analytics() -> dict:
    try:
        from speed_estimator import speed_estimator
        history = speed_estimator.get_history()
        status  = speed_estimator.status()
        if not history:
            return {'available': False, 'message': 'No speed readings yet'}
        speeds = [r['speed_kmh'] for r in history]
        return {
            'available':       True,
            'total_readings':  len(history),
            'avg_speed_kmh':   round(sum(speeds) / len(speeds), 1),
            'max_speed_kmh':   max(speeds),
            'overspeed_count': sum(1 for r in history if r['overspeed']),
            'overspeed_limit': status['overspeed_limit'],
            'top_offenders':   sorted([r for r in history if r['overspeed']], key=lambda r: -r['speed_kmh'])[:5],
            'recent':          history[:10],
        }
    except Exception:
        return {'available': False, 'message': 'Speed module not active'}


# ── Color Stats ────────────────────────────────────────────────────────────────
def get_color_stats() -> list:
    c    = _conn()
    rows = c.execute(
        "SELECT detected_color, COUNT(*) FROM logs WHERE detected_color!='' GROUP BY detected_color ORDER BY COUNT(*) DESC"
    ).fetchall()
    c.close()
    hex_map = {'White':'#EEEEEE','Black':'#222222','Silver':'#B0BEC5','Red':'#E53935',
               'Blue':'#1E88E5','Green':'#43A047','Yellow':'#FFD600','Orange':'#FF6D00',
               'Grey':'#9E9E9E','Brown':'#6D4C41','Purple':'#8E24AA'}
    return [{'color': r[0], 'count': r[1], 'hex': hex_map.get(r[0], '#888888')} for r in rows]


# ── Night / Day Split ──────────────────────────────────────────────────────────
def get_night_day_breakdown() -> dict:
    c    = _conn()
    rows = c.execute("SELECT log_hour, COUNT(*) FROM logs WHERE log_hour>=0 GROUP BY log_hour").fetchall()
    c.close()
    night, day = 0, 0
    for hour, cnt in rows:
        if hour >= 21 or hour <= 5:
            night += cnt
        else:
            day   += cnt
    total = max(night + day, 1)
    return {'night': night, 'day': day,
            'night_pct': round(night/total*100, 1), 'day_pct': round(day/total*100, 1)}


# ── Standard helpers (kept from v1) ───────────────────────────────────────────
def get_action_breakdown() -> list:
    c = _conn()
    rows = c.execute("SELECT action, COUNT(*) FROM logs GROUP BY action").fetchall()
    c.close()
    colors = {'ENTRY':'#00e87a','DENIED':'#ff3a5c','EXIT':'#00d4ff'}
    return [{'label': a, 'value': n, 'color': colors.get(a,'#888')} for a,n in rows]

def get_category_breakdown() -> list:
    c = _conn()
    rows = c.execute("SELECT category, COUNT(*) FROM vehicles GROUP BY category").fetchall()
    c.close()
    colors = {'Resident':'#00d4ff','Visitor':'#ffd040','Staff':'#9b6dff','Blacklisted':'#ff3a5c'}
    return [{'label': cat, 'value': n, 'color': colors.get(cat,'#888')} for cat,n in rows]

def get_vehicle_type_breakdown() -> list:
    c = _conn()
    rows = c.execute("SELECT vtype, COUNT(*) FROM logs WHERE vtype!='' GROUP BY vtype ORDER BY COUNT(*) DESC LIMIT 8").fetchall()
    c.close()
    return [{'label': vt or 'Unknown', 'value': n} for vt,n in rows]

def get_top_plates(limit=10) -> list:
    c = _conn()
    rows = c.execute(
        "SELECT l.plate,COUNT(*) as cnt,MAX(l.timestamp),v.owner,v.category "
        "FROM logs l LEFT JOIN vehicles v ON l.plate=v.plate "
        "WHERE l.plate IS NOT NULL AND l.plate!='' GROUP BY l.plate ORDER BY cnt DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    return [{'plate':r[0],'count':r[1],'last_seen':r[2],'owner':r[3] or '—','category':r[4] or 'Unregistered'} for r in rows]

def get_frequent_denials(limit=10) -> list:
    c = _conn()
    rows = c.execute(
        "SELECT plate,COUNT(*),MAX(timestamp) FROM logs WHERE action='DENIED' GROUP BY plate ORDER BY COUNT(*) DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    return [{'plate':r[0],'denials':r[1],'last_attempt':r[2]} for r in rows]

def get_gate_activity() -> list:
    c = _conn()
    rows = c.execute("SELECT gate, COUNT(*) FROM logs GROUP BY gate ORDER BY COUNT(*) DESC").fetchall()
    c.close()
    return [{'gate': g or 'Unknown', 'count': n} for g,n in rows]

def get_ocr_confidence_distribution() -> list:
    c = _conn()
    rows = c.execute("SELECT ocr_confidence FROM logs WHERE ocr_confidence>0").fetchall()
    c.close()
    b = {'0-20':0,'20-40':0,'40-60':0,'60-80':0,'80-100':0}
    for (conf,) in rows:
        if   conf<20:  b['0-20']   +=1
        elif conf<40:  b['20-40']  +=1
        elif conf<60:  b['40-60']  +=1
        elif conf<80:  b['60-80']  +=1
        else:          b['80-100'] +=1
    return [{'label':k,'value':v} for k,v in b.items()]

def get_alert_summary() -> dict:
    c = _conn()
    by_sev  = c.execute("SELECT severity,COUNT(*) FROM alerts GROUP BY severity").fetchall()
    by_type = c.execute("SELECT type,COUNT(*) FROM alerts GROUP BY type ORDER BY COUNT(*) DESC LIMIT 5").fetchall()
    unres   = c.execute("SELECT COUNT(*) FROM alerts WHERE resolved=0").fetchone()[0]
    c.close()
    return {
        'unresolved':  unres,
        'by_severity': [{'label':s,'value':n} for s,n in by_sev],
        'by_type':     [{'label':t,'value':n} for t,n in by_type],
    }


# ── Master payload ─────────────────────────────────────────────────────────────
def get_full_analytics() -> dict:
    return {
        'overview':               get_overview(),
        'hourly_heatmap':         get_hourly_heatmap(),
        'daily_trend':            get_daily_trend(14),
        'action_breakdown':       get_action_breakdown(),
        'category_breakdown':     get_category_breakdown(),
        'vehicle_type_breakdown': get_vehicle_type_breakdown(),
        'top_plates':             get_top_plates(10),
        'frequent_denials':       get_frequent_denials(8),
        'gate_activity':          get_gate_activity(),
        'ocr_confidence_dist':    get_ocr_confidence_distribution(),
        'ocr_accuracy_trend':     get_ocr_accuracy_trend(14),
        'alert_summary':          get_alert_summary(),
        'anomalies':              detect_anomalies(),
        'peak_prediction':        get_peak_hour_prediction(),
        'speed_analytics':        get_speed_analytics(),
        'color_stats':            get_color_stats(),
        'night_day':              get_night_day_breakdown(),
        'generated_at':           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }