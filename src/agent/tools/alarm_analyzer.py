"""
src/agent/tools/alarm_analyzer.py

Tool: Alarm Analyzer
Analyzes O-RAN alarm logs for root cause analysis.

Data sources (in priority order):
  1. data/raw/oran/filteredBenign1.csv   — real NetsLab-5GORAN-IDD testbed data
  2. data/synthetic/alarms.json          — synthetic O-RAN alarm events

Supports:
  - Alarm frequency analysis per cell/node
  - Alarm storm detection (burst of ≥3 alarms within 10 min window)
  - Probable cause correlation
  - Severity distribution summary
"""

import json
import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from src.config import DATA_SYNTHETIC_DIR, DATA_RAW_DIR

# ─── data paths ────────────────────────────────────────────────
SYNTHETIC_ALARMS_PATH = DATA_SYNTHETIC_DIR / "alarms.json"
REAL_ORAN_PATH = DATA_RAW_DIR / "oran" / "filteredBenign1.csv"

# ─── lazy loaded data ──────────────────────────────────────────
_alarm_data: Optional[list[dict]] = None


def _load_alarms() -> list[dict]:
    """Load and merge real O-RAN data + synthetic alarms."""
    global _alarm_data
    if _alarm_data is not None:
        return _alarm_data

    alarms: list[dict] = []

    # 1. Load real O-RAN testbed data (NetsLab-5GORAN-IDD)
    if REAL_ORAN_PATH.exists():
        try:
            with open(REAL_ORAN_PATH, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Map CSV columns → unified alarm schema
                    alarms.append({
                        "alarm_id": row.get("id", f"ORAN_{len(alarms):04d}"),
                        "alarm_type": row.get("type", row.get("alarm_type", "unknown")),
                        "severity": row.get("severity", "major").lower(),
                        "timestamp": row.get("timestamp", row.get("time", "")),
                        "cell_id": row.get("cell_id", row.get("node", "UNKNOWN")),
                        "node_id": row.get("node_id", row.get("node", "UNKNOWN")),
                        "probable_cause": row.get("probable_cause", row.get("cause", "")),
                        "description": row.get("description", row.get("msg", "")),
                        "source": "real_oran",
                        "is_storm": False,
                    })
        except Exception as e:
            print(f"[AlarmAnalyzer] Warning: Could not load real O-RAN data: {e}")

    # 2. Load synthetic alarms
    if SYNTHETIC_ALARMS_PATH.exists():
        try:
            with open(SYNTHETIC_ALARMS_PATH) as f:
                synthetic = json.load(f)
            for alarm in synthetic:
                alarm["source"] = "synthetic"
            alarms.extend(synthetic)
        except Exception as e:
            print(f"[AlarmAnalyzer] Warning: Could not load synthetic alarms: {e}")

    _alarm_data = alarms
    print(f"[AlarmAnalyzer] Loaded {len(alarms)} alarms "
          f"({sum(1 for a in alarms if a.get('source')=='real_oran')} real, "
          f"{sum(1 for a in alarms if a.get('source')=='synthetic')} synthetic)")
    return _alarm_data


def _detect_storms(alarms: list[dict], window_minutes: int = 10, min_size: int = 3) -> list[dict]:
    """
    Detect alarm storms: bursts of ≥ min_size alarms on the same cell
    within a window_minutes sliding window.
    """
    # Group by cell_id
    by_cell: dict[str, list] = defaultdict(list)
    for a in alarms:
        ts_str = a.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        by_cell[a.get("cell_id", "UNKNOWN")].append((ts, a))

    storms = []
    for cell_id, events in by_cell.items():
        events.sort(key=lambda x: x[0])
        i = 0
        while i < len(events):
            window = [events[i]]
            j = i + 1
            while j < len(events) and (events[j][0] - events[i][0]) <= timedelta(minutes=window_minutes):
                window.append(events[j])
                j += 1
            if len(window) >= min_size:
                alarm_types = Counter(a["alarm_type"] for _, a in window)
                storms.append({
                    "cell_id": cell_id,
                    "node_id": window[0][1].get("node_id", ""),
                    "start_time": window[0][0].isoformat(),
                    "end_time": window[-1][0].isoformat(),
                    "alarm_count": len(window),
                    "dominant_type": alarm_types.most_common(1)[0][0],
                    "alarm_types": dict(alarm_types),
                    "probable_causes": list({a.get("probable_cause", "") for _, a in window if a.get("probable_cause")}),
                })
                i = j  # skip past the storm
            else:
                i += 1

    return sorted(storms, key=lambda x: x["alarm_count"], reverse=True)


def alarm_analyzer_tool(
    cell_id: Optional[str] = None,
    node_id: Optional[str] = None,
    alarm_type: Optional[str] = None,
    severity: Optional[str] = None,
    top_n: int = 5,
) -> dict:
    """
    Analyze O-RAN alarm data for root cause analysis.

    Args:
        cell_id:    Filter to a specific cell (e.g. "CELL_001"). None = all cells.
        node_id:    Filter to a specific gNB node. None = all nodes.
        alarm_type: Filter by alarm type (e.g. "cellUnavailable"). None = all types.
        severity:   Filter by severity ("critical"/"major"/"minor"). None = all.
        top_n:      Number of top entries to return in frequency tables.

    Returns:
        {
            "total_alarms":    int,
            "filtered_alarms": int,
            "severity_dist":   dict,  {"critical": N, ...}
            "top_alarm_types": list,  [{"type": str, "count": int}, ...]
            "top_cells":       list,  [{"cell_id": str, "count": int}, ...]
            "storms_detected": list,  [{cell_id, start_time, alarm_count, dominant_type, probable_causes}, ...]
            "top_causes":      list,  [{"cause": str, "count": int}, ...]
            "rca_summary":     str,   human-readable root cause analysis
        }
    """
    alarms = _load_alarms()
    total = len(alarms)

    # Apply filters
    filtered = alarms
    if cell_id:
        filtered = [a for a in filtered if a.get("cell_id", "").upper() == cell_id.upper()]
    if node_id:
        filtered = [a for a in filtered if a.get("node_id", "").upper() == node_id.upper()]
    if alarm_type:
        filtered = [a for a in filtered if alarm_type.lower() in a.get("alarm_type", "").lower()]
    if severity:
        filtered = [a for a in filtered if a.get("severity", "").lower() == severity.lower()]

    if not filtered:
        return {
            "total_alarms": total,
            "filtered_alarms": 0,
            "severity_dist": {},
            "top_alarm_types": [],
            "top_cells": [],
            "storms_detected": [],
            "top_causes": [],
            "rca_summary": "No alarms found matching the specified filters.",
        }

    # Frequency analysis
    severity_dist = dict(Counter(a.get("severity", "unknown") for a in filtered))
    type_counts = Counter(a.get("alarm_type", "unknown") for a in filtered).most_common(top_n)
    cell_counts = Counter(a.get("cell_id", "unknown") for a in filtered).most_common(top_n)
    cause_counts = Counter(
        a.get("probable_cause", "") for a in filtered if a.get("probable_cause")
    ).most_common(top_n)

    # Storm detection
    storms = _detect_storms(filtered)[:top_n]

    # Generate RCA summary
    top_type = type_counts[0][0] if type_counts else "unknown"
    top_cell = cell_counts[0][0] if cell_counts else "unknown"
    critical_count = severity_dist.get("critical", 0)
    storm_summary = (
        f"{len(storms)} alarm storm(s) detected — most severe on {storms[0]['cell_id']} "
        f"({storms[0]['alarm_count']} alarms, dominant: {storms[0]['dominant_type']})."
        if storms else "No alarm storms detected."
    )
    top_cause = cause_counts[0][0] if cause_counts else "not identified"

    rca_summary = (
        f"Analysis of {len(filtered)} alarms"
        f"{' on ' + cell_id if cell_id else ''}:\n"
        f"• Most frequent alarm type: {top_type} ({type_counts[0][1] if type_counts else 0} occurrences)\n"
        f"• Most affected cell: {top_cell}\n"
        f"• Critical alarms: {critical_count}\n"
        f"• {storm_summary}\n"
        f"• Most probable root cause: {top_cause}"
    )

    return {
        "total_alarms": total,
        "filtered_alarms": len(filtered),
        "severity_dist": severity_dist,
        "top_alarm_types": [{"type": t, "count": c} for t, c in type_counts],
        "top_cells": [{"cell_id": cid, "count": c} for cid, c in cell_counts],
        "storms_detected": storms,
        "top_causes": [{"cause": cause, "count": c} for cause, c in cause_counts],
        "rca_summary": rca_summary,
    }


# ── smoke test ────────────────────────────────────────────────
if __name__ == "__main__":
    result = alarm_analyzer_tool(severity="critical", top_n=3)
    print(f"\nTotal alarms: {result['total_alarms']}")
    print(f"Critical alarms: {result['filtered_alarms']}")
    print(f"\nRCA Summary:\n{result['rca_summary']}")
    print(f"\nStorms detected: {len(result['storms_detected'])}")
