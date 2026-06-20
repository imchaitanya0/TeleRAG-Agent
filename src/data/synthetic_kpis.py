"""
Synthetic KPI Generator
Generates cell-level KPI time-series with anomalies correlated to alarm storm timestamps.
"""
import csv
import json
import random
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import DATA_SYNTHETIC_DIR


CELLS = [f"CELL_{i:03d}" for i in range(1, 11)]

# Normal KPI ranges
KPI_NORMAL = {
    "rsrp": {"mean": -82, "std": 5, "unit": "dBm"},      # Good: -75 to -90
    "sinr": {"mean": 15, "std": 3, "unit": "dB"},          # Good: 10-20
    "rrc_success_rate": {"mean": 99.5, "std": 0.3, "unit": "%"},  # Good: 98.5-100
}

# Degraded KPI ranges (during alarm storms)
KPI_DEGRADED = {
    "rsrp": {"mean": -105, "std": 3},       # Degraded: -100 to -110
    "sinr": {"mean": 2, "std": 1.5},         # Degraded: 0-5
    "rrc_success_rate": {"mean": 67, "std": 10},  # Degraded: 50-80
}


def load_storm_timestamps() -> dict:
    """
    Loads alarm storm timestamps so KPI anomalies can be correlated.
    Returns: {cell_id: [list of storm datetime objects]}
    """
    storm_file = DATA_SYNTHETIC_DIR / "storm_timestamps.json"
    cell_storm_times: dict[str, list[datetime]] = {}

    if not storm_file.exists():
        print("WARNING: storm_timestamps.json not found. Run synthetic_alarms.py first.")
        print("KPI anomalies will be generated randomly (uncorrelated).")
        return cell_storm_times

    with open(storm_file, "r") as f:
        storms = json.load(f)

    for storm_id, info in storms.items():
        cell = info["cell_id"]
        if cell not in cell_storm_times:
            cell_storm_times[cell] = []
        # Use the earliest timestamp in the storm as the anomaly trigger
        ts_list = sorted(info["timestamps"])
        if ts_list:
            cell_storm_times[cell].append(
                datetime.fromisoformat(ts_list[0])
            )

    return cell_storm_times


def is_during_storm(timestamp: datetime, storm_times: list[datetime], window_minutes: int = 30) -> bool:
    """
    Returns True if the given timestamp falls within a storm window.
    The anomaly persists for `window_minutes` after the storm starts.
    """
    for storm_start in storm_times:
        storm_end = storm_start + timedelta(minutes=window_minutes)
        if storm_start <= timestamp <= storm_end:
            return True
    return False


def generate_synthetic_kpis():
    """
    Generates cell-level KPI time-series (15-minute intervals, 7 days)
    with anomalies correlated to alarm storm timestamps.
    """
    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    out_file = DATA_SYNTHETIC_DIR / "kpis.csv"
    start_time = datetime.now() - timedelta(days=7)

    # Load storm timestamps for correlation
    cell_storms = load_storm_timestamps()

    print("Generating synthetic KPI time-series...")

    anomaly_count = 0
    total_rows = 0

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cell_id", "rsrp", "sinr", "rrc_success_rate", "is_anomaly"])

        # 7 days × 24 hours × 4 (15-min intervals) = 672 intervals
        for interval in range(672):
            timestamp = start_time + timedelta(minutes=interval * 15)
            ts_str = timestamp.isoformat()

            for cell in CELLS:
                # Check if this cell has a storm at this time
                storms_for_cell = cell_storms.get(cell, [])
                anomaly = is_during_storm(timestamp, storms_for_cell)

                if anomaly:
                    # Degraded KPIs during storm window
                    rsrp = random.gauss(KPI_DEGRADED["rsrp"]["mean"], KPI_DEGRADED["rsrp"]["std"])
                    sinr = random.gauss(KPI_DEGRADED["sinr"]["mean"], KPI_DEGRADED["sinr"]["std"])
                    rrc = random.gauss(KPI_DEGRADED["rrc_success_rate"]["mean"],
                                       KPI_DEGRADED["rrc_success_rate"]["std"])
                    anomaly_count += 1
                else:
                    # Normal KPIs
                    rsrp = random.gauss(KPI_NORMAL["rsrp"]["mean"], KPI_NORMAL["rsrp"]["std"])
                    sinr = random.gauss(KPI_NORMAL["sinr"]["mean"], KPI_NORMAL["sinr"]["std"])
                    rrc = random.gauss(KPI_NORMAL["rrc_success_rate"]["mean"],
                                       KPI_NORMAL["rrc_success_rate"]["std"])

                # Clamp values to realistic bounds
                rrc = max(0, min(100, rrc))

                writer.writerow([
                    ts_str,
                    cell,
                    round(rsrp, 2),
                    round(sinr, 2),
                    round(rrc, 2),
                    1 if anomaly else 0,
                ])
                total_rows += 1

    print(f"Generated {total_rows} KPI rows ({anomaly_count} anomalous)")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    generate_synthetic_kpis()
