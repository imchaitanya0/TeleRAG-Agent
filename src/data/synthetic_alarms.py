"""
Synthetic O-RAN Alarm Generator
Generates realistic alarm events with temporal correlations (alarm storms).
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from src.config import DATA_SYNTHETIC_DIR


# Alarm type definitions with realistic probable causes
ALARM_DEFINITIONS = {
    "cellUnavailable": {
        "probable_causes": [
            "Power supply failure in RU",
            "Fronthaul link disconnected",
            "Software crash in DU cell processing",
        ],
        "default_severity": "critical",
    },
    "connectionLossOAM": {
        "probable_causes": [
            "OAM interface timeout on gNB",
            "Network routing failure to SMO",
            "TLS certificate expired on O1 interface",
        ],
        "default_severity": "major",
    },
    "hwFault": {
        "probable_causes": [
            "FPGA temperature exceeded threshold",
            "Memory ECC uncorrectable error",
            "Fan unit failure detected",
        ],
        "default_severity": "minor",
    },
}

SEVERITIES = ["critical", "major", "minor", "warning"]
CELLS = [f"CELL_{i:03d}" for i in range(1, 11)]
NODES = [f"gNB_{i:02d}" for i in range(1, 4)]
# Map cells to nodes (each gNB serves ~3-4 cells)
CELL_TO_NODE = {cell: NODES[i // 4] for i, cell in enumerate(CELLS)}


def generate_alarm_storm(
    base_time: datetime, cell_id: str, storm_size: int = 5
) -> list[dict]:
    """
    Generates a cluster of correlated alarms (alarm storm) on a single cell.
    Alarms fire within a 2-10 minute window — this is how real O-RAN alarm
    storms behave (cascading failures).
    """
    alarms = []
    alarm_types = list(ALARM_DEFINITIONS.keys())

    # Pick a primary alarm type for the storm
    primary_type = random.choice(alarm_types)

    for i in range(storm_size):
        # Each alarm in the storm is offset by 0-10 minutes from base
        offset_minutes = random.uniform(0, 10)
        timestamp = base_time + timedelta(minutes=offset_minutes)

        # Primary alarm type dominates; occasionally other types cascade
        if i == 0 or random.random() < 0.6:
            alarm_type = primary_type
        else:
            alarm_type = random.choice(alarm_types)

        defn = ALARM_DEFINITIONS[alarm_type]
        alarms.append(
            {
                "alarm_id": None,  # Set later
                "alarm_type": alarm_type,
                "severity": defn["default_severity"]
                if i == 0
                else random.choice(SEVERITIES),
                "timestamp": timestamp.isoformat(),
                "cell_id": cell_id,
                "node_id": CELL_TO_NODE[cell_id],
                "probable_cause": random.choice(defn["probable_causes"]),
                "description": f"Alarm storm event {i+1}/{storm_size} on {cell_id}",
                "is_storm": True,
                "storm_id": None,  # Set later
            }
        )
    return alarms


def generate_synthetic_alarms() -> list[dict]:
    """
    Generates ~500 synthetic O-RAN alarm events with:
    - Individual random alarms (~350)
    - Alarm storms with temporal clustering (~150 across ~25 storms)
    """
    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    all_alarms = []
    start_time = datetime.now() - timedelta(days=7)
    alarm_types = list(ALARM_DEFINITIONS.keys())

    print("Generating synthetic O-RAN alarms...")

    # Generate ~25 alarm storms (each 4-8 alarms)
    storm_count = 0
    storm_alarm_count = 0
    for _ in range(25):
        storm_time = start_time + timedelta(
            minutes=random.randint(0, 7 * 24 * 60)
        )
        cell = random.choice(CELLS)
        storm_size = random.randint(4, 8)
        storm_alarms = generate_alarm_storm(storm_time, cell, storm_size)

        for alarm in storm_alarms:
            alarm["storm_id"] = f"STORM_{storm_count:03d}"

        all_alarms.extend(storm_alarms)
        storm_count += 1
        storm_alarm_count += len(storm_alarms)

    # Fill remaining with individual random alarms to reach ~500
    individual_target = max(0, 500 - storm_alarm_count)
    for _ in range(individual_target):
        offset = timedelta(minutes=random.randint(0, 7 * 24 * 60))
        timestamp = start_time + offset
        alarm_type = random.choice(alarm_types)
        defn = ALARM_DEFINITIONS[alarm_type]

        all_alarms.append(
            {
                "alarm_id": None,
                "alarm_type": alarm_type,
                "severity": random.choice(SEVERITIES),
                "timestamp": timestamp.isoformat(),
                "cell_id": random.choice(CELLS),
                "node_id": random.choice(NODES),
                "probable_cause": random.choice(defn["probable_causes"]),
                "description": f"Individual alarm: {alarm_type}",
                "is_storm": False,
                "storm_id": None,
            }
        )

    # Sort by timestamp and assign IDs
    all_alarms.sort(key=lambda x: x["timestamp"])
    for i, alarm in enumerate(all_alarms):
        alarm["alarm_id"] = f"ALM_{i:04d}"

    # Save
    out_file = DATA_SYNTHETIC_DIR / "alarms.json"
    with open(out_file, "w") as f:
        json.dump(all_alarms, f, indent=2)

    # Also save storm timestamps for KPI correlation
    storm_times_file = DATA_SYNTHETIC_DIR / "storm_timestamps.json"
    storm_events = {}
    for alarm in all_alarms:
        if alarm["is_storm"] and alarm["storm_id"]:
            sid = alarm["storm_id"]
            if sid not in storm_events:
                storm_events[sid] = {
                    "cell_id": alarm["cell_id"],
                    "timestamps": [],
                }
            storm_events[sid]["timestamps"].append(alarm["timestamp"])

    with open(storm_times_file, "w") as f:
        json.dump(storm_events, f, indent=2)

    print(f"Generated {len(all_alarms)} alarms ({storm_count} storms, {individual_target} individual)")
    print(f"Saved to: {out_file}")
    print(f"Storm timestamps saved to: {storm_times_file}")

    return all_alarms


if __name__ == "__main__":
    generate_synthetic_alarms()
