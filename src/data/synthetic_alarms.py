import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_synthetic_alarms():
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    alarm_types = ["cellUnavailable", "connectionLossOAM", "hwFault"]
    severities = ["critical", "major", "minor", "warning"]
    cells = [f"CELL_{i:03d}" for i in range(1, 11)]
    nodes = [f"gNB_{i:02d}" for i in range(1, 4)]
    
    alarms = []
    start_time = datetime.now() - timedelta(days=7)
    
    print("Generating synthetic O-RAN alarms...")
    for i in range(500):
        # Generate random timestamp within the last 7 days
        offset = timedelta(minutes=random.randint(0, 7 * 24 * 60))
        timestamp = start_time + offset
        
        alarm_type = random.choice(alarm_types)
        
        alarm = {
            "alarm_id": f"ALM_{i:04d}",
            "alarm_type": alarm_type,
            "severity": random.choice(severities) if alarm_type != "cellUnavailable" else "critical",
            "timestamp": timestamp.isoformat(),
            "cell_id": random.choice(cells),
            "node_id": random.choice(nodes),
            "probable_cause": f"Synthetic cause for {alarm_type}",
            "description": f"Generated synthetic alarm of type {alarm_type}"
        }
        alarms.append(alarm)
        
    # Sort by timestamp
    alarms.sort(key=lambda x: x["timestamp"])
    
    out_file = out_dir / "alarms.json"
    with open(out_file, "w") as f:
        json.dump(alarms, f, indent=2)
        
    print(f"Successfully generated 500 synthetic alarms at {out_file}")

if __name__ == "__main__":
    generate_synthetic_alarms()
