import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_synthetic_kpis():
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cells = [f"CELL_{i:03d}" for i in range(1, 11)]
    start_time = datetime.now() - timedelta(days=7)
    
    out_file = out_dir / "kpis.csv"
    
    print("Generating synthetic KPI time-series...")
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "cell_id", "rsrp", "sinr", "rrc_success_rate"])
        
        # 7 days, 15 minute intervals = 672 intervals
        for interval in range(672):
            timestamp = start_time + timedelta(minutes=interval * 15)
            ts_str = timestamp.isoformat()
            
            for cell in cells:
                # Normal behavior
                rsrp = random.uniform(-90, -75)
                sinr = random.uniform(10, 20)
                rrc = random.uniform(98.5, 100.0)
                
                # Inject anomaly randomly (1% chance)
                if random.random() < 0.01:
                    rsrp = random.uniform(-110, -100) # Degraded RSRP
                    sinr = random.uniform(0, 5)       # Degraded SINR
                    rrc = random.uniform(50.0, 80.0)  # Degraded RRC
                    
                writer.writerow([
                    ts_str,
                    cell,
                    round(rsrp, 2),
                    round(sinr, 2),
                    round(rrc, 2)
                ])
                
    print(f"Successfully generated KPI time-series at {out_file}")

if __name__ == "__main__":
    generate_synthetic_kpis()
