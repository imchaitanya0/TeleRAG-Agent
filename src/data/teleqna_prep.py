import json
import random
from pathlib import Path

def prepare_teleqna_splits():
    data_raw_dir = Path("data/raw/teleqna")
    data_processed_dir = Path("data/processed")
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    # In a real scenario, this would load the actual TeleQnA JSON file.
    # For this template/scaffolding, we'll create a dummy structure if the real one isn't found.
    # Replace with actual loading logic when the dataset is downloaded.
    print("Preparing TeleQnA data splits...")
    print("Note: Ensure the real TeleQnA dataset is placed in data/raw/teleqna/ before running for real.")
    
    # Placeholder for the actual logic that reads the 10,000 questions JSON
    # and splits them into train (3K), val (1K), test (2K), hard_test (1K).
    # Since we are scaffolding, we will just create empty JSONL files for now.
    
    splits = ["train", "val", "test", "hard_test"]
    for split in splits:
        file_path = data_processed_dir / f"teleqna_{split}.jsonl"
        with open(file_path, "w") as f:
            # Write a dummy entry just to initialize the file
            dummy_entry = {
                "instruction": "Answer the following telecom question...",
                "input": "Q: Placeholder question? \nA) Opt 1 B) Opt 2",
                "output": "The answer is A: Opt 1. \nExplanation: Placeholder."
            }
            f.write(json.dumps(dummy_entry) + "\n")
            
    print(f"Created stub JSONL files in {data_processed_dir}")
    print("TODO: Implement actual JSON parsing and splitting logic once data is downloaded.")

if __name__ == "__main__":
    prepare_teleqna_splits()
