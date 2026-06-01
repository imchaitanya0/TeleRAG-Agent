"""
TeleQnA Data Preparation Script
Loads the full 10,000-question TeleQnA dataset, splits by category,
and converts to instruction-tuning format for QLoRA fine-tuning.
"""
import json
import random
from pathlib import Path
from collections import Counter

# Use config for paths
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR


def load_teleqna(raw_dir: Path) -> list[dict]:
    """
    Loads the TeleQnA dataset JSON file.
    Expected format: list of dicts with keys like 'question', 'options', 'answer', 'category'.
    """
    # TeleQnA stores data in a single JSON file
    possible_paths = [
        raw_dir / "teleqna" / "TeleQnA.json",
        raw_dir / "teleqna" / "teleqna.json",
        raw_dir / "teleqna" / "data.json",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Loading TeleQnA from {path}")
            with open(path, "r") as f:
                data = json.load(f)
            # Handle both list and dict-wrapped formats
            if isinstance(data, dict):
                # TeleQnA wraps questions in a dict with numbered keys
                questions = list(data.values())
            else:
                questions = data
            print(f"Loaded {len(questions)} questions.")
            return questions

    print(f"ERROR: TeleQnA JSON not found in {raw_dir / 'teleqna/'}")
    print("Please download from https://github.com/netop-team/TeleQnA")
    print("Extract with password: teleqnadataset")
    return []


def get_category(question: dict) -> str:
    """Extracts the category from a TeleQnA question entry."""
    # TeleQnA uses 'category' field with values like:
    # 'Standards specifications', 'Standards overview', 'Research overview',
    # 'Research publications', 'Lexicon'
    return question.get("category", "Unknown")


def convert_to_instruction_format(question: dict) -> dict:
    """
    Converts a TeleQnA multiple-choice question into instruction-tuning format.
    """
    q_text = question.get("question", "")

    # Build options string from option_X keys
    options = []
    for key in sorted(question.keys()):
        if key.startswith("option"):
            # key format: "option 1", "option 2", etc.
            idx = key.replace("option ", "").strip()
            letter = chr(64 + int(idx)) if idx.isdigit() else idx  # 1->A, 2->B
            options.append(f"{letter}) {question[key]}")

    options_str = "\n".join(options)

    # Get answer
    answer_text = question.get("answer", "")

    # Build the instruction-tuning entry
    return {
        "instruction": "You are a telecom domain expert. Answer the following question accurately based on 3GPP standards and telecom knowledge.",
        "input": f"Question: {q_text}\n\nOptions:\n{options_str}",
        "output": f"The answer is: {answer_text}",
        "category": get_category(question),
    }


def split_dataset(questions: list[dict]) -> dict[str, list[dict]]:
    """
    Splits the 10K TeleQnA dataset into train/val/test/hard_test splits.

    Split strategy:
    - Train (3K): Standards Specifications (2K) + Standards Overview (1K) → QLoRA
    - Validation (1K): Stratified sample across all categories
    - Test (2K): Stratified sample, held out
    - Hard Test (1K): Standards Specifications only (GPT-4's weakest)
    - Remaining: Reserved
    """
    random.seed(42)  # Reproducibility

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for q in questions:
        cat = get_category(q)
        by_category.setdefault(cat, []).append(q)

    print("\nCategory distribution:")
    for cat, items in sorted(by_category.items()):
        print(f"  {cat}: {len(items)}")

    # Shuffle each category
    for cat in by_category:
        random.shuffle(by_category[cat])

    train, val, test, hard_test = [], [], [], []

    # Standards Specifications → 2K train + 1K hard_test + rest to test
    specs = by_category.get("Standards specifications", [])
    if len(specs) >= 3000:
        train.extend(specs[:2000])
        hard_test.extend(specs[2000:3000])
        test.extend(specs[3000:])
    else:
        # If fewer than 3K specs, take what we can
        split_point = len(specs) * 2 // 3
        train.extend(specs[:split_point])
        hard_test.extend(specs[split_point:])

    # Standards Overview → 1K train + rest to test
    overview = by_category.get("Standards overview", [])
    if len(overview) >= 1000:
        train.extend(overview[:1000])
        test.extend(overview[1000:])
    else:
        train.extend(overview)

    # Other categories → split proportionally into val and test
    for cat in by_category:
        if cat in ["Standards specifications", "Standards overview"]:
            continue
        items = by_category[cat]
        val_count = min(len(items) // 3, 300)  # Cap per category
        val.extend(items[:val_count])
        test.extend(items[val_count:])

    # If val is too small, move some from test
    if len(val) < 1000:
        needed = 1000 - len(val)
        random.shuffle(test)
        val.extend(test[:needed])
        test = test[needed:]

    # Cap test at 2K
    if len(test) > 2000:
        test = test[:2000]

    # Convert all to instruction format
    splits = {
        "train": [convert_to_instruction_format(q) for q in train],
        "val": [convert_to_instruction_format(q) for q in val],
        "test": [convert_to_instruction_format(q) for q in test],
        "hard_test": [convert_to_instruction_format(q) for q in hard_test],
    }

    return splits


def save_splits(splits: dict[str, list[dict]], output_dir: Path):
    """Saves each split as a JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in splits.items():
        file_path = output_dir / f"teleqna_{split_name}.jsonl"
        with open(file_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {split_name}: {len(data)} examples → {file_path}")


def prepare_teleqna_splits():
    """Main entry point for TeleQnA data preparation."""
    print("=" * 50)
    print("TeleQnA Data Preparation")
    print("=" * 50)

    questions = load_teleqna(DATA_RAW_DIR)
    if not questions:
        print("\nNo data loaded. Creating empty stub files for development...")
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test", "hard_test"]:
            stub_path = DATA_PROCESSED_DIR / f"teleqna_{split}.jsonl"
            with open(stub_path, "w") as f:
                dummy = {
                    "instruction": "You are a telecom domain expert.",
                    "input": "Question: Placeholder\nOptions:\nA) Opt1\nB) Opt2",
                    "output": "The answer is: A) Opt1",
                    "category": "stub",
                }
                f.write(json.dumps(dummy) + "\n")
            print(f"Created stub: {stub_path}")
        return

    splits = split_dataset(questions)
    save_splits(splits, DATA_PROCESSED_DIR)

    print(f"\nSplit summary:")
    for name, data in splits.items():
        cats = Counter(d["category"] for d in data)
        print(f"  {name}: {len(data)} total — {dict(cats)}")


if __name__ == "__main__":
    prepare_teleqna_splits()
