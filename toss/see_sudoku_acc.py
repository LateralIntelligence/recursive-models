import json
from pathlib import Path

# Change this to the directory containing all your experiment folders
base_dir = Path("/home/jsjung00/Desktop/Code/recursive-models/outputs/sweep-flm-infill-sudoku-easy")

for run_dir in sorted(base_dir.iterdir()):
    if not run_dir.is_dir():
        continue

    results_path = run_dir / "sudoku_eval" / "last" / "results.json"

    if not results_path.exists():
        print(f"{run_dir.name:80}  MISSING")
        continue

    try:
        with open(results_path) as f:
            results = json.load(f)

        accuracy = results.get("accuracy", "N/A")
        num_correct = results.get("num_correct", "N/A")
        num_total = results.get("num_total", "N/A")

        print(
            f"{run_dir.name:80}  "
            f"acc={accuracy:.4f}  "
            f"({num_correct}/{num_total})"
        )

    except Exception as e:
        print(f"{run_dir.name:80}  ERROR: {e}")