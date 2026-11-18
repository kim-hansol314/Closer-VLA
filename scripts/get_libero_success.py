import os
import argparse
from collections import defaultdict

def compute_success_rates(path):
    task_stats = defaultdict(lambda: {"success": 0, "total": 0})

    for filename in os.listdir(path):
        if not filename.endswith(".mp4"):
            continue

        parts = filename.split("_")
        if len(parts) < 3:
            continue  # Skip malformed filenames

        # Get the task name by removing rollout prefix and _success/_fail suffix
        rollout_part = parts[0]  # e.g., rollout0
        suffix = "_success.mp4"
        if not filename.endswith(suffix):
            suffix = "_failure.mp4"

        task_full = filename[len(rollout_part) + 1 : -len(suffix)]  # slice task name

        task_stats[task_full]["total"] += 1
        if filename.endswith("_success.mp4"):
            task_stats[task_full]["success"] += 1

    print(f"{'TASK':<80} | SUCCESS RATE")
    print("-" * 100)
    success_rate = 0.0
    for i, (task, stats) in enumerate(sorted(task_stats.items())):
        total = stats["total"]
        success = stats["success"]
        rate = success / total if total > 0 else 0
        print(f"{i} {task:<80} | {success}/{total} = {rate:.2f}")
        success_rate += rate

    if len(task_stats) > 0:
        print(f"Average success rate: {success_rate / len(task_stats):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute task-wise success rates.")
    parser.add_argument("--path", type=str, help="Directory containing rollout mp4 files")
    args = parser.parse_args()

    compute_success_rates(args.path)
