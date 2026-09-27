"""
run_all.py
===========
Runs the whole pipeline in order:
  1. screener.py       -> nepse_heston_screener_full.csv (all stocks, ranked)
                        -> top_n_paths.npz (simulated paths for the top N)
  2. dashboard_screener.py -> dashboard_screener_top10.html
  3. dashboard_performance.py -> dashboard_performance_top10.html
                              (needs --history-csv; writes an explanatory
                               placeholder if that file isn't found)

Usage:
    python run_all.py --data-dir data --out out --top-n 10 \
        --history-csv data/price_history.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--history-csv", default="data/price_history.csv")
    args = ap.parse_args()

    steps = [
        [sys.executable, "screener.py", "--data-dir", args.data_dir, "--out", args.out, "--top-n", str(args.top_n)],
        [sys.executable, "dashboard_screener.py", "--out", args.out],
        [sys.executable, "dashboard_performance.py", "--out", args.out, "--history-csv", args.history_csv],
    ]
    for cmd in steps:
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(project_root))


if __name__ == "__main__":
    main()
