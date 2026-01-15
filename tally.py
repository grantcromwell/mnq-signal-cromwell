#!/usr/bin/env python3
"""
Tally Script - Records simulated trades when signal is YES

Usage: python tally.py

Runs quick_signal.py and if signal is YES, records a trade to backtest.csv
"""

import sys
import os
import csv
import subprocess
from datetime import datetime
from pathlib import Path


def run_quick_signal():
    """Run quick_signal.py and capture output."""
    result = subprocess.run(
        [sys.executable, "quick_signal.py"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.stdout + result.stderr


def parse_signal(output):
    """Parse signal from quick_signal output."""
    for line in output.split("\n"):
        if ">>> SIGNAL:" in line:
            signal = line.split(">>> SIGNAL:")[1].strip()
            return signal.upper() == "YES"
    return None


def record_trade(backtest_path: str = "backtest.csv"):
    """Record a YES signal trade to backtest.csv."""
    timestamp = datetime.now().isoformat()

    file_exists = Path(backtest_path).exists()

    with open(backtest_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "signal", "entry_price", "fib_level", "status"]
            )

        writer.writerow([timestamp, "YES", "TBD", "TBD", "open"])

    print(f"Trade recorded to {backtest_path}")


def main():
    print("Running quick signal check...")
    output = run_quick_signal()

    signal = parse_signal(output)

    if signal is None:
        print("ERROR: Could not parse signal from quick_signal.py")
        print("Output:", output[-500:])
        sys.exit(1)

    if signal:
        print("Signal: YES - Recording trade...")
        record_trade()
    else:
        print("Signal: NO - No trade recorded")

    print("\nDone.")


if __name__ == "__main__":
    main()
