"""Utility functions for the modeling pipeline."""

from datetime import datetime
import psutil
import time

def print_step_time(step_name: str, start_time: float) -> None:
    """Print the execution time and memory usage for a processing step."""
    elapsed = time.time() - start_time
    memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"{datetime.now().strftime('%H:%M:%S')} - {step_name}: "
          f"{elapsed:.2f} seconds (Memory: {memory:.2f} MB)")

def calculate_age_from_dob(dob: str) -> float:
    """Calculate age from date of birth in YYMM format."""
    try:
        yy = int(str(dob)[:2])
        calculated_age = 98 - yy  # Dataset year is 1998
        return calculated_age
    except:
        return float('nan')