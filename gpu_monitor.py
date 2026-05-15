"""
gpu_monitor.py
--------------
Run this in a separate terminal while using the RAG system
to watch GPU/VRAM usage in real time.

Usage:
    python gpu_monitor.py
"""

import time
import torch

def monitor(interval_sec=1.0):
    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Nothing to monitor.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total    = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"Monitoring: {gpu_name}  |  Total VRAM: {total:.1f} GB")
    print(f"{'─'*45}")
    print(f"{'Used':>8}  {'Free':>8}  {'Usage':>8}  {'Status'}")
    print(f"{'─'*45}")

    try:
        while True:
            free, total_bytes = torch.cuda.mem_get_info(0)
            used  = (total_bytes - free) / 1024**3
            total = total_bytes / 1024**3
            pct   = (used / total) * 100

            if pct < 50:
                status = "● idle"
            elif pct < 80:
                status = "▲ active"
            else:
                status = "▲▲ heavy load"

            print(f"{used:>6.2f}GB  {free/1024**3:>6.2f}GB  {pct:>7.1f}%  {status}", end="\r")
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")

if __name__ == "__main__":
    monitor()
