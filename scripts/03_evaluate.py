"""Pooled out-of-fold evaluation of every variant with employee-level paired bootstrap.
Writes results/metrics/summary.csv, paired.csv and pooled_main.npz.

  python scripts/03_evaluate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from faenet.evaluation import main

if __name__ == "__main__":
    main()
