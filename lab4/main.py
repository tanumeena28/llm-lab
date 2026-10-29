"""
Master Orchestrator Script for Lab 4.
Runs end-to-end N-gram language modeling pipeline:
1. Loads 1,000,000 sentence training set, dev set, and test set from Lab3 data split.
2. Fits N-gram counts for N=1, 2, 3, 4.
3. Tunes Add-K parameter (K) and Interpolation constants (lambda) on Dev set.
4. Evaluates Test set perplexities across all 20 combinations (4 N-gram orders x 5 Smoothing techniques).
5. Exports quantitative results CSVs & generates publication-grade visualizations.
"""

import os
import sys
import time

# Ensure lab4 directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import run_full_lab4_evaluation
from visualize import generate_all_plots


def main():
    print("==========================================================================")
    print("  LAB 4: N-GRAM LANGUAGE MODELING & SMOOTHING ANALYSIS PIPELINE           ")
    print("==========================================================================")

    t_start = time.time()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lab3_data_dir = os.path.abspath(os.path.join(base_dir, "..", "Lab3", "data"))
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(lab3_data_dir):
        raise FileNotFoundError(
            f"Lab 3 data directory not found at '{lab3_data_dir}'. "
            "Please ensure Lab3/data/ contains train_1m.csv, dev.csv, and test.csv."
        )

    print(f"Using Lab 3 Dataset Directory: '{lab3_data_dir}'")
    print(f"Output Results Directory:      '{results_dir}'")

    # Step 1: Run Full Evaluation Pipeline
    eval_results = run_full_lab4_evaluation(data_dir=lab3_data_dir, results_dir=results_dir)

    # Step 2: Generate Publication-Grade Visualizations
    print("\n--- Generating Visualization Charts ---")
    generate_all_plots(results_dir=results_dir, plots_dir=plots_dir)

    total_time = time.time() - t_start
    print("\n==========================================================================")
    print(f"  LAB 4 PIPELINE COMPLETED SUCCESSFULLY IN {total_time / 60.0:.2f} MINUTES!  ")
    print("==========================================================================")


if __name__ == '__main__':
    main()
