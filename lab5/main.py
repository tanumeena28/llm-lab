"""
Master Orchestrator Script for Lab 5.
Trains Word2Vec, FastText, Doc2Vec on 1,000,000 sentences.
Performs K-Means clustering & nearest sentence search on Dev and Test sets.
"""

import os
import sys
import time

# Ensure lab5 directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import run_full_lab5_experiments


def main():
    print("==========================================================================")
    print("  LAB 5: WORD2VEC, FASTTEXT, DOC2VEC & K-MEANS CLUSTERING PIPELINE       ")
    print("==========================================================================")

    t_start = time.time()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    lab3_data_dir = os.path.abspath(os.path.join(base_dir, "..", "Lab3", "data"))
    results_dir = os.path.join(base_dir, "results")

    if not os.path.exists(lab3_data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found at '{lab3_data_dir}'. "
            "Please ensure Lab3/data/ contains train_1m.csv, dev.csv, and test.csv."
        )

    print(f"Using Lab 3 Dataset Directory: '{lab3_data_dir}'")
    print(f"Output Results Directory:      '{results_dir}'")

    # Run full Lab 5 pipeline
    results = run_full_lab5_experiments(
        data_dir=lab3_data_dir,
        results_dir=results_dir,
        vector_size=100,
        n_clusters=10,
        w2v_epochs=5,
        ft_epochs=5,
        d2v_epochs=10
    )

    total_time = time.time() - t_start
    print("\n==========================================================================")
    print(f"  LAB 5 PIPELINE COMPLETED SUCCESSFULLY IN {total_time / 60.0:.2f} MINUTES!  ")
    print("==========================================================================")


if __name__ == '__main__':
    main()
