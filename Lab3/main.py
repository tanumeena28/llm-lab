"""
Main Orchestrator Script for Lab 3.
Downloads dataset, sentence-tokenizes, creates length-stratified splits,
trains BPE and WordPiece tokenizers, runs scaling experiments (Task 3 & 4),
and generates visualization charts.
"""

import os
import sys
import time
import pandas as pd

# Add script directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import download_and_extract_sentences, create_length_stratified_splits
from experiments import (
    run_task3_data_size_experiments,
    run_task4_vocab_size_experiments,
    generate_qualitative_examples
)
from visualize import (
    plot_length_distributions,
    plot_task3_data_size_scaling,
    plot_task4_vocab_size_scaling
)


def main():
    print("==========================================================================")
    print("  LAB 3: INDICCORP V2 SUBWORD TOKENIZATION (BPE & WORDPIECE) PIPELINE    ")
    print("==========================================================================")
    
    t_start = time.time()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(base_dir, "results", "plots")
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n--- STEP 1: Data Downloading & Sentence Tokenization ---")
    sentences = download_and_extract_sentences(target_count=1050000)
    
    print("\n--- STEP 2: Creating Length-Stratified Splits (Train, Dev, Test) ---")
    splits = create_length_stratified_splits(sentences, dev_size=1000, test_size=1000, output_dir=data_dir)
    
    print("\n--- STEP 3: Generating Length Distribution Plot ---")
    plot_length_distributions(data_dir=data_dir, output_dir=plots_dir)
    
    print("\n--- STEP 4: Task 3 - Training Data Size Scaling Experiment (100k, 300k, 500k, 1M) ---")
    task3_results = run_task3_data_size_experiments(data_dir=data_dir, output_dir=results_dir, models_dir=models_dir)
    
    print("\n--- STEP 5: Task 4 - Vocabulary Size Scaling Experiment (20k, 30k, 50k) ---")
    task4_results = run_task4_vocab_size_experiments(data_dir=data_dir, output_dir=results_dir, models_dir=models_dir)
    
    print("\n--- STEP 6: Qualitative Tokenization Samples ---")
    generate_qualitative_examples(data_dir=data_dir, output_dir=results_dir, models_dir=models_dir)
    
    print("\n--- STEP 7: Plotting Task 3 & Task 4 Scaling Charts ---")
    plot_task3_data_size_scaling(results_csv=os.path.join(results_dir, "task3_data_size_results.csv"), output_dir=plots_dir)
    plot_task4_vocab_size_scaling(results_csv=os.path.join(results_dir, "task4_vocab_size_results.csv"), output_dir=plots_dir)
    
    total_time = time.time() - t_start
    print("\n==========================================================================")
    print(f"  LAB 3 PIPELINE COMPLETED SUCCESSFULLY IN {total_time / 60.0:.2f} MINUTES!  ")
    print("==========================================================================")


if __name__ == '__main__':
    main()
