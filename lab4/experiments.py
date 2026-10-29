"""
Experiments runner and hyperparameter tuning module for Lab 4.
Loads data, tunes smoothing parameters on Dev set, evaluates Test set perplexities,
and exports results CSVs.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from ngram_model import NGramLM


def load_dataset(csv_path: str) -> List[List[str]]:
    """
    Load sentences from CSV file and tokenize into list of word lists.
    """
    print(f"Loading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    corpus = [str(s).strip().split() for s in df['sentence'].dropna()]
    print(f"Loaded {len(corpus):,} sentences.")
    return corpus


def tune_add_k_parameter(
    model: NGramLM,
    dev_corpus: List[List[str]],
    k_candidates: List[float] = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
) -> Tuple[Dict[int, float], pd.DataFrame]:
    """
    Perform grid search for optimal Add-K parameter K (0 < K < 1) on Dev set for N=1, 2, 3, 4.
    """
    print("\n--- Tuning Add-K Smoothing Parameter (K) on Dev Set ---")
    tuning_records = []
    best_k_dict = {}

    for order in range(1, 5):
        best_k = k_candidates[0]
        best_ppl = float("inf")
        print(f"Tuning K for N={order}...")
        
        for k in k_candidates:
            ppl = model.compute_perplexity(dev_corpus, order=order, smoothing="add_k", k=k)
            tuning_records.append({
                "Order": f"N={order}",
                "N_val": order,
                "K": k,
                "Dev_Perplexity": ppl
            })
            print(f"  N={order}, K={k:0.3f} -> Dev PPL: {ppl:0.2f}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_k = k
                
        best_k_dict[order] = best_k
        print(f"--> Best K for N={order}: {best_k} (Dev PPL: {best_ppl:0.2f})")

    tuning_df = pd.DataFrame(tuning_records)
    return best_k_dict, tuning_df


def tune_interpolation_lambdas(
    model: NGramLM,
    dev_corpus: List[List[str]]
) -> Tuple[Dict[int, List[float]], pd.DataFrame]:
    """
    Tune interpolation constants lambda on Dev set for N=1, 2, 3, 4.
    For N=1: P(w) = lambda * P_MLE(w) + (1 - lambda) * (1 / V)
    For N > 1: Grid search over weight distributions.
    """
    print("\n--- Tuning Interpolation Lambdas on Dev Set ---")
    best_lambdas_dict = {}
    tuning_records = []

    # Order 1 tuning (single lambda)
    print("Tuning lambda for N=1 (Unigram Interpolation)...")
    best_lam1 = 0.9
    best_ppl1 = float("inf")
    for lam in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        ppl = model.compute_perplexity(dev_corpus, order=1, smoothing="interpolated", lambdas=[lam])
        tuning_records.append({"Order": "N=1", "Lambdas": str([lam]), "Dev_Perplexity": ppl})
        if ppl < best_ppl1:
            best_ppl1 = ppl
            best_lam1 = lam
    best_lambdas_dict[1] = [best_lam1]
    print(f"--> Best Lambda for N=1: {[best_lam1]} (Dev PPL: {best_ppl1:0.2f})")

    # Order 2 tuning (lambda2, lambda1)
    print("Tuning lambdas for N=2 (Bigram Interpolation)...")
    best_lam2 = [0.8, 0.2]
    best_ppl2 = float("inf")
    for l2 in [0.5, 0.7, 0.8, 0.9, 0.95]:
        l1 = round(1.0 - l2, 3)
        lams = [l2, l1]
        ppl = model.compute_perplexity(dev_corpus, order=2, smoothing="interpolated", lambdas=lams)
        tuning_records.append({"Order": "N=2", "Lambdas": str(lams), "Dev_Perplexity": ppl})
        if ppl < best_ppl2:
            best_ppl2 = ppl
            best_lam2 = lams
    best_lambdas_dict[2] = best_lam2
    print(f"--> Best Lambdas for N=2: {best_lam2} (Dev PPL: {best_ppl2:0.2f})")

    # Order 3 tuning (lambda3, lambda2, lambda1)
    print("Tuning lambdas for N=3 (Trigram Interpolation)...")
    best_lam3 = [0.7, 0.2, 0.1]
    best_ppl3 = float("inf")
    candidate_lams_3 = [
        [0.8, 0.15, 0.05],
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.5, 0.35, 0.15]
    ]
    for lams in candidate_lams_3:
        ppl = model.compute_perplexity(dev_corpus, order=3, smoothing="interpolated", lambdas=lams)
        tuning_records.append({"Order": "N=3", "Lambdas": str(lams), "Dev_Perplexity": ppl})
        if ppl < best_ppl3:
            best_ppl3 = ppl
            best_lam3 = lams
    best_lambdas_dict[3] = best_lam3
    print(f"--> Best Lambdas for N=3: {best_lam3} (Dev PPL: {best_ppl3:0.2f})")

    # Order 4 tuning (lambda4, lambda3, lambda2, lambda1)
    print("Tuning lambdas for N=4 (Quadram Interpolation)...")
    best_lam4 = [0.6, 0.25, 0.1, 0.05]
    best_ppl4 = float("inf")
    candidate_lams_4 = [
        [0.7, 0.2, 0.07, 0.03],
        [0.6, 0.25, 0.1, 0.05],
        [0.5, 0.3, 0.15, 0.05],
        [0.4, 0.3, 0.2, 0.1]
    ]
    for lams in candidate_lams_4:
        ppl = model.compute_perplexity(dev_corpus, order=4, smoothing="interpolated", lambdas=lams)
        tuning_records.append({"Order": "N=4", "Lambdas": str(lams), "Dev_Perplexity": ppl})
        if ppl < best_ppl4:
            best_ppl4 = ppl
            best_lam4 = lams
    best_lambdas_dict[4] = best_lam4
    print(f"--> Best Lambdas for N=4: {best_lam4} (Dev PPL: {best_ppl4:0.2f})")

    tuning_df = pd.DataFrame(tuning_records)
    return best_lambdas_dict, tuning_df


def run_full_lab4_evaluation(
    data_dir: str = "Lab3/data",
    results_dir: str = "lab4/results"
) -> Dict[str, pd.DataFrame]:
    """
    Run complete Lab 4 experiment pipeline across 1M dataset and test set.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train_1m.csv")
    dev_path = os.path.join(data_dir, "dev.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_corpus = load_dataset(train_path)
    dev_corpus = load_dataset(dev_path)
    test_corpus = load_dataset(test_path)

    # Initialize and fit N-gram Language Model (max N=4)
    lm = NGramLM(n=4, min_freq=2)
    lm.fit(train_corpus)

    # Step 1: Hyperparameter Tuning on Dev Set
    best_k_dict, add_k_df = tune_add_k_parameter(lm, dev_corpus)
    best_lambdas_dict, interp_df = tune_interpolation_lambdas(lm, dev_corpus)

    add_k_df.to_csv(os.path.join(results_dir, "add_k_dev_tuning.csv"), index=False)
    interp_df.to_csv(os.path.join(results_dir, "interpolation_dev_tuning.csv"), index=False)

    # Step 2: Final Test Set Evaluation Across All 20 Combinations (4 N-gram orders x 5 Smoothing methods)
    print("\n==========================================================================")
    print("  RUNNING FINAL TEST SET EVALUATION ACROSS ALL 20 MODEL COMBINATIONS      ")
    print("==========================================================================")

    smoothing_methods = [
        ("Unsmoothed", "unsmoothed"),
        ("Add-1 Smoothing", "add_one"),
        ("Add-K Smoothing", "add_k"),
        ("Interpolated Smoothing", "interpolated"),
        ("Kneser-Ney Smoothing (d=0.75)", "kneser_ney")
    ]

    orders = [
        (1, "Unigram (N=1)"),
        (2, "Bigram (N=2)"),
        (3, "Trigram (N=3)"),
        (4, "Quadram (N=4)")
    ]

    results_table = []
    
    # Pivot matrix for easy reporting: rows = Smoothing Methods, cols = N-gram Orders
    matrix_data = {
        "Smoothing Method": [name for name, _ in smoothing_methods]
    }

    for order_num, order_name in orders:
        matrix_data[order_name] = []
        for method_name, method_key in smoothing_methods:
            print(f"Evaluating {order_name} with {method_name}...")
            
            kwargs = {}
            if method_key == "add_k":
                kwargs["k"] = best_k_dict[order_num]
            elif method_key == "interpolated":
                kwargs["lambdas"] = best_lambdas_dict[order_num]
            elif method_key == "kneser_ney":
                kwargs["d"] = 0.75

            ppl = lm.compute_perplexity(test_corpus, order=order_num, smoothing=method_key, **kwargs)
            
            ppl_str = "INF" if np.isinf(ppl) else f"{ppl:0.2f}"
            print(f"  --> {order_name} + {method_name} -> Test PPL: {ppl_str}")

            results_table.append({
                "Order_Num": order_num,
                "Order": order_name,
                "Smoothing_Method": method_name,
                "Smoothing_Key": method_key,
                "Perplexity": ppl,
                "Perplexity_Display": ppl_str
            })
            
            matrix_data[order_name].append(ppl_str)

    results_df = pd.DataFrame(results_table)
    results_df.to_csv(os.path.join(results_dir, "ppl_comparison.csv"), index=False)

    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.to_csv(os.path.join(results_dir, "ppl_matrix.csv"), index=False)

    print("\n--- Final Test Set Perplexity Matrix ---")
    print(matrix_df.to_string(index=False))

    return {
        "results_df": results_df,
        "matrix_df": matrix_df,
        "add_k_df": add_k_df,
        "interp_df": interp_df
    }


if __name__ == '__main__':
    run_full_lab4_evaluation()
