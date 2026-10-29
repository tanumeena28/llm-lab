"""
Experiment Runner Module for Lab 5.
Executes complete pipeline:
1. Loads 1,000,000 training sentences, dev sentences, and test sentences.
2. Trains Word2Vec, FastText, and Doc2Vec models.
3. Runs K-Means clustering (K=10) and extracts 20 closest words per cluster centroid.
4. Performs nearest sentence retrieval on Dev and Test sets across all embedding representations.
5. Exports quantitative results CSVs.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any

from models import train_word2vec_model, train_fasttext_model, train_doc2vec_model
from clustering import perform_kmeans_word_clustering
from sentence_retrieval import build_sentence_matrices, find_closest_sentences
from visualize import plot_word_clusters_2d, plot_similarity_distributions, plot_mean_similarity_bar


def load_tokenized_corpus(csv_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Loads sentences from CSV and returns both raw text lines and tokenized word lists.
    """
    print(f"Loading CSV dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    raw_sentences = [str(s).strip() for s in df['sentence'].dropna()]
    tokenized = [s.split() for s in raw_sentences]
    print(f"Successfully loaded {len(tokenized):,} clean sentences.")
    return raw_sentences, tokenized


def run_full_lab5_experiments(
    data_dir: str,
    results_dir: str,
    vector_size: int = 100,
    n_clusters: int = 10,
    w2v_epochs: int = 5,
    ft_epochs: int = 5,
    d2v_epochs: int = 10
) -> Dict[str, Any]:
    """
    Main runner executing all tasks of Lab 5.
    """
    t_start = time.time()
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    models_dir = os.path.join(results_dir, "saved_models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Path setup
    train_path = os.path.join(data_dir, "train_1m.csv")
    dev_path = os.path.join(data_dir, "dev.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # Step 1: Load datasets
    print("\n==========================================================================")
    print("STEP 1: LOADING DATASETS (1,000,000 TRAIN, DEV, TEST)")
    print("==========================================================================")
    raw_train, train_tokens = load_tokenized_corpus(train_path)
    raw_dev, dev_tokens = load_tokenized_corpus(dev_path)
    raw_test, test_tokens = load_tokenized_corpus(test_path)

    # Step 2: Train Models (Task 1: Word2Vec, FastText, Doc2Vec)
    print("\n==========================================================================")
    print("STEP 2: TRAINING EMBEDDING MODELS (WORD2VEC, FASTTEXT, DOC2VEC)")
    print("==========================================================================")
    w2v_model = train_word2vec_model(
        train_tokens,
        vector_size=vector_size,
        epochs=w2v_epochs,
        save_path=os.path.join(models_dir, "word2vec.model")
    )
    ft_model = train_fasttext_model(
        train_tokens,
        vector_size=vector_size,
        epochs=ft_epochs,
        save_path=os.path.join(models_dir, "fasttext.model")
    )
    d2v_model = train_doc2vec_model(
        train_tokens,
        vector_size=vector_size,
        epochs=d2v_epochs,
        save_path=os.path.join(models_dir, "doc2vec.model")
    )

    # Step 3: K-Means Clustering on Words (Task 2a)
    print("\n==========================================================================")
    print(f"STEP 3: K-MEANS WORD CLUSTERING (K={n_clusters} CLUSTERS & TOP 20 WORDS)")
    print("==========================================================================")
    cluster_df, cluster_words_dict, kmeans_obj = perform_kmeans_word_clustering(
        w2v_model,
        n_clusters=n_clusters,
        top_n_words=25000,
        top_k_closest=20
    )
    cluster_csv_path = os.path.join(results_dir, "kmeans_clusters_top20.csv")
    cluster_df.to_csv(cluster_csv_path, index=False, encoding='utf-8-sig')
    print(f"Exported K-Means top 20 words per cluster to '{cluster_csv_path}'.")

    # Step 4: Sentence Representation & Nearest Sentence Retrieval (Task 2b)
    print("\n==========================================================================")
    print("STEP 4: SENTENCE RETRIEVAL & CLOSEST SENTENCE SEARCH (DEV & TEST SETS)")
    print("==========================================================================")
    
    print("\n--- Processing Validation Set (Dev - 1,000 sentences) ---")
    dev_mats = build_sentence_matrices(dev_tokens, w2v_model, ft_model, d2v_model)
    dev_results_dict = {}
    for method, mat in dev_mats.items():
        dev_results_dict[method] = find_closest_sentences(raw_dev, mat, method_name=method)

    print("\n--- Processing Test Set (1,000 sentences) ---")
    test_mats = build_sentence_matrices(test_tokens, w2v_model, ft_model, d2v_model)
    test_results_dict = {}
    for method, mat in test_mats.items():
        test_results_dict[method] = find_closest_sentences(raw_test, mat, method_name=method)

    # Combine into unified DataFrames and save
    all_dev_df = pd.concat(dev_results_dict.values(), ignore_index=True)
    all_test_df = pd.concat(test_results_dict.values(), ignore_index=True)

    dev_csv_path = os.path.join(results_dir, "dev_closest_sentences.csv")
    test_csv_path = os.path.join(results_dir, "test_closest_sentences.csv")
    all_dev_df.to_csv(dev_csv_path, index=False, encoding='utf-8-sig')
    all_test_df.to_csv(test_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved Dev closest sentences to '{dev_csv_path}'.")
    print(f"Saved Test closest sentences to '{test_csv_path}'.")

    # Calculate summary comparison statistics
    summary_records = []
    for split_name, res_dict in [("Validation", dev_results_dict), ("Test", test_results_dict)]:
        for method_name, df in res_dict.items():
            sims = df["Cosine_Similarity"].values
            summary_records.append({
                "Split": split_name,
                "Embedding_Method": method_name,
                "Sentence_Count": len(sims),
                "Mean_Cosine_Similarity": float(np.mean(sims)),
                "Std_Cosine_Similarity": float(np.std(sims)),
                "Median_Cosine_Similarity": float(np.median(sims)),
                "Min_Cosine_Similarity": float(np.min(sims)),
                "Max_Cosine_Similarity": float(np.max(sims))
            })

    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = os.path.join(results_dir, "sentence_similarity_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved sentence similarity summary to '{summary_csv_path}'.")

    # Step 5: Visualizations
    print("\n==========================================================================")
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("==========================================================================")
    plot_word_clusters_2d(
        cluster_words_dict,
        w2v_model,
        os.path.join(plots_dir, "word_clusters_pca.png")
    )
    plot_similarity_distributions(
        dev_results_dict,
        test_results_dict,
        os.path.join(plots_dir, "similarity_distribution.png")
    )
    plot_mean_similarity_bar(
        summary_df,
        os.path.join(plots_dir, "mean_similarity_comparison.png")
    )

    total_time = time.time() - t_start
    print(f"\n[Completed] Full pipeline executed in {total_time / 60.0:.2f} minutes.")

    return {
        "models": {"w2v": w2v_model, "ft": ft_model, "d2v": d2v_model},
        "clusters_df": cluster_df,
        "dev_results": dev_results_dict,
        "test_results": test_results_dict,
        "summary_df": summary_df
    }
