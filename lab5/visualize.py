"""
Visualization Module for Lab 5.
Generates publication-grade visual charts:
1. t-SNE 2D Projection of Word Embedding Clusters & Centroids.
2. Nearest Sentence Cosine Similarity Score Distribution (Word2Vec vs FastText vs Doc2Vec).
3. Mean Closest Sentence Similarity Comparison Bar Chart across Dev and Test sets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Configure matplotlib style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_word_clusters_2d(
    cluster_words_dict: Dict[int, List[tuple]],
    model: Any,
    output_path: str,
    n_words_per_cluster: int = 15
):
    """
    Plots a 2D PCA/t-SNE visualization of word clusters and their words.
    """
    print(f"Generating 2D word clusters plot to '{output_path}'...")
    all_words = []
    labels = []

    for cluster_id, word_sim_tuples in cluster_words_dict.items():
        for word, _ in word_sim_tuples[:n_words_per_cluster]:
            if word in model.wv:
                all_words.append(word)
                labels.append(cluster_id)

    word_vecs = np.array([model.wv[w] for w in all_words])
    
    # Use PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(word_vecs)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    cmap = plt.get_cmap('tab10')

    for cluster_id in sorted(set(labels)):
        mask = np.array(labels) == cluster_id
        cluster_coords = coords[mask]
        ax.scatter(
            cluster_coords[:, 0],
            cluster_coords[:, 1],
            color=cmap(cluster_id % 10),
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            edgecolors='k',
            s=80
        )

    # Annotate a subset of words
    for idx, (word, cid) in enumerate(zip(all_words, labels)):
        if idx % 3 == 0:  # Annotate every 3rd word to avoid clutter
            ax.annotate(
                word,
                (coords[idx, 0], coords[idx, 1]),
                fontsize=8,
                alpha=0.85,
                xytext=(3, 3),
                textcoords='offset points'
            )

    ax.set_title("2D PCA Projection of K-Means Word Embedding Clusters", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.legend(title="Clusters", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved word clusters plot to '{output_path}'.")


def plot_similarity_distributions(
    dev_results: Dict[str, pd.DataFrame],
    test_results: Dict[str, pd.DataFrame],
    output_path: str
):
    """
    Plots KDE / Histogram distribution of closest sentence similarity scores.
    """
    print(f"Generating similarity distribution plot to '{output_path}'...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)

    colors = {"Word2Vec": "#1f77b4", "FastText": "#ff7f0e", "Doc2Vec": "#2ca02c"}

    # Dev Plot
    for method, df in dev_results.items():
        sns.kdeplot(
            df["Cosine_Similarity"],
            ax=axes[0],
            label=method,
            color=colors.get(method, "#333333"),
            fill=True,
            alpha=0.25,
            linewidth=2
        )
    axes[0].set_title("Validation Set (Dev) - Closest Sentence Similarity", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Cosine Similarity", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].legend()

    # Test Plot
    for method, df in test_results.items():
        sns.kdeplot(
            df["Cosine_Similarity"],
            ax=axes[1],
            label=method,
            color=colors.get(method, "#333333"),
            fill=True,
            alpha=0.25,
            linewidth=2
        )
    axes[1].set_title("Test Set - Closest Sentence Similarity", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Cosine Similarity", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity distribution plot to '{output_path}'.")


def plot_mean_similarity_bar(
    summary_df: pd.DataFrame,
    output_path: str
):
    """
    Plots bar chart comparing mean similarity scores across methods and splits.
    """
    print(f"Generating mean similarity bar plot to '{output_path}'...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    sns.barplot(
        data=summary_df,
        x="Embedding_Method",
        y="Mean_Cosine_Similarity",
        hue="Split",
        palette="viridis",
        ax=ax
    )

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2., height / 2.0),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold'
            )

    ax.set_title("Average Closest Sentence Similarity by Embedding Method", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Embedding Method", fontsize=11)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved mean similarity bar plot to '{output_path}'.")
