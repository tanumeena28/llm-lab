"""
K-Means Word Clustering Module for Lab 5.
Clusters word vectors extracted from trained embedding models (Word2Vec / FastText)
and extracts the top 20 words closest to each cluster centroid.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def perform_kmeans_word_clustering(
    model: Any,
    n_clusters: int = 10,
    top_n_words: int = 25000,
    top_k_closest: int = 20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[int, List[Tuple[str, float]]], KMeans]:
    """
    Performs K-Means clustering on the vocabulary word vectors of the trained model.
    Extracts top 20 words closest to each cluster centroid.

    Args:
        model: Trained Gensim Word2Vec or FastText model.
        n_clusters: Number of clusters K for KMeans.
        top_n_words: Number of most frequent words to include in clustering.
        top_k_closest: Number of closest words to extract per centroid (default=20).
        random_state: Seed for reproducibility.

    Returns:
        summary_df: DataFrame summarizing cluster contents & top 20 words.
        cluster_words_dict: Map of cluster_id -> list of (word, similarity_score).
        kmeans_model: Fitted sklearn KMeans object.
    """
    print(f"\n[K-Means] Extracting word vectors for top {top_n_words:,} words...")
    
    # Extract words sorted by frequency count in model vocabulary
    if hasattr(model.wv, "index_to_key"):
        vocab_words = model.wv.index_to_key[:top_n_words]
    else:
        vocab_words = list(model.wv.key_to_index.keys())[:top_n_words]

    word_vectors = np.array([model.wv[word] for word in vocab_words])
    print(f"[K-Means] Matrix shape for clustering: {word_vectors.shape}")

    print(f"[K-Means] Running K-Means with K={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(word_vectors)
    centroids = kmeans.cluster_centers_

    # Normalize vectors and centroids for cosine similarity calculation
    word_vecs_norm = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + 1e-9)
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)

    # Calculate cosine similarity matrix between centroids and word vectors (shape: n_clusters x N_words)
    centroid_word_sim = np.dot(centroids_norm, word_vecs_norm.T)

    cluster_records = []
    cluster_words_dict = {}

    print(f"\n--- Top {top_k_closest} Words Closest to Each Cluster Centroid ---")
    for k in range(n_clusters):
        # Get sorted word indices by descending cosine similarity to centroid k
        sorted_indices = np.argsort(centroid_word_sim[k])[::-1][:top_k_closest]
        
        top_words_with_sim = [
            (vocab_words[idx], float(centroid_word_sim[k][idx]))
            for idx in sorted_indices
        ]
        cluster_words_dict[k] = top_words_with_sim
        
        top_words_str = ", ".join([f"{w} ({sim:.3f})" for w, sim in top_words_with_sim[:10]])
        full_top20_str = ", ".join([w for w, _ in top_words_with_sim])

        cluster_records.append({
            "Cluster_ID": k,
            "Total_Words_In_Cluster": int(np.sum(cluster_labels == k)),
            "Top_20_Closest_Words": full_top20_str,
            "Sample_Top_10_With_Sim": top_words_str
        })

        print(f"Cluster {k:02d} ({np.sum(cluster_labels == k):,}: words): {top_words_str}")

    summary_df = pd.DataFrame(cluster_records)
    return summary_df, cluster_words_dict, kmeans
