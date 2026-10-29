"""
Sentence Representation & Nearest Sentence Retrieval Module for Lab 5.
Vectorizes validation and test set sentences using:
1. Word2Vec Mean Embeddings
2. FastText Mean Embeddings
3. Doc2Vec Inferred Embeddings
And retrieves the closest matching sentence for each sentence in the dataset.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from scipy.spatial.distance import cdist


def sentence_to_w2v_vector(tokens: List[str], model: Any, vector_size: int = 100) -> np.ndarray:
    """
    Computes average Word2Vec vector for a tokenized sentence.
    """
    valid_vecs = [model.wv[w] for w in tokens if w in model.wv]
    if len(valid_vecs) == 0:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(valid_vecs, axis=0)


def sentence_to_fasttext_vector(tokens: List[str], model: Any, vector_size: int = 100) -> np.ndarray:
    """
    Computes average FastText vector for a tokenized sentence (utilizes subwords for OOV).
    """
    if len(tokens) == 0:
        return np.zeros(vector_size, dtype=np.float32)
    vecs = [model.wv[w] for w in tokens]
    return np.mean(vecs, axis=0)


def sentence_to_doc2vec_vector(tokens: List[str], model: Any) -> np.ndarray:
    """
    Infers Doc2Vec vector for a tokenized sentence.
    """
    if len(tokens) == 0:
        return np.zeros(model.vector_size, dtype=np.float32)
    return model.infer_vector(tokens)


def build_sentence_matrices(
    corpus: List[List[str]],
    w2v_model: Any,
    ft_model: Any,
    d2v_model: Any
) -> Dict[str, np.ndarray]:
    """
    Converts list of sentence token lists into matrix representations for W2V, FastText, and Doc2Vec.
    """
    vector_size = w2v_model.vector_size
    n_sents = len(corpus)
    
    print(f"Vectorizing {n_sents:,} sentences...")
    
    w2v_mat = np.zeros((n_sents, vector_size), dtype=np.float32)
    ft_mat = np.zeros((n_sents, vector_size), dtype=np.float32)
    d2v_mat = np.zeros((n_sents, vector_size), dtype=np.float32)

    for i, tokens in enumerate(corpus):
        w2v_mat[i] = sentence_to_w2v_vector(tokens, w2v_model, vector_size)
        ft_mat[i] = sentence_to_fasttext_vector(tokens, ft_model, vector_size)
        d2v_mat[i] = sentence_to_doc2vec_vector(tokens, d2v_model)

    return {
        "Word2Vec": w2v_mat,
        "FastText": ft_mat,
        "Doc2Vec": d2v_mat
    }


def find_closest_sentences(
    sentences_raw: List[str],
    embeddings_matrix: np.ndarray,
    method_name: str
) -> pd.DataFrame:
    """
    For each sentence in embeddings_matrix, finds the single closest sentence (excluding itself)
    using Cosine Similarity.

    Args:
        sentences_raw: List of original raw sentence strings.
        embeddings_matrix: Shape (N_sentences, Vector_Dim).
        method_name: Name of the embedding method ('Word2Vec', 'FastText', 'Doc2Vec').

    Returns:
        DataFrame containing original sentence, closest sentence index, closest sentence text, and similarity score.
    """
    n_sents = len(sentences_raw)
    
    # L2 Normalization for Cosine Similarity calculation via Dot Product
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_matrix = embeddings_matrix / norms

    # Pairwise Cosine Similarity matrix (N x N)
    sim_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    # Fill diagonal with -infinity to prevent self-matching
    np.fill_diagonal(sim_matrix, -np.inf)

    # Argmax along rows to find closest sentence index for each sentence
    best_indices = np.argmax(sim_matrix, axis=1)
    best_scores = np.max(sim_matrix, axis=1)

    records = []
    for i in range(n_sents):
        closest_idx = int(best_indices[i])
        score = float(best_scores[i])
        records.append({
            "Sentence_Index": i,
            "Original_Sentence": sentences_raw[i],
            "Closest_Sentence_Index": closest_idx,
            "Closest_Sentence": sentences_raw[closest_idx],
            "Cosine_Similarity": score,
            "Method": method_name
        })

    df = pd.DataFrame(records)
    return df
