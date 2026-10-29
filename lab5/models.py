"""
Model Training Module for Lab 5.
Trains Word2Vec, FastText, and Doc2Vec models using Gensim on 1,000,000 training sentences.
Follows Gensim official tutorials:
- Word2Vec: https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
- FastText: https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html
- Doc2Vec:  https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
"""

import os
import time
from typing import List, Optional
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def train_word2vec_model(
    corpus: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,  # 1 for Skip-gram, 0 for CBOW
    workers: int = 6,
    epochs: int = 5,
    save_path: Optional[str] = None
) -> Word2Vec:
    """
    Train Gensim Word2Vec model on pre-tokenized corpus sentences (with checkpoint loading).
    """
    if save_path and os.path.exists(save_path):
        try:
            print(f"\n[Word2Vec] Found existing model at '{save_path}'. Loading checkpoint...")
            model = Word2Vec.load(save_path)
            print(f"[Word2Vec] Loaded successfully! Vocabulary size: {len(model.wv):,} words.")
            return model
        except Exception as e:
            print(f"[Word2Vec] Checkpoint load failed ({e}). Re-training fresh model...")

    print(f"\n[Word2Vec] Training model (vector_size={vector_size}, sg={'Skip-gram' if sg==1 else 'CBOW'}, epochs={epochs}, workers={workers})...")
    t0 = time.time()
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs
    )
    t_el = time.time() - t0
    print(f"[Word2Vec] Training completed in {t_el:.2f}s. Vocabulary size: {len(model.wv):,} words.")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[Word2Vec] Saved model to '{save_path}'.")
    return model


def train_fasttext_model(
    corpus: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    workers: int = 6,
    epochs: int = 5,
    save_path: Optional[str] = None
) -> FastText:
    """
    Train Gensim FastText model on pre-tokenized corpus sentences (with checkpoint loading).
    """
    if save_path and os.path.exists(save_path):
        try:
            print(f"\n[FastText] Found existing model at '{save_path}'. Loading checkpoint...")
            model = FastText.load(save_path)
            print(f"[FastText] Loaded successfully! Vocabulary size: {len(model.wv):,} words.")
            return model
        except Exception as e:
            print(f"[FastText] Checkpoint load failed ({e}). Re-training fresh model...")

    print(f"\n[FastText] Training model (vector_size={vector_size}, sg={'Skip-gram' if sg==1 else 'CBOW'}, epochs={epochs}, workers={workers})...")
    t0 = time.time()
    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs
    )
    t_el = time.time() - t0
    print(f"[FastText] Training completed in {t_el:.2f}s. Vocabulary size: {len(model.wv):,} words.")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[FastText] Saved model to '{save_path}'.")
    return model


def train_doc2vec_model(
    corpus: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 6,
    epochs: int = 10,
    save_path: Optional[str] = None
) -> Doc2Vec:
    """
    Train Gensim Doc2Vec model on pre-tokenized corpus sentences wrapped as TaggedDocument (with checkpoint loading).
    """
    if save_path and os.path.exists(save_path):
        try:
            print(f"\n[Doc2Vec] Found existing model at '{save_path}'. Loading checkpoint...")
            model = Doc2Vec.load(save_path)
            print(f"[Doc2Vec] Loaded successfully! Vocabulary size: {len(model.wv):,} words.")
            return model
        except Exception as e:
            print(f"[Doc2Vec] Checkpoint load failed ({e}). Re-training fresh model...")

    print(f"\n[Doc2Vec] Tagging {len(corpus):,} sentences...")
    tagged_corpus = [
        TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(corpus)
    ]
    print(f"[Doc2Vec] Training model (vector_size={vector_size}, epochs={epochs}, workers={workers})...")
    t0 = time.time()
    model = Doc2Vec(
        documents=tagged_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    t_el = time.time() - t0
    print(f"[Doc2Vec] Training completed in {t_el:.2f}s. Vocabulary size: {len(model.wv):,} words.")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[Doc2Vec] Saved model to '{save_path}'.")
    return model
