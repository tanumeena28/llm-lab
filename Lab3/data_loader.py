"""
Data Loader & Sentence Tokenizer for IndicCorp v2 Dataset.
Handles paragraph streaming, sentence tokenization, length-distribution calculation,
and length-stratified split generation (Train, Dev, Test).
"""

import os
import re
import json
import random
import requests
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm


def sentence_tokenize(paragraph: str) -> List[str]:
    if not paragraph or not isinstance(paragraph, str):
        return []
    pattern = r'[।॥\.?!\n]'
    raw_sentences = re.split(pattern, paragraph)
    clean_sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        words = s_clean.split()
        if len(words) >= 3 and len(s_clean) >= 12 and len(words) <= 250:
            clean_sentences.append(s_clean)
    return clean_sentences


def download_and_extract_sentences(
    target_count: int = 1050000,
    url: str = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/hi-1.txt",
    output_cache: str = "Lab3/data/raw_sentences_1m.json"
) -> List[str]:
    os.makedirs(os.path.dirname(output_cache), exist_ok=True)
    if os.path.exists(output_cache):
        print(f"Loading raw cached sentences from '{output_cache}'...")
        with open(output_cache, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences):,} cached sentences.")
        if len(sentences) >= target_count:
            return sentences[:target_count]

    print(f"Streaming IndicCorp v2 corpus from {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    sentences = []
    buffer = ""
    pbar = tqdm(total=target_count, desc="Extracting Sentences")

    for chunk in response.iter_content(chunk_size=1024 * 1024, decode_unicode=True):
        if not chunk:
            continue
        buffer += chunk
        lines = buffer.split('\n')
        buffer = lines.pop()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            sents = sentence_tokenize(line_str)
            for s in sents:
                sentences.append(s)
                pbar.update(1)
                if len(sentences) >= target_count:
                    break
            if len(sentences) >= target_count:
                break
        if len(sentences) >= target_count:
            break

    pbar.close()
    print(f"Successfully extracted {len(sentences):,} clean sentences.")

    print(f"Caching extracted sentences to '{output_cache}'...")
    with open(output_cache, 'w', encoding='utf-8') as f:
        json.dump(sentences, f, ensure_ascii=False)

    return sentences


def create_length_stratified_splits(
    sentences: List[str],
    dev_size: int = 1000,
    test_size: int = 1000,
    output_dir: str = "Lab3/data",
    num_bins: int = 10,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    dev_path = os.path.join(output_dir, "dev.csv")
    test_path = os.path.join(output_dir, "test.csv")
    m1_path = os.path.join(output_dir, "train_1m.csv")

    if os.path.exists(dev_path) and os.path.exists(test_path) and os.path.exists(m1_path):
        print("Existing length-stratified split CSVs found on disk. Loading cached splits...")
        dev_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)
        splits = {
            'dev': dev_df,
            'test': test_df,
        }
        return splits

    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({'sentence': sentences})
    df['char_length'] = df['sentence'].apply(len)
    df['word_count'] = df['sentence'].apply(lambda s: len(s.split()))

    df['length_bin'] = pd.qcut(df['char_length'], q=num_bins, labels=False, duplicates='drop')

    print("\n--- Overall Length Statistics ---")
    print(f"Total Sentences: {len(df):,}")
    print(f"Mean Char Length: {df['char_length'].mean():.2f} ± {df['char_length'].std():.2f}")
    print(f"Mean Word Count:  {df['word_count'].mean():.2f} ± {df['word_count'].std():.2f}")

    dev_idx = []
    test_idx = []
    
    bin_counts = df['length_bin'].value_counts(normalize=True).sort_index()
    
    for bin_id, proportion in bin_counts.items():
        bin_df = df[df['length_bin'] == bin_id]
        n_dev = int(np.round(dev_size * proportion))
        n_test = int(np.round(test_size * proportion))

        available_indices = list(bin_df.index)
        random.shuffle(available_indices)

        selected_dev = available_indices[:n_dev]
        selected_test = available_indices[n_dev:n_dev + n_test]

        dev_idx.extend(selected_dev)
        test_idx.extend(selected_test)

    all_selected = set(dev_idx + test_idx)
    remaining_idx = [i for i in df.index if i not in all_selected]

    while len(dev_idx) < dev_size:
        dev_idx.append(remaining_idx.pop())
    while len(dev_idx) > dev_size:
        dev_idx.pop()

    while len(test_idx) < test_size:
        test_idx.append(remaining_idx.pop())
    while len(test_idx) > test_size:
        test_idx.pop()

    dev_df = df.loc[dev_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    
    used_set = set(dev_idx + test_idx)
    train_indices = [i for i in df.index if i not in used_set]
    train_full_df = df.loc[train_indices].reset_index(drop=True)

    print(f"\nCreated Splits:")
    print(f"Dev Set:   {len(dev_df):,} sentences (Mean Char Length: {dev_df['char_length'].mean():.2f})")
    print(f"Test Set:  {len(test_df):,} sentences (Mean Char Length: {test_df['char_length'].mean():.2f})")
    print(f"Train Set: {len(train_full_df):,} sentences (Mean Char Length: {train_full_df['char_length'].mean():.2f})")

    dev_df[['sentence', 'char_length', 'word_count']].to_csv(dev_path, index=False, encoding='utf-8')
    test_df[['sentence', 'char_length', 'word_count']].to_csv(test_path, index=False, encoding='utf-8')

    subsets = {
        '100k': 100000,
        '300k': 300000,
        '500k': 500000,
        '1m': min(1000000, len(train_full_df))
    }

    train_splits = {}
    for name, size in subsets.items():
        sub_df = train_full_df.iloc[:size].reset_index(drop=True)
        sub_path = os.path.join(output_dir, f"train_{name}.csv")
        sub_df[['sentence', 'char_length', 'word_count']].to_csv(sub_path, index=False, encoding='utf-8')
        train_splits[name] = sub_df
        print(f"Saved {name} Training Split: {len(sub_df):,} sentences to '{sub_path}'")

    splits_dict = {
        'dev': dev_df,
        'test': test_df,
        'train_full': train_full_df,
        **train_splits
    }
    
    return splits_dict
