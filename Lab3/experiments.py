"""
Experiments Module for Lab 3.
Evaluates BPE and WordPiece tokenizers across training data sizes (100k, 300k, 500k, 1M)
and vocabulary sizes (20k, 30k, 50k) on Development and Test datasets.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bpe import BPETokenizer
from wordpiece import WordPieceTokenizer


def evaluate_tokenizer(
    tokenizer: Any,
    sentences: List[str]
) -> Dict[str, float]:
    total_words = 0
    total_tokens = 0
    total_chars = 0
    unk_count = 0
    unk_token = tokenizer.unk_token
    
    for sentence in sentences:
        words = sentence.strip().split()
        chars = len(sentence)
        tokens = tokenizer.tokenize(sentence)
        
        total_words += len(words)
        total_tokens += len(tokens)
        total_chars += chars
        unk_count += tokens.count(unk_token)
        
    fertility = total_tokens / total_words if total_words > 0 else 0.0
    tokens_per_sentence = total_tokens / len(sentences) if len(sentences) > 0 else 0.0
    unk_rate = (unk_count / total_tokens * 100.0) if total_tokens > 0 else 0.0
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'total_words': total_words,
        'total_tokens': total_tokens,
        'fertility': round(fertility, 4),
        'tokens_per_sentence': round(tokens_per_sentence, 2),
        'unk_rate_percent': round(unk_rate, 4),
        'compression_ratio': round(compression_ratio, 4)
    }


def slice_tokenizer(tokenizer_obj: Any, target_vocab_size: int):
    """Slices a trained tokenizer object to a target vocabulary size."""
    if len(tokenizer_obj.vocab) <= target_vocab_size:
        return tokenizer_obj
        
    sliced = tokenizer_obj.__class__(vocab_size=target_vocab_size)
    sliced.special_tokens = tokenizer_obj.special_tokens
    
    sorted_items = sorted(tokenizer_obj.vocab.items(), key=lambda x: x[1])[:target_vocab_size]
    sliced.vocab = {tok: idx for idx, (tok, _) in enumerate(sorted_items)}
    sliced.inv_vocab = {idx: tok for tok, idx in sliced.vocab.items()}
    
    if hasattr(tokenizer_obj, 'merge_rules'):
        sliced.merge_rules = [m for m in tokenizer_obj.merge_rules if m[2] < target_vocab_size]
        sliced.merges = {(m[0], m[1]): m[2] for m in sliced.merge_rules}
        
    return sliced


def run_task3_data_size_experiments(
    data_dir: str = "Lab3/data",
    output_dir: str = "Lab3/results",
    models_dir: str = "Lab3/models",
    fixed_vocab_size: int = 5000
) -> pd.DataFrame:
    print("\n=======================================================")
    print(f"  RUNNING TASK 3: TRAINING DATA SIZE SCALING EXPERIMENT  ")
    print(f"  (Fixed Vocabulary Target: {fixed_vocab_size:,} subwords) ")
    print("=======================================================\n")
    
    dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    dev_sentences = dev_df['sentence'].tolist()
    test_sentences = test_df['sentence'].tolist()
    
    data_sizes = ['100k', '300k', '500k', '1m']
    results = []
    
    for size_name in data_sizes:
        train_path = os.path.join(data_dir, f"train_{size_name}.csv")
        print(f"\n---> Loading Training Data ({size_name.upper()}): {train_path}")
        train_df = pd.read_csv(train_path)
        train_sentences = train_df['sentence'].tolist()
        
        # 1. BPE Model
        bpe_model_path = os.path.join(models_dir, f"bpe_{size_name}_v{fixed_vocab_size}.json")
        bpe = BPETokenizer(vocab_size=fixed_vocab_size)
        if os.path.exists(bpe_model_path):
            print(f"Loading pre-trained BPE model from '{bpe_model_path}'...")
            bpe.load(bpe_model_path)
        else:
            t0 = time.time()
            bpe.train(train_sentences)
            bpe.save(bpe_model_path)
            print(f"BPE ({size_name}) trained in {time.time() - t0:.2f}s")
            
        bpe_dev = evaluate_tokenizer(bpe, dev_sentences)
        bpe_test = evaluate_tokenizer(bpe, test_sentences)
        
        results.append({
            'Algorithm': 'BPE',
            'Data_Size': size_name,
            'Vocab_Size': fixed_vocab_size,
            'Dev_Fertility': bpe_dev['fertility'],
            'Dev_Tokens_Per_Sent': bpe_dev['tokens_per_sentence'],
            'Dev_UNK_Rate_%': bpe_dev['unk_rate_percent'],
            'Dev_Compression': bpe_dev['compression_ratio'],
            'Test_Fertility': bpe_test['fertility'],
            'Test_Tokens_Per_Sent': bpe_test['tokens_per_sentence'],
            'Test_UNK_Rate_%': bpe_test['unk_rate_percent'],
            'Test_Compression': bpe_test['compression_ratio']
        })
        
        # 2. WordPiece Model
        wp_model_path = os.path.join(models_dir, f"wordpiece_{size_name}_v{fixed_vocab_size}.json")
        wp = WordPieceTokenizer(vocab_size=fixed_vocab_size)
        if os.path.exists(wp_model_path):
            print(f"Loading pre-trained WordPiece model from '{wp_model_path}'...")
            wp.load(wp_model_path)
        else:
            t0 = time.time()
            wp.train(train_sentences)
            wp.save(wp_model_path)
            print(f"WordPiece ({size_name}) trained in {time.time() - t0:.2f}s")
            
        wp_dev = evaluate_tokenizer(wp, dev_sentences)
        wp_test = evaluate_tokenizer(wp, test_sentences)
        
        results.append({
            'Algorithm': 'WordPiece',
            'Data_Size': size_name,
            'Vocab_Size': fixed_vocab_size,
            'Dev_Fertility': wp_dev['fertility'],
            'Dev_Tokens_Per_Sent': wp_dev['tokens_per_sentence'],
            'Dev_UNK_Rate_%': wp_dev['unk_rate_percent'],
            'Dev_Compression': wp_dev['compression_ratio'],
            'Test_Fertility': wp_test['fertility'],
            'Test_Tokens_Per_Sent': wp_test['tokens_per_sentence'],
            'Test_UNK_Rate_%': wp_test['unk_rate_percent'],
            'Test_Compression': wp_test['compression_ratio']
        })
        
    df_results = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, "task3_data_size_results.csv")
    df_results.to_csv(res_path, index=False)
    print(f"\nTask 3 Results Saved to '{res_path}'")
    return df_results


def run_task4_vocab_size_experiments(
    data_dir: str = "Lab3/data",
    output_dir: str = "Lab3/results",
    models_dir: str = "Lab3/models",
    fixed_data_size: str = "1m"
) -> pd.DataFrame:
    print("\n=======================================================")
    print(f"  RUNNING TASK 4: VOCABULARY SIZE SCALING EXPERIMENT   ")
    print(f"  (Fixed Training Corpus Size: 1,000,000 Sentences)      ")
    print("=======================================================\n")
    
    dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    dev_sentences = dev_df['sentence'].tolist()
    test_sentences = test_df['sentence'].tolist()
    
    train_path = os.path.join(data_dir, f"train_{fixed_data_size}.csv")
    train_df = pd.read_csv(train_path)
    train_sentences = train_df['sentence'].tolist()
    
    vocab_sizes = [2000, 3500, 5000]
    results = []
    
    # Train 5000 base models on 1M dataset
    bpe_base_path = os.path.join(models_dir, f"bpe_{fixed_data_size}_v5000.json")
    bpe_base = BPETokenizer(vocab_size=5000)
    if os.path.exists(bpe_base_path):
        bpe_base.load(bpe_base_path)
    else:
        print("Training base BPE model on 1M sentences...")
        bpe_base.train(train_sentences)
        bpe_base.save(bpe_base_path)
        
    wp_base_path = os.path.join(models_dir, f"wordpiece_{fixed_data_size}_v5000.json")
    wp_base = WordPieceTokenizer(vocab_size=5000)
    if os.path.exists(wp_base_path):
        wp_base.load(wp_base_path)
    else:
        print("Training base WordPiece model on 1M sentences...")
        wp_base.train(train_sentences)
        wp_base.save(wp_base_path)

    for v_size in vocab_sizes:
        print(f"---> Evaluating Vocabulary Size {v_size:,}")
        
        bpe = slice_tokenizer(bpe_base, v_size)
        bpe_dev = evaluate_tokenizer(bpe, dev_sentences)
        bpe_test = evaluate_tokenizer(bpe, test_sentences)
        
        results.append({
            'Algorithm': 'BPE',
            'Data_Size': fixed_data_size,
            'Vocab_Size': v_size,
            'Dev_Fertility': bpe_dev['fertility'],
            'Dev_Tokens_Per_Sent': bpe_dev['tokens_per_sentence'],
            'Dev_UNK_Rate_%': bpe_dev['unk_rate_percent'],
            'Dev_Compression': bpe_dev['compression_ratio'],
            'Test_Fertility': bpe_test['fertility'],
            'Test_Tokens_Per_Sent': bpe_test['tokens_per_sentence'],
            'Test_UNK_Rate_%': bpe_test['unk_rate_percent'],
            'Test_Compression': bpe_test['compression_ratio']
        })
        
        wp = slice_tokenizer(wp_base, v_size)
        wp_dev = evaluate_tokenizer(wp, dev_sentences)
        wp_test = evaluate_tokenizer(wp, test_sentences)
        
        results.append({
            'Algorithm': 'WordPiece',
            'Data_Size': fixed_data_size,
            'Vocab_Size': v_size,
            'Dev_Fertility': wp_dev['fertility'],
            'Dev_Tokens_Per_Sent': wp_dev['tokens_per_sentence'],
            'Dev_UNK_Rate_%': wp_dev['unk_rate_percent'],
            'Dev_Compression': wp_dev['compression_ratio'],
            'Test_Fertility': wp_test['fertility'],
            'Test_Tokens_Per_Sent': wp_test['tokens_per_sentence'],
            'Test_UNK_Rate_%': wp_test['unk_rate_percent'],
            'Test_Compression': wp_test['compression_ratio']
        })
        
    df_results = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, "task4_vocab_size_results.csv")
    df_results.to_csv(res_path, index=False)
    print(f"\nTask 4 Results Saved to '{res_path}'")
    return df_results


def generate_qualitative_examples(
    data_dir: str = "Lab3/data",
    output_dir: str = "Lab3/results",
    models_dir: str = "Lab3/models"
):
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_df['word_cnt'] = test_df['sentence'].apply(lambda s: len(s.split()))
    
    short_sent = test_df[test_df['word_cnt'].between(4, 7)]['sentence'].iloc[0]
    med_sent = test_df[test_df['word_cnt'].between(12, 18)]['sentence'].iloc[0]
    long_sent = test_df[test_df['word_cnt'] >= 25]['sentence'].iloc[0]
    
    sample_sents = [
        ("Short", short_sent),
        ("Medium", med_sent),
        ("Long", long_sent)
    ]
    
    records = []
    
    configs = [
        ("BPE", "100k", 5000, f"bpe_100k_v5000.json"),
        ("BPE", "1m", 5000, f"bpe_1m_v5000.json"),
        ("BPE", "1m", 2000, f"bpe_1m_v5000.json"),
        ("WordPiece", "100k", 5000, f"wordpiece_100k_v5000.json"),
        ("WordPiece", "1m", 5000, f"wordpiece_1m_v5000.json"),
        ("WordPiece", "1m", 2000, f"wordpiece_1m_v5000.json"),
    ]
    
    for label, sent in sample_sents:
        for algo, dsize, vsize, mfile in configs:
            mpath = os.path.join(models_dir, mfile)
            if not os.path.exists(mpath):
                continue
            if algo == "BPE":
                tok_model = BPETokenizer(vocab_size=vsize)
            else:
                tok_model = WordPieceTokenizer(vocab_size=vsize)
            tok_model.load(mpath)
            tok_model = slice_tokenizer(tok_model, vsize)
            
            tokens = tok_model.tokenize(sent)
            records.append({
                'Category': label,
                'Algorithm': algo,
                'Data_Size': dsize,
                'Vocab_Size': vsize,
                'Sentence': sent,
                'Tokens': " | ".join(tokens),
                'Num_Tokens': len(tokens)
            })
            
    df_qual = pd.DataFrame(records)
    qpath = os.path.join(output_dir, "qualitative_tokenization_samples.csv")
    df_qual.to_csv(qpath, index=False, encoding='utf-8')
    print(f"Qualitative tokenization samples saved to '{qpath}'")


if __name__ == '__main__':
    t3 = run_task3_data_size_experiments()
    t4 = run_task4_vocab_size_experiments()
    generate_qualitative_examples()
