import os
import re
import sys
import json
import random
import collections
import numpy as np
import pandas as pd
import requests

# Reconfigure stdout to use UTF-8 on Windows to avoid print encoding errors
sys.stdout.reconfigure(encoding='utf-8')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration for the 23 languages and their files in IndicCorpV2
LANG_FILES = {
    "asm_Beng": "as.txt",
    "ben_Beng": "bn.txt",
    "brx_Deva": "bd.txt",
    "doi_Deva": "dg.txt",
    "gom_Deva": "gom.txt",
    "guj_Gujr": "gu.txt",
    "hin_Deva": "hi-1.txt",
    "kan_Knda": "kn.txt",
    "kas_Arab": "ks.txt",
    "mai_Deva": "mai.txt",
    "mal_Mlym": "ml.txt",
    "mar_Deva": "mr.txt",
    "mni_Mtei": "mni.txt",
    "npi_Deva": "ne.txt",
    "ory_Orya": "or.txt",
    "pan_Guru": "pa.txt",
    "san_Deva": "sa.txt",
    "snd_Deva": "sd.txt",
    "tam_Taml": "ta.txt",
    "tel_Telu": "te.txt",
    "urd_Arab": "ur.txt",
    "khasi": "kha.txt",
    "santhali": "sat.txt"
}

BASE_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data/"
CACHE_FILE = "../raw_sentences_cache.json" if os.path.exists("../raw_sentences_cache.json") else "raw_sentences_cache.json"

# ==========================================
# 1. Dataset Construction & Tokenization
# ==========================================

def clean_sentence(text):
    text = text.strip()
    if len(text) < 10:
        return None
    return text

def download_sentences(lang, filename, target_count=1000):
    url = BASE_URL + filename
    sentences = []
    print(f"Streaming {lang} from {filename}...", flush=True)
    split_pattern = re.compile(r'[।॥\.?!\u06d4]\s*')
    
    response = requests.get(url, stream=True, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {filename}: HTTP {response.status_code}")
    
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        try:
            line = raw_line.decode('utf-8', errors='ignore')
        except Exception:
            continue
            
        parts = split_pattern.split(line)
        for part in parts:
            s = clean_sentence(part)
            if s:
                sentences.append(s)
                if len(sentences) >= target_count:
                    break
        if len(sentences) >= target_count:
            break
            
    response.close()
    return sentences[:target_count]

def get_dataset():
    if os.path.exists(CACHE_FILE):
        print(f"Loading raw dataset from local cache: {CACHE_FILE}...", flush=True)
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print("Local cache not found. Starting direct downloads from Hugging Face...", flush=True)
    dataset = {}
    for i, (lang, filename) in enumerate(LANG_FILES.items(), 1):
        print(f"[{i}/{len(LANG_FILES)}] Downloading {lang}...", flush=True)
        try:
            sentences = download_sentences(lang, filename, target_count=1000)
            dataset[lang] = sentences
        except Exception as e:
            print(f"Error downloading {lang}: {e}. Retrying...", flush=True)
            sentences = download_sentences(lang, filename, target_count=1000)
            dataset[lang] = sentences
                
    with open("raw_sentences_cache.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    return dataset

def split_and_stratify(dataset):
    print("Splitting dataset (80% Train, 10% Val, 10% Test) using stratified sampling...", flush=True)
    train_data = []
    val_data = []
    test_data = []
    
    for lang, sentences in sorted(dataset.items()):
        shuffled = list(sentences)
        # Fixed seed per language for identical splits across runs
        rnd = random.Random(42 + hash(lang) % 10000)
        rnd.shuffle(shuffled)
        
        train_end = 800
        val_end = 900
        
        train_seq = shuffled[:train_end]
        val_seq = shuffled[train_end:val_end]
        test_seq = shuffled[val_end:]
        
        for s in train_seq:
            train_data.append({"text": s, "label": lang})
        for s in val_seq:
            val_data.append({"text": s, "label": lang})
        for s in test_seq:
            test_data.append({"text": s, "label": lang})
            
    random.Random(42).shuffle(train_data)
    random.Random(42).shuffle(val_data)
    random.Random(42).shuffle(test_data)
    
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_test = pd.DataFrame(test_data)
    
    return df_train, df_val, df_test

# ==========================================
# 2. Custom Configurable TF-IDF Vectorizer
# ==========================================

def extract_ngrams(text):
    words = text.lower().split()
    ngrams = []
    
    # Word unigrams
    for w in words:
        ngrams.append(f"w1:{w}")
        
    # Word bigrams
    for i in range(len(words) - 1):
        ngrams.append(f"w2:{words[i]}_{words[i+1]}")
        
    # Character n-grams (2, 3, and 4)
    chars = text.lower()
    for i in range(len(chars) - 1):
        ngrams.append(f"c2:{chars[i:i+2]}")
    for i in range(len(chars) - 2):
        ngrams.append(f"c3:{chars[i:i+3]}")
    for i in range(len(chars) - 3):
        ngrams.append(f"c4:{chars[i:i+4]}")
        
    return ngrams

class CustomTFIDFVectorizer:
    """
    Custom TF-IDF Vectorizer supporting:
    TF Types:
      - 'unnormalized': Raw n-gram frequency count
      - 'total_words': Normalized by total number of words in sentence
      - 'max_freq': Normalized by frequency of the most frequent word in sentence
    IDF Types:
      - 'unnormalized': Standard smooth IDF based on total document count N
      - 'normalized': Normalized IDF based on most frequent word across sentences (df_max)
    """
    def __init__(self, max_features=20000, tf_type="unnormalized", idf_type="unnormalized"):
        self.max_features = max_features
        self.tf_type = tf_type
        self.idf_type = idf_type
        self.vocabulary = {}
        self.idf = None
        
    def fit(self, parsed_docs):
        ngram_counts = collections.Counter()
        doc_counts = collections.Counter()
        
        for ngrams in parsed_docs:
            ngram_counts.update(ngrams)
            doc_counts.update(set(ngrams))
            
        most_common = [item[0] for item in ngram_counts.most_common(self.max_features)]
        self.vocabulary = {ng: i for i, ng in enumerate(most_common)}
        
        N = len(parsed_docs)
        df_max = max(doc_counts.values()) if len(doc_counts) > 0 else 1
        
        self.idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for ng, idx in self.vocabulary.items():
            df = doc_counts[ng]
            if self.idf_type == "normalized":
                # IDF normalized based on the most frequent word across sentences (df_max)
                self.idf[idx] = np.log((1 + df_max) / (1 + df)) + 1
            else:
                # Standard unnormalized IDF based on N
                self.idf[idx] = np.log((1 + N) / (1 + df)) + 1
                
        return self
        
    def transform(self, parsed_docs, raw_documents):
        N_doc = len(parsed_docs)
        V = len(self.vocabulary)
        X = np.zeros((N_doc, V), dtype=np.float32)
        
        for doc_idx, (ngrams, doc) in enumerate(zip(parsed_docs, raw_documents)):
            counts = collections.Counter(ngrams)
            if not counts:
                continue
                
            total_words = max(len(doc.split()), 1)
            max_freq = max(counts.values())
            
            for ng, count in counts.items():
                if ng in self.vocabulary:
                    f_idx = self.vocabulary[ng]
                    
                    if self.tf_type == "total_words":
                        tf_val = count / total_words
                    elif self.tf_type == "max_freq":
                        tf_val = count / max_freq
                    else:
                        tf_val = count
                        
                    X[doc_idx, f_idx] = tf_val
                    
        # Apply IDF weighting
        X *= self.idf
        
        # Row-wise L2 normalization for cosine scale invariant representation
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        X /= row_norms
        
        return X
        
    def fit_transform(self, parsed_docs, raw_documents):
        return self.fit(parsed_docs).transform(parsed_docs, raw_documents)

# ==========================================
# 3. Custom Softmax Regression (Logistic Regression)
# ==========================================

class ScratchSoftmaxRegression:
    def __init__(self, lr=0.05, l2_reg=1e-4, epochs=25, batch_size=256):
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        
    def _softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
    def fit(self, X, y):
        N, F = X.shape
        C = int(np.max(y)) + 1
        
        self.weights = np.zeros((C, F), dtype=np.float32)
        self.bias = np.zeros(C, dtype=np.float32)
        
        y_one_hot = np.zeros((N, C), dtype=np.float32)
        y_one_hot[np.arange(N), y] = 1.0
        
        m_w = np.zeros_like(self.weights)
        v_w = np.zeros_like(self.weights)
        m_b = np.zeros_like(self.bias)
        v_b = np.zeros_like(self.bias)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        t = 0
        
        for epoch in range(1, self.epochs + 1):
            indices = np.arange(N)
            np.random.RandomState(42 + epoch).shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled_oh = y_one_hot[indices]
            
            num_batches = int(np.ceil(N / self.batch_size))
            
            for b in range(num_batches):
                start_idx = b * self.batch_size
                end_idx = min(start_idx + self.batch_size, N)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch_oh = y_shuffled_oh[start_idx:end_idx]
                B = X_batch.shape[0]
                
                Z = np.dot(X_batch, self.weights.T) + self.bias
                probs = self._softmax(Z)
                
                dZ = probs - y_batch_oh
                dW = np.dot(dZ.T, X_batch) / B + self.l2_reg * self.weights
                db = np.sum(dZ, axis=0) / B
                
                t += 1
                m_w = beta1 * m_w + (1 - beta1) * dW
                v_w = beta2 * v_w + (1 - beta2) * (dW ** 2)
                m_w_corrected = m_w / (1 - beta1 ** t)
                v_w_corrected = v_w / (1 - beta2 ** t)
                self.weights -= self.lr * m_w_corrected / (np.sqrt(v_w_corrected) + eps)
                
                m_b = beta1 * m_b + (1 - beta1) * db
                v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
                m_b_corrected = m_b / (1 - beta1 ** t)
                v_b_corrected = v_b / (1 - beta2 ** t)
                self.bias -= self.lr * m_b_corrected / (np.sqrt(v_b_corrected) + eps)
                
        return self
        
    def predict_proba(self, X):
        Z = np.dot(X, self.weights.T) + self.bias
        return self._softmax(Z)
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ==========================================
# 4. Custom Macro-F1 Metric
# ==========================================

def calculate_macro_f1(y_true, y_pred, num_classes=23):
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
    return np.mean(f1_scores)

# ==========================================
# 5. Main Execution & Comparison Pipeline
# ==========================================

def main():
    print("="*75, flush=True)
    print("  LAB 2: Language Identification with 6 TF-IDF Normalization Combinations", flush=True)
    print("="*75, flush=True)
    
    # 1. Load dataset
    dataset = get_dataset()
    languages_loaded = sorted(list(dataset.keys()))
    print(f"Loaded dataset with {len(languages_loaded)} languages.", flush=True)
    
    # 2. Create splits
    df_train, df_val, df_test = split_and_stratify(dataset)
    
    os.makedirs("splits", exist_ok=True)
    df_train.to_csv("splits/train.csv", index=False, encoding="utf-8")
    df_val.to_csv("splits/val.csv", index=False, encoding="utf-8")
    df_test.to_csv("splits/test.csv", index=False, encoding="utf-8")
    
    label_to_id = {label: i for i, label in enumerate(languages_loaded)}
    y_train = np.array([label_to_id[l] for l in df_train['label']], dtype=np.int32)
    y_val = np.array([label_to_id[l] for l in df_val['label']], dtype=np.int32)
    y_test = np.array([label_to_id[l] for l in df_test['label']], dtype=np.int32)
    
    # Pre-extract n-grams once to avoid repeated computation across 6 combinations
    print("Pre-extracting n-grams for Train, Val, and Test splits...", flush=True)
    train_docs = df_train['text'].tolist()
    val_docs = df_val['text'].tolist()
    test_docs = df_test['text'].tolist()
    
    train_ngrams = [extract_ngrams(doc) for doc in train_docs]
    val_ngrams = [extract_ngrams(doc) for doc in val_docs]
    test_ngrams = [extract_ngrams(doc) for doc in test_docs]
    print("N-gram extraction complete.\n", flush=True)
    
    # 6 Combinations of TF & IDF
    combinations = [
        ("Combo 1", "unnormalized", "unnormalized", "Unnormalized TF + Unnormalized IDF (Lab 1 Baseline)"),
        ("Combo 2", "unnormalized", "normalized",   "Unnormalized TF + Max-DF Normalized IDF"),
        ("Combo 3", "total_words",  "unnormalized", "Total-Words Normalized TF + Unnormalized IDF"),
        ("Combo 4", "total_words",  "normalized",   "Total-Words Normalized TF + Max-DF Normalized IDF"),
        ("Combo 5", "max_freq",     "unnormalized", "Max-Freq Normalized TF + Unnormalized IDF"),
        ("Combo 6", "max_freq",     "normalized",   "Max-Freq Normalized TF + Max-DF Normalized IDF")
    ]
    
    results = []
    
    print("Starting evaluation across all 6 combinations...\n", flush=True)
    
    for combo_id, tf_type, idf_type, description in combinations:
        print(f"[{combo_id}] Running: {description}...", flush=True)
        
        # Feature extraction
        vectorizer = CustomTFIDFVectorizer(max_features=20000, tf_type=tf_type, idf_type=idf_type)
        X_train = vectorizer.fit_transform(train_ngrams, train_docs)
        X_val = vectorizer.transform(val_ngrams, val_docs)
        X_test = vectorizer.transform(test_ngrams, test_docs)
        
        # Model training
        model = ScratchSoftmaxRegression(lr=0.05, l2_reg=1e-4, epochs=25, batch_size=256)
        model.fit(X_train, y_train)
        
        # Model evaluation
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        
        val_acc = np.mean(val_preds == y_val)
        val_f1 = calculate_macro_f1(y_val, val_preds, num_classes=len(languages_loaded))
        
        test_acc = np.mean(test_preds == y_test)
        test_f1 = calculate_macro_f1(y_test, test_preds, num_classes=len(languages_loaded))
        
        results.append({
            "Combo ID": combo_id,
            "TF Normalization": tf_type,
            "IDF Normalization": idf_type,
            "Description": description,
            "Val Acc": val_acc,
            "Val Macro-F1": val_f1,
            "Test Acc": test_acc,
            "Test Macro-F1": test_f1
        })
        
        print(f"   --> Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | Test Acc: {test_acc:.4f} | Test Macro-F1: {test_f1:.4f}\n", flush=True)
        
    # Print Comparative Results Table
    df_results = pd.DataFrame(results)
    
    print("="*95, flush=True)
    print("                        LAB 2 COMPARISON PERFORMANCE TABLE", flush=True)
    print("="*95, flush=True)
    print(f"{'Combo':<8} | {'TF Scheme':<14} | {'IDF Scheme':<14} | {'Test Acc':<10} | {'Test Macro-F1':<14}")
    print("-"*95)
    for res in results:
        print(f"{res['Combo ID']:<8} | {res['TF Normalization']:<14} | {res['IDF Normalization']:<14} | {res['Test Acc']:<10.4f} | {res['Test Macro-F1']:<14.4f}")
    print("="*95 + "\n")
    
    # Export results to CSV
    df_results.to_csv("results_comparison.csv", index=False, encoding="utf-8")
    print("Saved evaluation results comparison to Lab2/results_comparison.csv.\n", flush=True)

if __name__ == "__main__":
    main()
