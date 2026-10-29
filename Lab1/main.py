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
CACHE_FILE = "raw_sentences_cache.json"

# ==========================================
# 1. Dataset Construction & Tokenization
# ==========================================

def clean_sentence(text):
    text = text.strip()
    # Filter out very short lines or numeric/punctuation-only noise
    if len(text) < 10:
        return None
    return text

def download_sentences(lang, filename, target_count=1000):
    url = BASE_URL + filename
    sentences = []
    print(f"Streaming {lang} from {filename}...", flush=True)
    
    # Sentence splitter pattern
    # Splits on: U+0964 (।), U+0965 (॥), U+06D4 (۔), period (.), question mark (?), exclamation (!)
    # followed by optional whitespace.
    split_pattern = re.compile(r'[।॥\.?!\u06d4]\s*')
    
    response = requests.get(url, stream=True, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch {filename}: HTTP {response.status_code}")
    
    # Read chunk-by-chunk and decode line-by-line to avoid UnicodeDecodeError
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        try:
            line = raw_line.decode('utf-8', errors='ignore')
        except Exception:
            continue
            
        # Split paragraph/line into sentences
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
    print(f"  --> Successfully collected {len(sentences)} sentences for {lang}.", flush=True)
    return sentences[:target_count]

def get_dataset():
    # Check if cached file exists
    if os.path.exists(CACHE_FILE):
        print(f"Loading raw dataset from local cache: {CACHE_FILE}...", flush=True)
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print("Local cache not found. Starting direct downloads from Hugging Face...", flush=True)
    dataset = {}
    for i, (lang, filename) in enumerate(LANG_FILES.items(), 1):
        print(f"[{i}/{len(LANG_FILES)}] ", end="", flush=True)
        try:
            sentences = download_sentences(lang, filename, target_count=1000)
            dataset[lang] = sentences
        except Exception as e:
            print(f"Error downloading {lang}: {e}. Retrying with fallback...", flush=True)
            # Try once more with a longer timeout
            try:
                sentences = download_sentences(lang, filename, target_count=1000)
                dataset[lang] = sentences
            except Exception as e2:
                print(f"Failed again for {lang}: {e2}. Continuing...", flush=True)
                
    # Save cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved raw dataset cache to {CACHE_FILE}.", flush=True)
    return dataset

def split_and_stratify(dataset):
    print("Splitting dataset (80% Train, 10% Val, 10% Test) using stratified sampling...", flush=True)
    train_data = []
    val_data = []
    test_data = []
    
    for lang, sentences in dataset.items():
        # Shuffle sentences for this language to ensure random distribution
        shuffled = list(sentences)
        random.shuffle(shuffled)
        
        # 80:10:10 splits
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
            
    # Shuffle the final datasets so model doesn't see one language at a time
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    df_test = pd.DataFrame(test_data)
    
    return df_train, df_val, df_test

# ==========================================
# 2. Custom TF-IDF Vectorizer from Scratch
# ==========================================

def extract_ngrams(text):
    # Word unigrams and bigrams
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
    
    # Char 2-grams
    for i in range(len(chars) - 1):
        ngrams.append(f"c2:{chars[i:i+2]}")
        
    # Char 3-grams
    for i in range(len(chars) - 2):
        ngrams.append(f"c3:{chars[i:i+3]}")
        
    # Char 4-grams
    for i in range(len(chars) - 3):
        ngrams.append(f"c4:{chars[i:i+4]}")
        
    return ngrams

class ScratchTFIDFVectorizer:
    def __init__(self, max_features=20000):
        self.max_features = max_features
        self.vocabulary = {} # feature -> index
        self.idf = None # array of idf values
        
    def fit(self, raw_documents):
        print(f"Fitting TF-IDF Vectorizer with max_features={self.max_features}...", flush=True)
        # 1. Count overall frequencies of ngrams to select top features
        ngram_counts = collections.Counter()
        # 2. Count document frequency (df) for each ngram
        doc_counts = collections.Counter()
        
        for doc in raw_documents:
            ngrams = extract_ngrams(doc)
            ngram_counts.update(ngrams)
            doc_counts.update(set(ngrams))
            
        # Select top max_features by frequency
        most_common = [item[0] for item in ngram_counts.most_common(self.max_features)]
        self.vocabulary = {ng: i for i, ng in enumerate(most_common)}
        
        # Calculate IDF values
        # idf_i = ln((1 + N) / (1 + df_i)) + 1
        N = len(raw_documents)
        self.idf = np.zeros(len(self.vocabulary), dtype=np.float32)
        for ng, idx in self.vocabulary.items():
            df = doc_counts[ng]
            self.idf[idx] = np.log((1 + N) / (1 + df)) + 1
            
        print(f"Built vocabulary of size {len(self.vocabulary)}.", flush=True)
        return self
        
    def transform(self, raw_documents):
        N = len(raw_documents)
        V = len(self.vocabulary)
        # Initialize dense matrix in float32 to conserve RAM
        X = np.zeros((N, V), dtype=np.float32)
        
        for doc_idx, doc in enumerate(raw_documents):
            ngrams = extract_ngrams(doc)
            counts = collections.Counter(ngrams)
            
            # Fill document term frequencies
            for ng, count in counts.items():
                if ng in self.vocabulary:
                    f_idx = self.vocabulary[ng]
                    X[doc_idx, f_idx] = count
                    
        # Compute TF-IDF using NumPy broadcasting
        X *= self.idf
        
        # Vectorized L2-normalize document vectors
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        X /= row_norms
                
        return X
        
    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)

# ==========================================
# 3. Custom Logistic Regression (Softmax)
# ==========================================

class ScratchSoftmaxRegression:
    def __init__(self, lr=0.01, l2_reg=1e-4, epochs=30, batch_size=128, verbose=True):
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.weights = None # shape: (C, F)
        self.bias = None # shape: (C,)
        
    def _softmax(self, Z):
        # Shifts Z for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
    def fit(self, X, y, X_val=None, y_val=None, val_eval_fn=None):
        N, F = X.shape
        C = int(np.max(y)) + 1
        
        print(f"Training Softmax Regression on {N} samples with {F} features across {C} classes...", flush=True)
        # Initialize weights and bias to zeros
        self.weights = np.zeros((C, F), dtype=np.float32)
        self.bias = np.zeros(C, dtype=np.float32)
        
        # One-hot encode targets
        y_one_hot = np.zeros((N, C), dtype=np.float32)
        y_one_hot[np.arange(N), y] = 1.0
        
        # Adam optimizer parameters
        m_w = np.zeros_like(self.weights)
        v_w = np.zeros_like(self.weights)
        m_b = np.zeros_like(self.bias)
        v_b = np.zeros_like(self.bias)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        t = 0
        
        for epoch in range(1, self.epochs + 1):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled_oh = y_one_hot[indices]
            
            epoch_loss = 0.0
            num_batches = int(np.ceil(N / self.batch_size))
            
            for b in range(num_batches):
                start_idx = b * self.batch_size
                end_idx = min(start_idx + self.batch_size, N)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch_oh = y_shuffled_oh[start_idx:end_idx]
                B = X_batch.shape[0]
                
                # Forward Pass
                Z = np.dot(X_batch, self.weights.T) + self.bias
                probs = self._softmax(Z)
                
                # Compute Loss (Cross-Entropy + L2 Penalty)
                loss = -np.sum(y_batch_oh * np.log(probs + 1e-15)) / B
                loss += 0.5 * self.l2_reg * np.sum(self.weights ** 2)
                epoch_loss += loss * B
                
                # Backward Pass (dZ = predicted - actual)
                dZ = probs - y_batch_oh
                
                # Gradients
                dW = np.dot(dZ.T, X_batch) / B + self.l2_reg * self.weights
                db = np.sum(dZ, axis=0) / B
                
                # Adam Updates
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
                
            epoch_loss /= N
            
            if self.verbose and (epoch == 1 or epoch % 5 == 0 or epoch == self.epochs):
                val_str = ""
                if X_val is not None and y_val is not None:
                    val_preds = self.predict(X_val)
                    val_acc = np.mean(val_preds == y_val)
                    if val_eval_fn is not None:
                        val_f1 = val_eval_fn(y_val, val_preds)
                        val_str = f" | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}"
                    else:
                        val_str = f" | Val Acc: {val_acc:.4f}"
                print(f"Epoch {epoch:02d}/{self.epochs} | Train Loss: {epoch_loss:.4f}{val_str}", flush=True)
                
        return self
        
    def predict_proba(self, X):
        Z = np.dot(X, self.weights.T) + self.bias
        return self._softmax(Z)
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ==========================================
# 4. Custom Macro-F1 Metric from Scratch
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

def print_classification_report(y_true, y_pred, class_labels):
    print("\n" + "="*65)
    print(f"{'Language Label':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("="*65)
    
    total_tp = 0
    total_support = 0
    f1_list = []
    
    for idx, name in enumerate(class_labels):
        tp = np.sum((y_true == idx) & (y_pred == idx))
        fp = np.sum((y_true != idx) & (y_pred == idx))
        fn = np.sum((y_true == idx) & (y_pred != idx))
        support = np.sum(y_true == idx)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        total_tp += tp
        total_support += support
        f1_list.append(f1)
        
        print(f"{name:<15} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {support:<8}")
        
    print("-"*65)
    accuracy = total_tp / total_support if total_support > 0 else 0.0
    macro_f1 = np.mean(f1_list)
    print(f"{'Accuracy':<15} | {'':<10} | {'':<10} | {accuracy:<10.4f} | {total_support:<8}")
    print(f"{'Macro-F1':<15} | {'':<10} | {'':<10} | {macro_f1:<10.4f} | {total_support:<8}")
    print("="*65 + "\n")

# ==========================================
# 5. Main Execution Pipeline
# ==========================================

def main():
    print("="*70, flush=True)
    print("  Language Identification System: Custom Pipeline on IndicCorpV2", flush=True)
    print("="*70, flush=True)
    
    # 1. Load data
    dataset = get_dataset()
    
    # 2. Check dataset integrity
    languages_loaded = list(dataset.keys())
    print(f"Loaded {len(languages_loaded)} languages from raw source/cache.", flush=True)
    if len(languages_loaded) != 23:
        print(f"Warning: Expected 23 languages, but loaded {len(languages_loaded)}.", flush=True)
        
    # 3. Create splits
    df_train, df_val, df_test = split_and_stratify(dataset)
    print(f"Train split size: {len(df_train)} sentences.", flush=True)
    print(f"Val split size:   {len(df_val)} sentences.", flush=True)
    print(f"Test split size:  {len(df_test)} sentences.", flush=True)
    
    # Print a table of counts per language to verify stratified splits
    print("\nStratification verification (sample counts per language):")
    train_counts = df_train['label'].value_counts()
    val_counts = df_val['label'].value_counts()
    test_counts = df_test['label'].value_counts()
    
    print(f"{'Language':<15} | {'Train':<6} | {'Val':<6} | {'Test':<6}")
    print("-"*42)
    for lang in sorted(languages_loaded):
        print(f"{lang:<15} | {train_counts.get(lang, 0):<6} | {val_counts.get(lang, 0):<6} | {test_counts.get(lang, 0):<6}")
        
    # Save splits to local directory
    os.makedirs("splits", exist_ok=True)
    df_train.to_csv("splits/train.csv", index=False, encoding="utf-8")
    df_val.to_csv("splits/val.csv", index=False, encoding="utf-8")
    df_test.to_csv("splits/test.csv", index=False, encoding="utf-8")
    print("\nSaved stratified splits to splits/train.csv, splits/val.csv, and splits/test.csv.\n", flush=True)
    
    # Map label names to class index numbers
    class_labels = sorted(languages_loaded)
    label_to_id = {label: i for i, label in enumerate(class_labels)}
    
    y_train = np.array([label_to_id[l] for l in df_train['label']], dtype=np.int32)
    y_val = np.array([label_to_id[l] for l in df_val['label']], dtype=np.int32)
    y_test = np.array([label_to_id[l] for l in df_test['label']], dtype=np.int32)
    
    # 4. Feature extraction (TF-IDF)
    vectorizer = ScratchTFIDFVectorizer(max_features=20000)
    
    print("Transforming training texts...", flush=True)
    X_train = vectorizer.fit_transform(df_train['text'].tolist())
    
    print("Transforming validation texts...", flush=True)
    X_val = vectorizer.transform(df_val['text'].tolist())
    
    print("Transforming test texts...", flush=True)
    X_test = vectorizer.transform(df_test['text'].tolist())
    
    print(f"Feature matrix shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}", flush=True)
    
    # 5. Model initialization & training
    # Learning rate = 0.05, regularization lambda = 1e-4, epochs = 25
    model = ScratchSoftmaxRegression(lr=0.05, l2_reg=1e-4, epochs=25, batch_size=256, verbose=True)
    
    # Helper lambda to evaluate validation Macro-F1 during training
    val_f1_fn = lambda y_true, y_pred: calculate_macro_f1(y_true, y_pred, num_classes=len(class_labels))
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, val_eval_fn=val_f1_fn)
    
    # 6. Evaluation
    print("\nRunning evaluation on test set...", flush=True)
    test_preds = model.predict(X_test)
    
    test_acc = np.mean(test_preds == y_test)
    test_macro_f1 = calculate_macro_f1(y_test, test_preds, num_classes=len(class_labels))
    
    print(f"\nFinal Test Set Accuracy: {test_acc:.4f}")
    print(f"Final Test Set Macro-F1: {test_macro_f1:.4f}")
    
    print_classification_report(y_test, test_preds, class_labels)
    
    print("Done! Everything finished successfully.", flush=True)

if __name__ == "__main__":
    main()
