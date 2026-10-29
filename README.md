# LLM Lab Assignments

This repository contains implementations, empirical evaluations, and reports for Large Language Models (LLM) lab assignments using IndicCorp v2 datasets (primarily Hindi and 22 other Indic languages).

---

## Repository Structure

```
.
├── Lab1/                  # Lab 1: Indic Language Identification Pipeline
│   ├── splits/            # Stratified train, val, test splits for 23 Indic languages
│   └── main.py            # Dataset streaming, stratification, preprocessing & baseline
│
├── Lab2/                  # Lab 2: TF-IDF Variants & Naive Bayes Classification
│   ├── splits/            # Dataset splits
│   ├── main.py            # Implementation of 6 TF-IDF weighting schemes & evaluation
│   └── results_comparison.csv # Performance metrics comparison across TF-IDF variants
│
├── Lab3/                  # Lab 3: Subword Tokenization (BPE & WordPiece) from Scratch
│   ├── bpe.py             # Scratch Byte-Pair Encoding (BPE) implementation
│   ├── wordpiece.py       # Scratch WordPiece tokenizer implementation
│   ├── data_loader.py     # Sentence tokenization and length-stratified split generator
│   ├── experiments.py     # Data-size and vocab-size scaling benchmarks
│   ├── visualize.py       # Publication-ready charts and distribution plots
│   ├── main.py            # Master pipeline runner
│   ├── README.md          # Comprehensive Lab 3 documentation and analysis
│   ├── models/            # Saved BPE and WordPiece vocabulary models
│   └── results/           # Quantitative CSV results and visualization plots
│
├── lab4/                  # Lab 4: N-gram Language Modeling & Smoothing Techniques
│   ├── ngram_model.py     # Unigram, Bigram, and Trigram language models with Add-k & Interpolation
│   ├── experiments.py     # Perplexity benchmarks and parameter tuning
│   ├── visualize.py       # Tuning and perplexity heatmap/bar plot generation
│   ├── main.py            # Experiment orchestrator
│   ├── README.md          # Detailed Lab 4 documentation
│   └── results/           # Perplexity scores and comparison plots
│
└── lab5/                  # Lab 5: Word & Document Embeddings & Clustering
    ├── models.py          # Word2Vec (CBOW & Skip-Gram), FastText, and Doc2Vec wrappers
    ├── clustering.py      # K-Means clustering and cluster analysis
    ├── sentence_retrieval.py # Cosine similarity and nearest sentence search
    ├── experiments.py     # Full training and evaluation pipeline on 1M sentences
    ├── visualize.py       # 2D PCA/t-SNE visualization and elbow curves
    ├── main.py            # Master runner for Lab 5
    └── results/           # Output metrics, cluster visualizations, and saved models
```

---

## Lab Descriptions

### Lab 1: Multilingual Preprocessing & Language Identification
- Streams IndicCorp v2 corpus across 23 scheduled Indian languages.
- Cleans and sentence-tokenizes noisy web text, filtering by length and valid script codepoints.
- Creates stratified training, validation, and test splits with balanced language representation.

### Lab 2: TF-IDF Normalization & Naive Bayes Classification
- Evaluates 6 distinct TF-IDF term-frequency and document-frequency weighting schemes:
  - Binary, Natural, Sublinear, and Logarithmic Term Frequency.
  - Standard IDF and Probabilistic IDF.
- Trains a Multinomial Naive Bayes classifier on top of custom feature extractors.
- Evaluates accuracy, macro F1, and micro F1 across all 23 Indic languages.

### Lab 3: Subword Tokenization Algorithms (BPE & WordPiece)
- Scratch implementation of **Byte-Pair Encoding (BPE)** and **WordPiece** tokenizers.
- Empirical scaling study across corpus sizes (100k, 300k, 500k, 1M sentences) and vocabulary sizes (20k, 30k, 50k tokens).
- Evaluates tokenization efficiency, compression ratio, fertility rate, and unigram entropy.

### Lab 4: N-Gram Language Modeling & Smoothing
- Implementation of Unigram, Bigram, and Trigram language models.
- Smoothing techniques:
  - **Add-$k$ Laplace Smoothing** with optimal $k$ grid search on development set.
  - **Linear Interpolation Smoothing** using Jelinek-Mercer EM / grid search optimization.
- Evaluates perplexity (PPL) on unseen Hindi test data.

### Lab 5: Distributed Word & Document Embeddings with Clustering
- Vector space representation using **Word2Vec (CBOW & Skip-gram)**, **FastText** (subword n-grams), and **Doc2Vec (PV-DM & PV-DBOW)**.
- Trained on 1,000,000 sentences from IndicCorp v2 (Hindi).
- Sentence representation via average word vectors and Doc2Vec document vectors.
- Unsupervised semantic grouping using **K-Means Clustering** and semantic nearest neighbor sentence retrieval.

---

## Setup & Requirements

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib seaborn gensim tqdm requests
```
