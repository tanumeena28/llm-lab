# Lab 4: N-gram Language Modeling (Unigram, Bigram, Trigram, Quadram) & Smoothing Analysis

This directory contains the complete implementation, hyperparameter optimization, and empirical evaluation for **Lab 4**, focusing on building **N-gram Language Models** ($N=1, 2, 3, 4$) from scratch on the **IndicCorp v2 Hindi corpus (1,000,000 sentences)** under **5 smoothing techniques**:
1. **Unsmoothed (Maximum Likelihood Estimation - Raw Counts)**
2. **Add-1 (Laplace) Smoothing**
3. **Add-K Smoothing** ($0 < K < 1$)
4. **Interpolated Smoothing** (Linear Interpolation with $\lambda$)
5. **Kneser-Ney Smoothing** ($d = 0.75$)

---

## Workspace Layout

```
lab4/
├── ngram_model.py          # Core NGramLM implementation supporting N=1..4 & 5 smoothing algorithms
├── experiments.py          # Hyperparameter tuning on Dev set & Test set evaluation pipeline
├── visualize.py            # Publication-grade chart & heatmap generator
├── main.py                 # Master orchestrator script
├── README.md               # Complete lab report & documentation
└── results/                # Evaluation output files & generated plots
    ├── add_k_dev_tuning.csv
    ├── interpolation_dev_tuning.csv
    ├── ppl_comparison.csv
    ├── ppl_matrix.csv
    └── plots/
        ├── ppl_comparison_bar.png
        ├── ppl_heatmap.png
        └── add_k_tuning.png
```

---

## Task 1: Dataset & Corpus Preprocessing

- **Corpus**: IndicCorp v2 Hindi (`hin_Deva`) dataset containing **1,000,000 training sentences**, **1,000 development sentences**, and **1,000 test sentences** (reused directly from the length-stratified splits of Lab 3).
- **Vocabulary ($V$)**: Constructed from the 1,000,000 training sentences by retaining words with minimum frequency $f \ge 2$. All low-frequency words ($f < 2$) and unseen words in Dev/Test are mapped to the `<unk>` token. Special boundary tokens `<bos>` (Beginning-Of-Sentence) and `<eos>` (End-Of-Sentence) are added according to N-gram order.
- **Vocabulary Size $|V|$**: **253,844 unique tokens**.
- **Unique Extracted N-grams (1M Sentences)**:
  - Unigrams ($N=1$): **253,843**
  - Bigrams ($N=2$): **5,573,768**
  - Trigrams ($N=3$): **16,897,532**
  - Quadrams ($N=4$): **26,472,864**

---

## Task 2: Mathematical Formulations of Smoothing Techniques

For an N-gram of order $N$, let $w_i$ be the target token, $context = (w_{i-N+1}, \dots, w_{i-1})$ be the preceding context of length $N-1$, $c(context, w_i)$ be the joint N-gram count, $c(context)$ be the context count, and $|V|$ be the vocabulary size.

### 1. Unsmoothed (Maximum Likelihood Estimation - MLE)
- **Unigram ($N=1$)**:
  $$P_{\text{MLE}}(w_i) = \frac{c(w_i)}{N_{\text{tokens}}}$$
- **$N$-gram ($N > 1$)**:
  $$P_{\text{MLE}}(w_i \mid context) = \frac{c(context, w_i)}{c(context)}$$
  *(Note: If $c(context) = 0$ or $c(context, w_i) = 0$, $P_{\text{MLE}} = 0$, leading to $\text{Perplexity} = \infty$.)*

### 2. Add-1 (Laplace) Smoothing
Adds 1 to every N-gram count to eliminate zero probabilities:
- **Unigram ($N=1$)**:
  $$P_{\text{Laplace}}(w_i) = \frac{c(w_i) + 1}{N_{\text{tokens}} + |V|}$$
- **$N$-gram ($N > 1$)**:
  $$P_{\text{Laplace}}(w_i \mid context) = \frac{c(context, w_i) + 1}{c(context) + |V|}$$

### 3. Add-K Smoothing ($0 < K < 1$)
Adds a fractional count $K$ ($0 < K < 1$) to avoid over-allocating probability mass to unseen transitions on large vocabularies:
- **Unigram ($N=1$)**:
  $$P_{\text{Add-K}}(w_i) = \frac{c(w_i) + K}{N_{\text{tokens}} + K \cdot |V|}$$
- **$N$-gram ($N > 1$)**:
  $$P_{\text{Add-K}}(w_i \mid context) = \frac{c(context, w_i) + K}{c(context) + K \cdot |V|}$$

### 4. Interpolated Smoothing (Linear Interpolation)
Combines higher-order MLE counts with lower-order smoothed/uniform estimates:
- **Unigram ($N=1$)**:
  $$P_{\text{interp}}(w_i) = \lambda P_{\text{MLE}}(w_i) + (1 - \lambda) \frac{1}{|V|}$$
- **$N$-gram ($N > 1$)**:
  $$P_{\text{interp}}(w_i \mid context) = \sum_{j=1}^{N} \lambda_j P_{\text{MLE}}(w_i \mid context_{N-j}) \quad \text{where } \sum_{j=1}^{N} \lambda_j = 1$$

### 5. Kneser-Ney Smoothing ($d = 0.75$)
Uses absolute discounting $d = 0.75$ and scales lower-order predictions by continuation counts:
- **Unigram ($N=1$)**:
  $$P_{\text{KN}}(w_i) = \frac{\max(c(w_i) - d, 0)}{N_{\text{tokens}}} + \frac{d \cdot |V_{\text{seen}}|}{N_{\text{tokens}}} \cdot P_{\text{continuation}}(w_i)$$
- **$N$-gram ($N > 1$)**:
  $$P_{\text{KN}}(w_i \mid context) = \frac{\max(c(context, w_i) - d, 0)}{c(context)} + \lambda(context) \cdot P_{\text{KN}}(w_i \mid context_{\text{lower}})$$
  $$\lambda(context) = \frac{d}{c(context)} \cdot |\{w : c(context, w) > 0\}|$$

---

## Task 3: Hyperparameter Tuning on Dev Set

Parameters $K$ (for Add-K) and $\lambda$ (for Interpolation) were optimized on the 1,000-sentence Dev set.

### 1. Add-K Parameter Optimization ($K \in (0, 1)$)
- **Best $K = 0.001$ across all N-gram orders**:
  - $N=1$: $K=0.001 \rightarrow \text{Dev PPL} = 1754.57$
  - $N=2$: $K=0.001 \rightarrow \text{Dev PPL} = 293.65$
  - $N=3$: $K=0.001 \rightarrow \text{Dev PPL} = 1341.53$
  - $N=4$: $K=0.001 \rightarrow \text{Dev PPL} = 12631.82$

### 2. Interpolation Lambdas Optimization ($\sum \lambda_j = 1$)
- **Unigram ($N=1$)**: $\lambda = [0.99] \rightarrow \text{Dev PPL} = 1759.66$
- **Bigram ($N=2$)**: $\lambda = [0.8, 0.2] \rightarrow \text{Dev PPL} = 308.18$
- **Trigram ($N=3$)**: $\lambda = [0.5, 0.35, 0.15] \rightarrow \text{Dev PPL} = 230.81$
- **Quadram ($N=4$)**: $\lambda = [0.4, 0.3, 0.2, 0.1] \rightarrow \text{Dev PPL} = 273.73$

---

## Task 4: Empirical Test Set Perplexity Evaluation

Perplexity evaluated across the 1,000-sentence Test set for all 20 model configurations:

| Smoothing Method | Unigram (N=1) | Bigram (N=2) | Trigram (N=3) | Quadram (N=4) |
|---|:---:|:---:|:---:|:---:|
| **Unsmoothed (MLE)** | 1707.42 | **INF** | **INF** | **INF** |
| **Add-1 (Laplace)** | 1710.98 | 2371.83 | 27464.88 | 97396.15 |
| **Add-K ($K=0.001$)** | 1707.43 | 265.00 | 1206.03 | 11968.56 |
| **Interpolated** | 1713.07 | 284.30 | 210.34 | 247.32 |
| **Kneser-Ney ($d=0.75$)** | **1706.30** | **251.14** | **174.53** | **177.69** |

---

## Empirical Findings & Comparative Analysis

1. **Failure of Unsmoothed Models ($N > 1$)**:
   - For $N \ge 2$, unseen N-gram transitions on the test set receive raw count $0 \rightarrow P = 0$. Consequently, total log-likelihood drops to $-\infty$, resulting in infinite perplexity ($\text{INF}$). Unsmoothed models cannot generalize to unseen test text.

2. **Flaw of Add-1 (Laplace) Smoothing on Large Vocabularies**:
   - Because $|V| = 253,844$, adding $1$ to every possible transition adds $253,844$ pseudo-counts to the denominator for every context. For higher-order contexts ($N=3, 4$) that appear infrequently, adding $253,844$ severely degrades probability estimation, inflating Quadram perplexity to **97,396.15**.

3. **Superiority of Add-K ($K < 1$) over Add-1**:
   - Decreasing $K$ from $1.0$ to $0.001$ reduces the denominator penalty from $253,844$ to $253.844$. This dramatically improves Bigram perplexity from **2371.83 down to 265.00**. However, Add-K still suffers at $N=3, 4$ because it assigns uniform mass to all unobserved words regardless of their unigram frequency.

4. **Power of Linear Interpolation**:
   - Interpolation solves context sparsity by falling back to lower-order models whenever a higher-order context is unobserved. Trigram interpolation achieves a perplexity of **210.34** (compared to 1206.03 for Add-K).

5. **Kneser-Ney Smoothing ($d=0.75$) Achieves State-of-the-Art Performance**:
   - Kneser-Ney achieves the overall **lowest perplexity of 174.53** (at $N=3$). By combining absolute discounting ($d=0.75$) with continuation probabilities (which measure how likely a word completes an arbitrary context rather than its unigram raw frequency), Kneser-Ney handles zero-frequency contexts with state-of-the-art precision.

---

## How to Run the Code

To execute the entire training, hyperparameter tuning, evaluation, and plotting workflow end-to-end:

```bash
py -3.12 -u lab4/main.py
```
