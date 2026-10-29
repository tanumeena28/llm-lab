"""
Hyper-Optimized N-gram Language Model Implementation for Lab 4.
Supports N=1, 2, 3, 4 (Unigram, Bigram, Trigram, Quadram)
and 5 Smoothing Algorithms:
1. Unsmoothed (MLE Raw Counts)
2. Add-1 (Laplace) Smoothing
3. Add-K Smoothing (0 < K < 1)
4. Interpolated Smoothing (Linear Interpolation)
5. Kneser-Ney Smoothing (with d = 0.75)
"""

import math
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Union, Optional
from tqdm import tqdm


BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


class NGramLM:
    """
    N-gram Language Model class capable of building models up to order N=4
    and evaluating smoothed/unsmoothed sentence probabilities & perplexity.
    """

    def __init__(self, n: int = 4, min_freq: int = 2):
        self.n = n
        self.min_freq = min_freq
        
        # Vocabulary
        self.vocab: Set[str] = set()
        self.vocab_size: int = 0
        self.word_counts: Counter = Counter()
        
        # N-gram count dictionaries: ngram_counts[k][(w1, ..., wk)] -> count
        self.ngram_counts: Dict[int, Dict[Tuple[str, ...], int]] = {
            k: {} for k in range(1, self.n + 1)
        }
        
        # Continuation counts for Kneser-Ney: kn_continuations[k][context] -> number of distinct next words
        self.kn_continuations: Dict[int, Dict[Tuple[str, ...], int]] = {
            k: defaultdict(int) for k in range(1, self.n + 1)
        }
        
        # Continuation count for unigram: number of unique left-contexts for word w
        self.unigram_left_continuations: Dict[str, int] = defaultdict(int)
        self.total_left_continuations: int = 0
        self.total_unigram_tokens: int = 0

    def build_vocab(self, corpus: List[List[str]]) -> None:
        """
        Build vocabulary from training corpus and determine <unk> mapping.
        """
        print("Building vocabulary from training corpus...", flush=True)
        raw_counts = Counter()
        for sentence in corpus:
            raw_counts.update(sentence)
            
        self.vocab = {BOS, EOS, UNK}
        for word, count in raw_counts.items():
            if count >= self.min_freq:
                self.vocab.add(word)
                
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary built: {self.vocab_size:,} unique tokens (min_freq={self.min_freq}).", flush=True)

    def _preprocess_sentence(self, sentence: List[str], order: int) -> List[str]:
        """
        Preprocess sentence by mapping OOVs to <unk> and adding order-1 <bos> tokens and 1 <eos> token.
        """
        clean_words = [w if w in self.vocab else UNK for w in sentence]
        bos_padding = [BOS] * (order - 1)
        return bos_padding + clean_words + [EOS]

    def fit(self, corpus: List[List[str]]) -> None:
        """
        Fit N-gram counts up to order self.n over the input corpus with instant C-speed counting.
        """
        if not self.vocab:
            self.build_vocab(corpus)

        print(f"Preprocessing {len(corpus):,} sentences for max N-gram order N={self.n}...", flush=True)
        vocab_set = self.vocab
        
        # Precompute clean padded tokens
        padded_corpus = []
        bos_pad = [BOS] * (self.n - 1)
        for sent in tqdm(corpus, desc="Preprocessing sentences"):
            clean = [w if w in vocab_set else UNK for w in sent]
            padded_corpus.append(bos_pad + clean + [EOS])

        print(f"Counting N-grams for orders 1 to {self.n}...", flush=True)
        for k in range(1, self.n + 1):
            print(f"  Extracting N-grams for order k={k}...", flush=True)
            counter = Counter()
            
            if k == 1:
                start_idx = self.n - 1
                for sent in padded_corpus:
                    tokens = sent[start_idx:]
                    counter.update((w,) for w in tokens)
            elif k == 2:
                start_idx = self.n - 2
                for sent in padded_corpus:
                    s = sent[start_idx:]
                    counter.update(zip(s, s[1:]))
            elif k == 3:
                start_idx = self.n - 3
                for sent in padded_corpus:
                    s = sent[start_idx:]
                    counter.update(zip(s, s[1:], s[2:]))
            elif k == 4:
                start_idx = self.n - 4
                for sent in padded_corpus:
                    s = sent[start_idx:]
                    counter.update(zip(s, s[1:], s[2:], s[3:]))

            ng_dict = dict(counter)
            self.ngram_counts[k] = ng_dict
            print(f"  Order k={k} extracted {len(ng_dict):,} unique N-grams.", flush=True)

            if k == 1:
                for (w,), count in ng_dict.items():
                    if w != BOS:
                        self.word_counts[w] = count
                        self.total_unigram_tokens += count
            else:
                # Fast continuation counts over unique N-grams
                kn_dict = self.kn_continuations[k]
                for ngram in ng_dict.keys():
                    kn_dict[ngram[:-1]] += 1

        print("Computing Kneser-Ney continuation statistics...", flush=True)
        if self.n >= 2:
            bigram_predecessors = defaultdict(set)
            for (w1, w2) in self.ngram_counts[2].keys():
                bigram_predecessors[w2].add(w1)
            
            for w2, pred_set in bigram_predecessors.items():
                self.unigram_left_continuations[w2] = len(pred_set)
            self.total_left_continuations = sum(self.unigram_left_continuations.values())

        print("N-gram model fitting completed successfully.", flush=True)

    def get_context_count(self, context: Tuple[str, ...]) -> int:
        """
        Context count c(w_1, ..., w_{k-1}) is equal to the count of the (k-1)-gram context.
        """
        k = len(context) + 1
        if k == 1:
            return self.total_unigram_tokens
        return self.ngram_counts[k - 1].get(context, 0)

    # -------------------------------------------------------------------------
    # 1. Unsmoothed (MLE) Probability
    # -------------------------------------------------------------------------
    def prob_unsmoothed(self, word: str, context: Tuple[str, ...]) -> float:
        order = len(context) + 1
        ngram = context + (word,)
        if order == 1:
            cnt = self.word_counts.get(word, 0)
            return cnt / float(self.total_unigram_tokens) if self.total_unigram_tokens > 0 else 0.0
        
        ctx_cnt = self.get_context_count(context)
        if ctx_cnt == 0:
            return 0.0
        ngram_cnt = self.ngram_counts[order].get(ngram, 0)
        return ngram_cnt / float(ctx_cnt)

    # -------------------------------------------------------------------------
    # 2. Add-1 (Laplace) Smoothing
    # -------------------------------------------------------------------------
    def prob_add_one(self, word: str, context: Tuple[str, ...]) -> float:
        order = len(context) + 1
        ngram = context + (word,)
        if order == 1:
            cnt = self.word_counts.get(word, 0)
            return (cnt + 1.0) / (self.total_unigram_tokens + self.vocab_size)
        
        ctx_cnt = self.get_context_count(context)
        ngram_cnt = self.ngram_counts[order].get(ngram, 0)
        return (ngram_cnt + 1.0) / (ctx_cnt + self.vocab_size)

    # -------------------------------------------------------------------------
    # 3. Add-K Smoothing (0 < K < 1)
    # -------------------------------------------------------------------------
    def prob_add_k(self, word: str, context: Tuple[str, ...], k: float = 0.1) -> float:
        order = len(context) + 1
        ngram = context + (word,)
        if order == 1:
            cnt = self.word_counts.get(word, 0)
            return (cnt + k) / (self.total_unigram_tokens + k * self.vocab_size)
        
        ctx_cnt = self.get_context_count(context)
        ngram_cnt = self.ngram_counts[order].get(ngram, 0)
        return (ngram_cnt + k) / (ctx_cnt + k * self.vocab_size)

    # -------------------------------------------------------------------------
    # 4. Interpolated Smoothing
    # -------------------------------------------------------------------------
    def prob_interpolated(self, word: str, context: Tuple[str, ...], lambdas: Optional[List[float]] = None) -> float:
        order = len(context) + 1
        
        if lambdas is None:
            if order == 1:
                lambdas = [0.9]
            elif order == 2:
                lambdas = [0.8, 0.2]
            elif order == 3:
                lambdas = [0.7, 0.2, 0.1]
            elif order == 4:
                lambdas = [0.6, 0.25, 0.1, 0.05]

        if order == 1:
            lam = lambdas[0]
            p_mle = self.prob_unsmoothed(word, ())
            return lam * p_mle + (1.0 - lam) * (1.0 / self.vocab_size)
        
        prob = 0.0
        for i in range(order):
            lam = lambdas[i]
            ctx_sub = context[i:]
            if len(ctx_sub) == 0:
                p_uni = self.prob_unsmoothed(word, ())
                prob += lam * p_uni
            else:
                prob += lam * self.prob_unsmoothed(word, ctx_sub)
                
        return max(prob, 1e-12)

    # -------------------------------------------------------------------------
    # 5. Kneser-Ney Smoothing (d = 0.75)
    # -------------------------------------------------------------------------
    def prob_kneser_ney(self, word: str, context: Tuple[str, ...], d: float = 0.75) -> float:
        order = len(context) + 1
        
        if order == 1:
            if self.total_left_continuations > 0:
                p_continuation = self.unigram_left_continuations.get(word, 0) / float(self.total_left_continuations)
            else:
                p_continuation = 1.0 / self.vocab_size
                
            cnt = self.word_counts.get(word, 0)
            first_term = max(cnt - d, 0) / float(self.total_unigram_tokens) if self.total_unigram_tokens > 0 else 0.0
            lambda_uni = (d * len(self.word_counts)) / float(self.total_unigram_tokens) if self.total_unigram_tokens > 0 else 1.0
            prob = first_term + lambda_uni * p_continuation
            return prob if prob > 0 else (1.0 / self.vocab_size)

        ctx_cnt = self.get_context_count(context)
        lower_context = context[1:]
        lower_prob = self.prob_kneser_ney(word, lower_context, d=d)
        
        if ctx_cnt == 0:
            return lower_prob
        
        ngram = context + (word,)
        ngram_cnt = self.ngram_counts[order].get(ngram, 0)
        first_term = max(ngram_cnt - d, 0) / float(ctx_cnt)
        
        n_continuations = self.kn_continuations[order].get(context, 0)
        lambda_ctx = (d / float(ctx_cnt)) * n_continuations
        
        prob = first_term + lambda_ctx * lower_prob
        return max(prob, 1e-12)

    # -------------------------------------------------------------------------
    # Sentence & Corpus Perplexity Calculation
    # -------------------------------------------------------------------------
    def evaluate_sentence_logprob(
        self,
        sentence: List[str],
        order: int,
        smoothing: str = "add_one",
        **kwargs
    ) -> Tuple[float, int]:
        tokens = self._preprocess_sentence(sentence, order)
        total_log_prob = 0.0
        eval_words = 0
        
        for i in range(order - 1, len(tokens)):
            word = tokens[i]
            context = tuple(tokens[i - (order - 1):i]) if order > 1 else ()
            
            if smoothing == "unsmoothed":
                prob = self.prob_unsmoothed(word, context)
            elif smoothing == "add_one":
                prob = self.prob_add_one(word, context)
            elif smoothing == "add_k":
                k = kwargs.get("k", 0.1)
                prob = self.prob_add_k(word, context, k=k)
            elif smoothing == "interpolated":
                lambdas = kwargs.get("lambdas", None)
                prob = self.prob_interpolated(word, context, lambdas=lambdas)
            elif smoothing == "kneser_ney":
                d = kwargs.get("d", 0.75)
                prob = self.prob_kneser_ney(word, context, d=d)
            else:
                raise ValueError(f"Unknown smoothing method: {smoothing}")

            if prob <= 0.0:
                return float("-inf"), len(tokens) - (order - 1)
            
            total_log_prob += math.log2(prob)
            eval_words += 1

        return total_log_prob, eval_words

    def compute_perplexity(
        self,
        test_corpus: List[List[str]],
        order: int,
        smoothing: str = "add_one",
        **kwargs
    ) -> float:
        total_log_prob = 0.0
        total_eval_words = 0
        
        for sentence in test_corpus:
            log_prob, n_words = self.evaluate_sentence_logprob(sentence, order=order, smoothing=smoothing, **kwargs)
            if math.isinf(log_prob) and log_prob < 0:
                return float("inf")
            total_log_prob += log_prob
            total_eval_words += n_words
            
        if total_eval_words == 0:
            return float("inf")
        
        avg_neg_log_prob = - (total_log_prob / float(total_eval_words))
        ppl = math.pow(2.0, avg_neg_log_prob)
        return ppl
