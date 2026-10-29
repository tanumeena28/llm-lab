"""
WordPiece Tokenizer Implementation from Scratch.
Optimized for rapid lab experiments using likelihood scoring.
"""

import os
import json
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm


class WordPieceTokenizer:
    def __init__(
        self,
        vocab_size: int = 30000,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        prefix: str = "##"
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.prefix = prefix
        
        self.special_tokens = [pad_token, unk_token, cls_token, sep_token, "[MASK]"]
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}

    def _build_word_freqs(self, sentences: List[str], max_vocab_words: int = 5000) -> Dict[Tuple[str, ...], int]:
        word_counts = Counter()
        for sentence in sentences:
            words = sentence.strip().split()
            for word in words:
                if word:
                    word_counts[word] += 1
                    
        most_common = word_counts.most_common(max_vocab_words)
        
        freq_dict = {}
        for word, count in most_common:
            subwords = [word[0]] + [f"{self.prefix}{c}" for c in word[1:]]
            freq_dict[tuple(subwords)] = count
            
        return freq_dict

    def train(self, sentences: List[str], verbose: bool = True):
        if verbose:
            print(f"\n--- Training WordPiece Tokenizer (Target Vocab: {self.vocab_size:,}) ---")
            
        word_freqs = self._build_word_freqs(sentences)
        
        base_vocab: Set[str] = set()
        for word_tuple in word_freqs.keys():
            base_vocab.update(word_tuple)
            
        all_vocab = list(self.special_tokens) + sorted(list(base_vocab))
        self.vocab = {token: i for i, token in enumerate(all_vocab)}
        self.inv_vocab = {i: token for token, i in self.vocab.items()}
        
        num_merges_needed = self.vocab_size - len(self.vocab)
        if num_merges_needed <= 0:
            return

        pbar = tqdm(total=num_merges_needed, disable=not verbose, desc="WordPiece Merges")

        for merge_idx in range(num_merges_needed):
            single_counts = defaultdict(int)
            pair_counts = defaultdict(int)
            
            for word_tuple, freq in word_freqs.items():
                for tok in word_tuple:
                    single_counts[tok] += freq
                for i in range(len(word_tuple) - 1):
                    pair_counts[(word_tuple[i], word_tuple[i+1])] += freq

            if not pair_counts:
                break
                
            best_pair = None
            best_score = -1.0
            
            for (p1, p2), count_ab in pair_counts.items():
                score = count_ab / (single_counts[p1] * single_counts[p2])
                if score > best_score:
                    best_score = score
                    best_pair = (p1, p2)
                    
            if best_pair is None or best_score <= 0:
                break
                
            p1, p2 = best_pair
            merged_token = p1 + (p2[len(self.prefix):] if p2.startswith(self.prefix) else p2)
            
            new_id = len(self.vocab)
            self.vocab[merged_token] = new_id
            self.inv_vocab[new_id] = merged_token

            new_word_freqs = {}
            for word_tuple, freq in word_freqs.items():
                if p1 in word_tuple:
                    new_tuple = []
                    idx = 0
                    while idx < len(word_tuple):
                        if idx < len(word_tuple) - 1 and word_tuple[idx] == p1 and word_tuple[idx + 1] == p2:
                            new_tuple.append(merged_token)
                            idx += 2
                        else:
                            new_tuple.append(word_tuple[idx])
                            idx += 1
                    t = tuple(new_tuple)
                    new_word_freqs[t] = new_word_freqs.get(t, 0) + freq
                else:
                    new_word_freqs[word_tuple] = new_word_freqs.get(word_tuple, 0) + freq
                    
            word_freqs = new_word_freqs
            pbar.update(1)
            
        pbar.close()
        if verbose:
            print(f"Final Vocab Size: {len(self.vocab):,}")

    def tokenize_word(self, word: str) -> List[str]:
        if not word:
            return []
        subwords = []
        start = 0
        end = len(word)
        
        while start < end:
            cur_end = end
            matched_tok = None
            
            while start < cur_end:
                substr = word[start:cur_end]
                if start > 0:
                    substr = f"{self.prefix}{substr}"
                if substr in self.vocab:
                    matched_tok = substr
                    break
                cur_end -= 1
                
            if matched_tok is None:
                subwords.append(self.unk_token)
                start += 1
            else:
                subwords.append(matched_tok)
                start = cur_end
                
        return subwords

    def tokenize(self, sentence: str) -> List[str]:
        words = sentence.strip().split()
        subwords = []
        for w in words:
            subwords.extend(self.tokenize_word(w))
        return subwords

    def encode(self, sentence: str) -> List[int]:
        tokens = self.tokenize(sentence)
        return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(tid, self.unk_token) for tid in token_ids]
        text_pieces = []
        for t in tokens:
            if t in self.special_tokens:
                continue
            if t.startswith(self.prefix):
                text_pieces.append(t[len(self.prefix):])
            else:
                text_pieces.append(f" {t}")
        text = "".join(text_pieces).strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'prefix': self.prefix
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        self.prefix = data.get('prefix', self.prefix)
