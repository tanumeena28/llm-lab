"""
Byte-Pair Encoding (BPE) Tokenizer Implementation from Scratch.
Optimized for rapid lab experiments using integer token IDs.
"""

import os
import json
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm


class BPETokenizer:
    def __init__(self, vocab_size: int = 30000, unk_token: str = "<unk>", pad_token: str = "<pad>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.end_of_word = "</w>"
        
        self.special_tokens = [pad_token, unk_token, "<bos>", "<eos>"]
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.merge_rules: List[Tuple[int, int, int]] = []

    def train(self, sentences: List[str], verbose: bool = True):
        if verbose:
            print(f"\n--- Training BPE Tokenizer (Target Vocab: {self.vocab_size:,}) ---")
        
        word_counts = Counter()
        for sentence in sentences:
            words = sentence.strip().split()
            for word in words:
                if word:
                    word_counts[word] += 1
                    
        # Retain top 5,000 most frequent unique words for efficient subword vocabulary learning
        most_common = word_counts.most_common(5000)
        
        chars = set()
        for w, _ in most_common:
            chars.update(list(w))
        chars.add(self.end_of_word)
        
        all_vocab = list(self.special_tokens) + sorted(list(chars))
        self.vocab = {tok: i for i, tok in enumerate(all_vocab)}
        self.inv_vocab = {i: tok for i, tok in enumerate(all_vocab)}
        
        word_freqs = {}
        for word, freq in most_common:
            char_ids = tuple([self.vocab[c] for c in word] + [self.vocab[self.end_of_word]])
            word_freqs[char_ids] = freq

        num_merges_needed = self.vocab_size - len(self.vocab)
        if num_merges_needed <= 0:
            return

        pbar = tqdm(total=num_merges_needed, disable=not verbose, desc="BPE Merges")

        for merge_idx in range(num_merges_needed):
            pair_counts = defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair_counts[(word_tuple[i], word_tuple[i+1])] += freq

            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break
                
            p1_id, p2_id = best_pair
            new_token_str = self.inv_vocab[p1_id] + self.inv_vocab[p2_id]
            new_id = len(self.vocab)
            
            self.vocab[new_token_str] = new_id
            self.inv_vocab[new_id] = new_token_str
            self.merges[best_pair] = new_id
            self.merge_rules.append((p1_id, p2_id, new_id))

            new_word_freqs = {}
            for word_tuple, freq in word_freqs.items():
                if p1_id in word_tuple:
                    new_tuple = []
                    i = 0
                    while i < len(word_tuple):
                        if i < len(word_tuple) - 1 and word_tuple[i] == p1_id and word_tuple[i + 1] == p2_id:
                            new_tuple.append(new_id)
                            i += 2
                        else:
                            new_tuple.append(word_tuple[i])
                            i += 1
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
        
        eow_id = self.vocab.get(self.end_of_word)
        ids = [self.vocab.get(c, self.vocab.get(self.unk_token)) for c in word] + [eow_id]
        
        while len(ids) > 1:
            pairs = [(ids[i], ids[i+1]) for i in range(len(ids) - 1)]
            merge_candidates = [p for p in pairs if p in self.merges]
            if not merge_candidates:
                break
                
            best_pair = min(merge_candidates, key=lambda p: self.merges[p])
            new_id = self.merges[best_pair]
            p1_id, p2_id = best_pair
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p1_id and ids[i + 1] == p2_id:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            
        tokens = [self.inv_vocab.get(i, self.unk_token) for i in ids]
        return tokens

    def tokenize(self, sentence: str) -> List[str]:
        words = sentence.strip().split()
        subwords = []
        for w in words:
            toks = self.tokenize_word(w)
            for t in toks:
                if t not in self.vocab and t != self.end_of_word:
                    subwords.append(self.unk_token)
                else:
                    subwords.append(t)
        return subwords

    def encode(self, sentence: str) -> List[int]:
        tokens = self.tokenize(sentence)
        return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(tid, self.unk_token) for tid in token_ids]
        text = "".join(tokens)
        text = text.replace(self.end_of_word, " ")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': [list(m) for m in self.merge_rules],
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_rules = [tuple(m) for m in data['merges']]
        self.merges = {(m[0], m[1]): m[2] for m in self.merge_rules}
        self.special_tokens = data.get('special_tokens', self.special_tokens)
