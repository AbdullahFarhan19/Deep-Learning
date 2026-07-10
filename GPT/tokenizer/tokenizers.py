import regex as re
from GPT.tokenizer.helper import get_stats, merge

class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx:bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose = False):
        num_merges = vocab_size - 256

        tokens = text.encode("utf-8")
        ids = list(tokens)

        for i in range(num_merges):
            stats = get_stats(ids)
          
            pair = max(stats, key=stats.get)
            idx = 256 + i
            self.merges[pair] = idx
            self.vocab[256 + i] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = merge(ids, pair, idx)
            
            if verbose == True:
                    print(f"{i+1}/{num_merges} pair {pair} -> idx {idx}\n{self.vocab[idx]} had {stats[pair]} number of occurrences")

    def encode(self, text):
        tokens = text.encode("utf-8")
        ids = list(tokens)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key = lambda p : self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids
            
    def decode(self, ids):
        return (b"".join(self.vocab[idx] for idx in ids)).decode("utf-8", errors = "replace")
    
class RegexTokenizer:
    def __init__(self, pattern = None):
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {}
        self.vocab = {idx : bytes([idx]) for idx in range(256)}

    def train(self, text, vocab_size, verbose = False):
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_merges):
            stats = {}

            for chunk in ids:
                stats = get_stats(chunk, stats)
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            self.merges[pair] = idx
            self.vocab[256 + i] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = [merge(chunk, pair, idx) for chunk in ids]
                
            if verbose == True:
                print(f"{i+1}/{num_merges} pair {pair} -> idx {idx}\n{self.vocab[idx]} had {stats[pair]} number of occurrences")

    def encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        newids = []

        for chunk in ids:
            while len(chunk) >= 2:
                stats = get_stats(chunk)

                pair = min(stats, key = lambda p : self.merges.get(p, float("inf")))

                if pair not in self.merges:
                    break

                idx = self.merges[pair]
                chunk = merge(chunk, pair, idx)

            newids.extend(chunk)

        return newids
    
    def decode(self, ids):
        return (b"".join(self.vocab[idx] for idx in ids)).decode("utf-8", errors = "replace")

