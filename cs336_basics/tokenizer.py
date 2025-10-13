import json
import regex as re
from typing import Iterable, Iterator
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.vocab_rev = { value: token_id for token_id, value in vocab.items() }
        self.merges = merges
        self.special_tokens = [] if special_tokens is None else sorted(special_tokens, key=len, reverse=True)
        
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        unicode_to_byte = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            from_vocab_file = json.load(f)
            vocab = {
                token_id: bytes([unicode_to_byte[unicode] for unicode in vocab_item])
                for vocab_item, token_id in from_vocab_file.items()
            }
        with open(merges_filepath, encoding="utf-8") as f:
            merges_from_file = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([unicode_to_byte[unicode] for unicode in merge_token_1]),
                    bytes([unicode_to_byte[unicode] for unicode in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in merges_from_file
            ]
        return cls(vocab, merges, special_tokens)
        
        
    def encode(self, text: str) -> list[int]:
        parts = [text]
        if len(self.special_tokens) > 0:
            regex_delimiters = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            parts = re.split(regex_delimiters, text)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        result = []
        for part in parts:
            if part in self.special_tokens:
                result.append(self.vocab_rev[part.encode()])
                continue
            for match in re.finditer(PAT, part):
                token_ids = [self.vocab_rev[b.to_bytes()] for b in match.group(0).encode()]
                while True:
                    pairs = list(zip(token_ids[0:], token_ids[1:]))
                    has_merge = False
                    pair_to_merge_index = 0
                    index_in_merges = 0
                    for i in range(len(pairs)):
                        pair = pairs[i]
                        pair_bytes = (self.vocab[pair[0]], self.vocab[pair[1]])
                        try:
                            index_of_pair_in_merges = self.merges.index(pair_bytes)
                            if not has_merge or index_of_pair_in_merges < index_in_merges:
                                has_merge = True
                                pair_to_merge_index = i
                                index_in_merges = index_of_pair_in_merges
                        except ValueError:
                            # the pair cannot be merged, continue
                            continue
                    # do merge
                    if has_merge == True:
                        removed_token1 = token_ids.pop(pair_to_merge_index)
                        removed_token2 = token_ids.pop(pair_to_merge_index)
                        merged = self.vocab[removed_token1] + self.vocab[removed_token2]
                        token_ids.insert(pair_to_merge_index, self.vocab_rev[merged])
                        continue
                    break
                result.extend(token_ids)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result = b""
        for id in ids:
            result += self.vocab[id]
        return result.decode(errors='replace')
