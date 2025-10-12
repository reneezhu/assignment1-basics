import json
import regex as re

from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.vocab_rev = { value: token_id for token_id, value in vocab }
        self.merges = merges
        self.special_tokens = special_tokens
        
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, encoding="utf-8") as f:
            from_vocab_file = json.load(f)
            vocab = {
                token_id: bytes([gpt2_byte_decoder[token] for token in vocab_item])
                for vocab_item, token_id in from_vocab_file.items()
            }
        with open(merges_filepath, encoding="utf-8") as f:
            merges_from_file = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in merges_from_file
            ]
        return cls(vocab, merges, special_tokens)
        
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, text):
            tokens = match.group(0).encode()
            continue_merge = True
            while continue_merge:
                pairs = zip(tokens[0:], tokens[1:])
                continue_merge = False
                for i in range(len(pairs)):
                    pair = pairs[i]
                    if pair in self.merges:
                        # do merge
                        merged_bytes = pair[0] + pair[1]
                        merged_token = self.vocab_rev[merged_bytes]
                        tokens[i] = merged_token
                        
                        continue_merge = True
                        break

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        result = b""
        for id in ids:
            result += self.vocab[id]
        return result.decode()