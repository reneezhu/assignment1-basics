import os
import regex as re
from typing import BinaryIO

    

class Pretoken:
    def __init__(self, text):
        self.text = text
        self.text_bytes = text.encode()
        # list of token_id
        self.tokens = list(self.text_bytes)
        # token_pair -> count
        self.token_pairs = {}
        self.init_token_pairs()
    
    def init_token_pairs(self):
        for i in range(len(self.tokens)-1):
            token_pair = (self.tokens[i], self.tokens[i+1])
            if token_pair in self.token_pairs:
                self.token_pairs[token_pair] += 1
            else:
                self.token_pairs[token_pair] = 1
    
    # update token_pairs based on tokens and return the diff
    def update_token_pairs(self):
        old_token_pairs = self.token_pairs
        updated_token_pairs = {}
        token_pairs_diff = {}
        for i in range(len(self.tokens)-1):
            token_pair = (self.tokens[i], self.tokens[i+1])
            if token_pair in updated_token_pairs:
                updated_token_pairs[token_pair] += 1
            else:
                updated_token_pairs[token_pair] = 1
        for pair, count in updated_token_pairs.items():
            old_count = old_token_pairs.pop(pair, 0)
            if count != old_count:
                token_pairs_diff[pair] = count - old_count
        for pair, old_count in old_token_pairs.items():
            token_pairs_diff[pair] = 0 - old_count
        self.token_pairs = updated_token_pairs
        return token_pairs_diff

    # Apply the merge, update tokens, token_pairs and return the diff
    def merge_tokens(self, merge: tuple[int, int], new_token_id: int) -> dict[tuple[int, int], int]:
        updated_tokens = []
        i = 0
        while i < len(self.tokens):
            # last token, won't be merged, directly append
            if i == len(self.tokens) - 1:
                updated_tokens.append(self.tokens[i])
                i += 1
                continue
            if self.tokens[i] == merge[0] and self.tokens[i+1] == merge[1]:
                updated_tokens.append(new_token_id)
                i += 1 # The next token has been merged, so skip it.
            else:
                updated_tokens.append(self.tokens[i])
            i += 1
        if self.tokens != updated_tokens:
            self.tokens = updated_tokens
            diff = self.update_token_pairs()
            return diff
        return {}
        

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_chunk(chunk: str, special_tokens: list[str]) -> list[str]:
    regex_delimiters = "|".join(map(re.escape, special_tokens))
    result = re.split(regex_delimiters, chunk)
    return result

def init_pairs(pretokens: dict[Pretoken, int]) -> dict[tuple[bytes, bytes], int]:
    # key is tuple <token1>, <token2>, value is the count
    pair_count = {}
    # key is tuple <token1>, <token2>, value is a set of pretokens.
    pair_to_pretoken_map = {}
    for pretoken, count in pretokens.items():
        for pair, count_in_pretoken in pretoken.token_pairs.items():
            if pair in pair_count:
                pair_count[pair] += count * count_in_pretoken
            else:
                pair_count[pair] = count * count_in_pretoken
            if pair in pair_to_pretoken_map:
                pair_to_pretoken_map[pair].add(pretoken)
            else:
                pair_to_pretoken_map[pair] = {pretoken}
    return pair_count, pair_to_pretoken_map

def apply_merge(
        pretokens: dict[Pretoken, int],
        pair_count: dict[tuple[int, int], int],
        pair_to_pretoken_map: dict[tuple[int, int], set[Pretoken]],
        pair_to_merge: tuple[int, int],
        new_token_id: int):
    pretokens_need_to_be_updated = pair_to_pretoken_map.pop(pair_to_merge)
    for pretoken in pretokens_need_to_be_updated:
        count = pretokens[pretoken]
        token_pairs_diff = pretoken.merge_tokens(pair_to_merge, new_token_id)
        for update_pair, update_pair_count in token_pairs_diff.items():
            if update_pair in pair_count:
                pair_count[update_pair] += count * update_pair_count
            else:
                pair_count[update_pair] = count * update_pair_count
            # If the pair is not in this pretoken, remove it from the map
            if update_pair_count < 0 and update_pair != pair_to_merge and update_pair not in pretoken.token_pairs:
                pair_to_pretoken_map[update_pair].remove(pretoken)
                continue
            if update_pair in pair_to_pretoken_map:
                pair_to_pretoken_map[update_pair].add(pretoken)
            else:
                pair_to_pretoken_map[update_pair] = {pretoken}


def train_bpe(pretokens: dict[Pretoken, int], vocab_size: int, special_tokens: list[str]):
    vocab = {i: bytes([i]) for i in range(0, 256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()
    # list[typle[bytes, bytes]]
    merges = []
    # pair_count: key is tuple <token1>, <token2>, value is the count
    pair_count, pair_to_pretoken_map = init_pairs(pretokens)
    while len(vocab) < vocab_size:
        max_pair = max(pair_count, key=lambda k: (pair_count[k], (vocab[k[0]], vocab[k[1]])))
        merge = (vocab[max_pair[0]], vocab[max_pair[1]])
        merged_token = merge[0] + merge[1]
        merges.append(merge)
        new_token_id = len(vocab)
        vocab[new_token_id] = merged_token
        apply_merge(pretokens, pair_count, pair_to_pretoken_map, max_pair, new_token_id)
        if (len(vocab) % 100 == 0):
            print("Current vocab size: %d", len(vocab))
    return vocab, merges

        
def pretokenization(docs: list[str]) -> dict[Pretoken, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # dict[str, pretoken]
    str_to_pretokens_map = {}
    # dict[pretoken, int]
    pretokens = {} # stores pretoken and their count
    for doc in docs:
        for match in re.finditer(PAT, doc):
            pretoken_str = match.group(0)
            if pretoken_str in str_to_pretokens_map:
                pretoken = str_to_pretokens_map[pretoken_str]
                pretokens[pretoken] += 1
            else:
                pretoken = Pretoken(pretoken_str)
                pretokens[pretoken] = 1
                str_to_pretokens_map[pretoken_str] = pretoken
    return pretokens

    
    
    
## Usage
# with open("assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#     # 0 is the "<|endoftext|>" token， 1-256 representing the bytes
#     # vocab = {i: bytes([i-1]) for i in range(256)}
#     # vocab[0] = "<|endoftext|>"

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start) 
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token
#         # Split the chunk by speical token
#         docs = split_chunk(chunk, r"<\|endoftext\|>")
#         pretokens = pretokenization(docs)
#         train_bpe(pretokens, 1000)