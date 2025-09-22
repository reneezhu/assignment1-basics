import os
import regex as re
from typing import BinaryIO

    

class Pretoken:
    def __init__(self, text):
        self.text = text
        self.text_bytes = text.encode()
        self.tokens = self.text_bytes
        # token_pair -> count
        self.token_pairs = {}
        self.init_tokens()
    
    def init_tokens(self):
        for i in range(len(self.text_bytes)-1):
            token_pair = (self.text_bytes[i], self.text_bytes[i+1])
            if token_pair in self.token_pairs:
                self.token_pairs[token_pair] += 1
            else:
                self.token_pairs[token_pair] = 1

    def merge_tokens(self, merge: tuple[bytes, bytes], new_token_id: int):
        updated_tokens = []
        # {((pre_pair), (now_pair)): count}
        token_updates = {}
        updated_token_pairs = {}
        for i in range(len(self.tokens) - 1):
            # print(i)
            if self.tokens[i] == merge[0] and self.tokens[i+1] == merge[1]:
                token = bytes(merge)
                if i !=0:
                    pre_pair = (self.tokens[i-1], self.tokens[i])
                    post_pair = (updated_tokens[-1], new_token_id)
                    # update the previous token pair
                    updated_token_pairs[pre_pair] -= 1
                    if post_pair in updated_token_pairs:
                        updated_token_pairs[post_pair] += 1
                    else:
                        updated_token_pairs[post_pair] = 1
                    # if previous token was merged, the current token is also merged, we need to update the token_updates
                    # TODO
                    updates = (pre_pair, post_pair)
                    if updates in token_updates:
                        token_updates[updates] += 1
                    else:
                        token_updates[updates] = 1
                if i < len(self.tokens) - 2:
                    pre_pair = (self.tokens[i+1], self.tokens[i+2])
                    #TODO: need to consider the case if the next token also need to be merged.
                    post_pair = (token, self.tokens[i+2])
                    updates = (pre_pair, post_pair)
                    if updates in token_updates:
                        token_updates[updates] += 1
                    else:
                        token_updates[updates] = 1
                updated_tokens.append(bytes(merge))
                i += 1 # The next token has been merged, so skip it.
            else:
                updated_tokens.append(self.tokens[i])
                # We assume the next token is not updated at here, if next token need to be updated, we will update the value when processing the next token.
                pair = (self.tokens[i], self.tokens[i+1])
                if pair in updated_token_pairs:
                    updated_token_pairs[pair] += 1
                else:
                    updated_token_pairs[pair] = 1
        return token_updates
        

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

def split_chunk(chunk: str, split_special_token: str) -> list[str]:
    result = re.split(split_special_token, chunk)
    return result

def load_dataset(input_path: str, special_token: str) -> list[list[str]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, special_token.encode())
        dataset = []

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            # Split the chunk by speical token
            docs = re.split(re.escape(special_token), chunk)
            dataset.append(docs)
    return dataset

def init_pairs(pretokens: dict[Pretoken, int]) -> dict[tuple[bytes, bytes], int]:
    # key is tuple <token1>, <token2>, value is the count
    pair_count = {}
    for pretoken, count in pretokens.items():
        for pair, indexes in pretoken.token_pairs.items():
            if pair in pair_count:
                pair_count[pair] += count * len(indexes)
            else:
                pair_count[pair] = count * len(indexes)
    return pair_count

def merge(pretokens: dict[Pretoken, int], pair_count: dict[tuple[bytes, bytes], int], pair_to_merge: tuple[bytes, bytes], new_token_id):
    for (pretoken, count) in pretokens.items():
        pairs_to_update = pretoken.merge_tokens(pair_to_merge, new_token_id)
        for update_pair, update_pair_count in pairs_to_update.items():
            pre_pair, post_pair = update_pair
            pair_count[pre_pair] -= count * update_pair_count
            if post_pair in pair_count:
                pair_count[post_pair] += count * update_pair_count
            else:
                pair_count[post_pair] = count * update_pair_count
            
                   
                

def train_bpe(pretokens: dict[Pretoken, int], vocab_size: int):
    vocab = {i: bytes([i]) for i in range(0, 256)}
    # list[typle[bytes, bytes]]
    merges = []
    # key is tuple <token1>, <token2>, value is the count
    pair_count = init_pairs(pretokens)
    while len(vocab) < vocab_size:
        max_pair = max(pair_count, key=pair_count.get)
        max_value = pair_count.pop(max_pair)
        print(max_pair)
        print(max_value)
        merges.append(max_pair)
        new_token_id = len(vocab)
        vocab[new_token_id] = bytes(max_pair)
        merge(pretokens, pair_count, max_pair, new_token_id)

        
def pretokenization(docs: list[str]) -> dict[Pretoken, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = {} # stores pretoken and their count
    for doc in docs:
        for match in re.finditer(PAT, doc):
            pretoken = Pretoken(match.group(0))
            if pretoken in pretokens:
                pretokens[pretoken] += 1
            else:
                pretokens[pretoken] = 1
    return pretokens

    
    
    
# train_bpe("assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 270, "<|endoftext|>")

## Usage
with open("assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # 0 is the "<|endoftext|>" token， 1-256 representing the bytes
    # vocab = {i: bytes([i-1]) for i in range(256)}
    # vocab[0] = "<|endoftext|>"

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start) 
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        # Split the chunk by speical token
        docs = split_chunk(chunk, r"<\|endoftext\|>")
        pretokens = pretokenization(docs)
        # Get top 5 items by value
        top_5 = sorted(pretokens.items(), key=lambda item: item[1], reverse=True)[:5]

        # Print them
        for key, value in top_5:
            print(f"{key}: {value}")
        train_bpe(pretokens, 1000)