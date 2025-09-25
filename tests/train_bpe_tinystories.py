import json
import time
import cProfile
import pstats
import io

from .adapters import run_train_bpe
from .common import DATA_PATH, gpt2_bytes_to_unicode


def train_bpe_tinystories():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on the tinystories dataset.
    """
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 30 * 60

cProfile.run('train_bpe_tinystories()')