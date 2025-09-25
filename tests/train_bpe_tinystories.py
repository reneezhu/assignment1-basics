import json
import time
import cProfile
import pstats
import io

from .adapters import run_train_bpe
from .common import DATA_PATH, gpt2_bytes_to_unicode


input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
if __name__ == '__main__':
    # 1. Create a profiler object
    pr = cProfile.Profile()

    # 2. Enable the profiler and run the function
    pr.enable()

    # start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    # end_time = time.time()

    # 3. Disable the profiler immediately after the function returns
    pr.disable()

    # 4. Process and print the statistics
    s = io.StringIO()
    sortby = 'cumulative'  # Sort by the cumulative time spent in the function and its sub-functions
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    print("\n--- Profiling Results (Sorted by Cumulative Time) ---")
    print(s.getvalue())
# cProfile.run('train_bpe_tinystories()')