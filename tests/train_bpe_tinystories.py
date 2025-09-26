import json
import time
import cProfile
import pstats
import io

from .adapters import run_train_bpe
from .common import DATA_PATH, OUTPUT_PATH, gpt2_bytes_to_unicode

def bytes_to_str(gpt2_byte_encoder, bytes_input):
    result = ""
    for b in bytes_input:
        result += gpt2_byte_encoder[b]
    return result


input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
if __name__ == '__main__':
    # 1. Create a profiler object
    pr = cProfile.Profile()

    # 2. Enable the profiler and run the function
    pr.enable()

    vocab, merge = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # 3. Disable the profiler immediately after the function returns
    pr.disable()

    # 4. Process and print the statistics
    s = io.StringIO()
    sortby = 'cumulative'  # Sort by the cumulative time spent in the function and its sub-functions
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    print("\n--- Profiling Results (Sorted by Cumulative Time) ---")
    print(s.getvalue())

    # Save the vocab and merge into file
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    with open(OUTPUT_PATH / "tinystories-vocab.json", 'w') as f:
        # Transform the vocab to out format, key is the string, and value is the token id
        output_dict = {
            bytes_to_str(gpt2_byte_encoder, token_bytes): token_id
            for token_id, token_bytes in vocab.items()
        }
        json.dump(output_dict, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_PATH / "tinystories-merges.txt", 'w') as f:
        for pair in merge:
            str0 = bytes_to_str(gpt2_byte_encoder, pair[0])
            str1 = bytes_to_str(gpt2_byte_encoder, pair[1])
            f.write(f"{str0} {str1}\n")