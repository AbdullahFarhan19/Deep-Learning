import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir) # extract dir name from current file and then append local_dir to it
os.makedirs(DATA_CACHE_DIR, exist_ok = True) # Prevents file from crashing if a file of the same name exists on the disk, does not create a dir if it already exists

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train") # get the 10BT training sample 

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"] # returns 50256

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"])) # encode the ordinary (non special) tokens of the raw text of the doc
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"

    tokens_np_uint16 = tokens_np.astype(np.uint16) # force to treat every integer as unint16 rather than 64 bits integers
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np) # highly optimized function that writes the array into the file using serialized binary

nprocs = max(1, os.cpu_count() // 2) # gets number of CPU cores

with mp.pool(nprocs) as pool: # create a "pool" of processes, context manager safely frees memory after it has been used
    shard_index = 0
    all_tokens_np = np.empty((shard_size, ), dtype = np.uint16) # allocate 100M random garbage values to be written in later
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize = 16): # 16 docs at once, apply tokenize to fw
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens # allocate tokens to memory chunk that we reserved
            token_count += len(tokens)

            if progress_bar is None:
                progress_bar = tqdm(total = shard_size, unit = "tokens", desc = f"Shard {shard_index}")
            progress_bar.update(len(tokens))

        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np) # write to data file

            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:] # reinitialize using remainder tokens
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])       

