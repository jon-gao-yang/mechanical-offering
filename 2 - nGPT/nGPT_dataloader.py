# NOTE: based on the following scripts:
# https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py,
# https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
enc = tiktoken.get_encoding("gpt2") # init the tokenizer
dataset = 'shakespeare' # NOTE: CHECK THIS BEFORE RUNNING SCRIPT

if dataset == 'edufineweb':
    DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/edu_fineweb10B")
    data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    eot = enc.encode_single_token('<|endoftext|>') # end of text token

elif dataset == 'ultrachat':
    DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/ultrachat_200k")
    data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    eot = enc.encode_single_token('<|endoftext|>') # end of text token

    # TODO: CHANGE MESSAGE COLUMN NAME TO TEXT?

elif dataset == 'shakespeare':
    DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/shakespeare")
    data = load_dataset("text", data_files = os.path.join(DATA_CACHE_DIR, "input.txt"), split="train")
    eot = enc.encode_single_token('\n') # end of line token

else:
    raise ValueError("DATASET NAME NOT RECOGNIZED")

# data should have this structure:
# >>> data
# Dataset({
#     features: ['text'],
#     num_rows: 40000
# })

def tokenize(doc):
    # tokenizes a single document / shakepeare line and returns a numpy array of uint16 tokens
    tokens = [eot] # the special eot token delimits all documents / shakespeare lines
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, data, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{dataset}_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{dataset}_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])