import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size = 1024):
    resp = requests.get(url, stream = True) # stream = True prevents the file from being loaded in the RAM all at once
    total = int(resp.headers.get("context-length", 0)) # extracts file size from the HTTP header

    with open(fname, "wb") as file, tqdm( # write binary mode
        desc = fname, 
        total = total,
        unit = "iB", # unit is binary bytes
        unit_scale = True, # does unit scaling, for example 1M bytes could become 1 MiB                         
        unit_divisor = 1024,
        
    ) as bar:
        
        for data in resp.iter_content(chunk_size = chunk_size): # incoming 1024 byte stream assigned to var data
            size = file.write(data)
            bar.update(size)
 
# github urls

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok = True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.json")

    if not os.path.exists(data_filename):
        print(f"downloading {data_url} to {data_filename}")
        download_file(data_url, data_filename)

def iterate_examples(split):
    download(split)

    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line) 
            yield example

def render_examples(example):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": []
    }

    ctx_tokens = enc(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []

    for end in endings:
        end_tokens = enc.encode("" + end) # gets the end token and prepends a space since this will be appended to the ctx token
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens)) # 0s for ctx and 1s for endings
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows) # tok rows may be diff sizes
    tokens  = torch.zeros((4, max_len), dtype = torch.long)
    mask    = torch.zeros((4, max_len), dtype = torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)]  = torch.tensor(mask_row)

    return data, tokens, mask, label

@torch.no_grad
def evaluate(model_type, device):
    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)

    num_correct_norm = 0
    num_correct      = 0
    num_total        = 0

    for example in iterate_examples("val"): # there are 10,042 examples in total in val
        data, tokens, mask, label = render_examples(example)
        tokens, mask = tokens.to(device), mask.to(device)

        # tokens -> (4, max_len)
        # logits -> (4, max_len, vocab_size)
        
        logits = model(tokens).logits
        shift_tokens = tokens[..., 1:]
        shift_logits = (logits[..., :-1, :]).contiguous() # get all logits except the last one, since the last one has no prediction after it
        flat_shift_tokens = shift_tokens.view(-1)
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(1))

        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # prevents reduction into one single value 
        shift_losses.view(shift_tokens.size(0), -1) # converts the 1D losses tensor into a (4, max_len)
        shift_mask = (mask[..., 1:]).contiguous()
        shift_masked_losses = shift_mask * shift_losses # mask the losses such that they only show the ending tokens

        sum_loss = shift_masked_losses.sum(dim = 1) # 4, 1
        avg_loss = sum_loss / shift_mask.sum(dim = 1) # 4, 1

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

if __name__ == "__main__": # checks if the script is being run from the cmd
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use") # user can specify -m or --model_type, expects a string
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use") 
    args = parser.parse_args()
    evaluate(args.model_type, args.device)