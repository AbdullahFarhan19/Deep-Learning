from dataclasses import dataclass
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import tiktoken
import numpy as np
from hellaswag import render_example, iterate_examples


@dataclass
class GPTConfig:
    # Hyper Params 
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer    : int = 12
    n_head     : int = 12
    n_embd     : int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # Must be able to be splitted

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd) # Projected to a higher dimension to be split later on
        self.c_proj = nn.Linear(self.n_embd, self.n_embd) 

        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim = 2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # B, nh, T, hs --> B, T, C
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)
        self.ln_1  = nn.LayerNorm(config.n_embd) 
        self.ln_2  = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dict of tokens, pos, blocks and final layer norm
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd) 
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight sharing 
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02

        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            
            torch.nn.init.normal_(module.weights, mean = 0.0, std = std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn : p for pn, p in self.named_parameters()}
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for pn, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.ndim < 2]

        optim_groups = [
            {"params" : decay_params, "weight_decay" : weight_decay},
            {"params" : nodecay_params, "weight_decay" : 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"Number of decayed parameter tensors : {len(decay_params)}, Number of decayed parameters : {num_decay_params}")
        print(f"Number of non decayed parameter tensors : {len(nodecay_params)}, Number of Non Decayed parameters : {num_nodecay_params}")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW : {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8)

        return optimizer

    def forward(self, idx, targets = None):
        B, T = idx.shape 

        assert T <= self.config.block_size, f"arg idx passed has a block size of {T}, allowed block size is {self.config.block_size}"

        pos = torch.arange(T, dtype = torch.long, device = idx.device)
        tok_embd = self.transformer.wte(idx) # B, T, n_embd 
        pos_embd = self.transformer.wpe(pos) # T, n_embd
        x = pos_embd + tok_embd

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # B, T, vs

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # BT, vs and BT; viewing provides a single dimensional tensor of correct indices (from vocab_size) which is then used to index into logits

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt {model_type}")

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # ignore the params that are registered as buffers

        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()

        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith("attn.bias") and not k.endswith("attn.masked_bias")]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_hf_keys) == len(sd_keys), f"mismatched keys {len(sd_hf_keys)} != {len(sd_keys)}"

        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

ddp = int(os.environ.get("RANK", -1)) != -1 # Pytorch injects environment variables into your system when you use torchrun, one of these is rank, this basically checks if you are using torchrun
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend = "nccl") # initializes communication protocol between GPUs
    ddp_rank = int(os.environ["RANK"]) # current number of GPU
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # if you have multiple clusters or blocks of GPUs, for example 2 blocks of 4 each, this determines its local number, the 2nd GPU of the second block would have a local rank of 1
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # total number of participating GPUs
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # used for logging, you dont want all GPUs printing and logging

else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

def load_tokens(filename):
    npt = np.load(filename)
    npt = torch.tensor(npt, dtype = torch.long)
    return npt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"]

        root_dir = "edu_fineweb10B"
        shards = os.listdir(root_dir)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(root_dir, s) for s in shards]
        self.shards = shards
        
        assert len(shards) > 0, f"no shards found for split {split}"

        print(f"Loading {len(self.tokens)} number of tokens")
        print(f"Number of batches: {len(self.tokens) // (B*T)}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        buf = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        self.current_position += self.B * self.T * self.num_processes

        if (self.current_position + self.B * self.T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y

if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

batch_size = 524388 # nice number, 2**19
B = 16
T = 1024
assert batch_size % (B * T * ddp_world_size), "batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {batch_size}")
    print(f"Number of grad accum steps: {grad_accum_steps}")

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = (ddp_local_rank, ))

raw_model = model.module if ddp else model

train_loader = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "train")
val_loader   = DataLoaderLite(B = B, T = T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "val")

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 715 # 375 million tokens divided by 2 ^ 19 tokens for 715 steps
max_steps = 19073 # 10B tokens divided by 2 ^ 19 tokens for a total of 19073 steps

def get_lr(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps * max_lr # 1/10 of max_lr, 2/10 of max_lr, ..... i.e x/10 * max_lr which is a linear function
    
    elif step > max_steps:
        return min_lr 

    decay_rate = (step - warmup_steps) / (max_steps - warmup_steps) # crush step into a number between 0 and 1.0
    assert 0 <= decay_rate <= 1.0 
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate)) # math.cos goes from cos(0) = 1 to cos(pi) = -1, adding 1 makes it 0 to 2 and multiplying by 0.5 makes it 0 to 1.0
    return min_lr + coeff * (max_lr - min_lr) # when step is 10, coeff is 1 and we get max_lr, when step is 50 we get coeff = 0 and min_lr is returned

optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)
use_compile = False # unable to debug due to inability of running the program

log_dir = "log"
os.makedirs(log_dir, exist_ok = True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

if use_compile:
    model = torch.compile(model)

def get_most_likely_row(tokens, mask, logits):
    shift_tokens = tokens[..., 1:].contiguous()
    shift_logits = logits[..., -1, :].contiguous()
    shift_mask   = mask[..., 1:].contiguous()

    flat_shift_tokens = shift_tokens.view(-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction = "none")
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_masked_losses = shift_mask * shift_losses
    sum_loss = shift_masked_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    return avg_loss.argmin().item()

for step in range(max_steps):
    t0 = time.time()
    last_step = max_steps - 1

    # ====================== compute val loss =======================

    if step % 100 == 0:
        model.eval()
        val_loader.reset()

        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20

            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type = device, dtype = torch.bfloat16):
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            
        if ddp:
            dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

     # ====================== hellaswag =======================

    if (step % 250 == 0 or last_step) and (not use_compile):
        num_total = 0
        num_correct_norm = 0

        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        
        if ddp:
            num_total        = torch.tensor(num_total,        dtype = torch.long, device = device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype = torch.long, device = device)
            dist.all_reduce(num_total,        op = dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op = dist.ReduceOp.SUM)
            num_total        = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total

        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


     # ====================== sample from model =======================

    if ((step % 100 == 0 and step > 0) or step == max_steps - 1) and not use_compile:
        model.eval()
        num_return_sequences = 4
        max_length = 32

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode("Hello, I'm a language model")
        tokens = torch.tensor(tokens, dtype = torch.long)
        x = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # 4, 8
        x = x.to(device)
        sample_rng = torch.Generator(device = device)
        sample_rng.manual_seed(42 + ddp_rank)

        while x.size(1) <= max_length:
            with torch.no_grad():
                logits, loss = model(x) # 4, 8, vs

                logits = logits[:, -1, :] # 4, vs

                probs = F.softmax(logits, -1)

                topk_probs, topk_indices = torch.topk(probs, 50, -1) # 4, 50

                ix = torch.multinomial(topk_probs, num_samples = 1) # 4, 1

                xcol = torch.gather(topk_indices, -1, ix)

                x = torch.cat((x, xcol), -1)

        for i in range(num_return_sequences):      
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    
    # ====================== train model =======================

    model.train()
    loss_accum = 0.0
    optimizer.zero_grad()

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type = device, dtype = torch.bfloat16):
            logits, loss = model(x, y)

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()

    # ====================== print logistics =======================

    t1 = time.time()
    dt = t1 - t0
    
    tokens_per_sec = (train_loader.T * train_loader.B * grad_accum_steps * ddp_world_size) / dt

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
