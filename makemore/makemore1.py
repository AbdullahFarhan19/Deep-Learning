import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(chars)}
itos = {i+1:s for i,s in enumerate(chars)}

stoi["."] = 0
itos[0] = "."

N = torch.zeros(27,27, dtype = torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]

    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

P = (N + 1).float()
P /= P.sum(dim = 1, keepdims = True)

g = torch.Generator().manual_seed(2147483647)
ix = 0
for i in range(10):
    outs = []
    while True:
        p = P[ix]

        ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()

        outs.append(itos[ix])

        if ix == 0:
            break

    # print(''.join(outs))

log_likelihood = 0.0
n = 0

for w in words:
    chars = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n+=1
        # print(f"{ch1=} {ch2=} {log_prob=}")

# print(f"{-log_likelihood=}")
# print(f"{-log_likelihood/n}")

xs = []
ys = []

for w in words:
    chars = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# print(len(xs))

num = xs.nelement()

W = torch.randn((27,27), requires_grad = True)

for i in range(100):
    xenc = F.one_hot(xs, num_classes = 27).float()
    logits = xenc @ W # for each xenc, the respective probabilities of each next character
    counts = logits.exp()
    probs = counts / counts.sum(dim = 1, keepdims = True)

    loss = -probs[torch.arange(num), ys].log().mean()

    W.grad = None # Basically just zero but more efficient
    loss.backward()
    W.data += -50 * W.grad

    if i % 10 == 0:
        print(loss.data)
