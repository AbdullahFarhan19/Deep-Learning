import torch
import torch.nn.functional as F
import random

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {s:(i+1) for i,s in enumerate(chars)}
itos = {(i+1):s for i,s in enumerate(chars)}

block_size = 5

itos[0] = "."
stoi["."] = 0

def build_dataset(words):
    X = []
    Y = []

    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            X.append(context)
            Y.append(stoi[ch])

            context = context[1:] + [stoi[ch]]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

C = torch.randn((32, 30))
W1 = torch.randn((150, 1000)); b1 = torch.randn(1000)
W2 = torch.randn((1000,27)); b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

for i in range(200000):
    ix = torch.randint(0, Xtr.shape[0], (64,))

    emb = C[Xtr[ix]]
    h = (emb.view(-1, 150) @ W1 + b1).tanh()
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 100000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

emb = C[Xtr]
h = (emb.view(-1, 150) @ W1 + b1).tanh()
logits = h @ W2 + b2

losstr = F.cross_entropy(logits, Ytr)

emb = C[Xdev]
h = (emb.view(-1, 150) @ W1 + b1).tanh()
logits = h @ W2 + b2

lossdev = F.cross_entropy(logits, Ydev)

print(losstr.item())

print(lossdev.item())

for i in range(20):
    out = []

    context = [0] * block_size

    while True:
        emb = C[torch.tensor(context)]
        h = (emb.view(1,-1) @ W1 + b1).tanh()
        logits = h @ W2 + b2

        probs = F.softmax(logits, dim = 1)

        ix = torch.multinomial(probs, num_samples = 1).item()

        if ix == 0:
            break

        out.append(ix)
        context = context[1:] + [ix]

    print(''.join(itos[i] for i in out))  