import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {s:(i+1) for i,s in enumerate(chars)}
itos = {(i+1):s for i,s in enumerate(chars)}

block_size = 3
vocab_size = len(itos)

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

class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.w = torch.randn(fan_in, fan_out) / (fan_in)**0.5
        self.b = torch.zeroes(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.w + (self.b if self.b is not None else 0)
        return self.out 
    
    def parameters(self):
        return [self.w] + ([self.b] if self.b is not None else [])

class BatchNorm1D:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(dim = 0, keepdim = True)
            xvar = x.var(dim = 0, keepdim = True, unbiased = True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        norm = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * norm + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * (1 - self.momentum) + xmean * self.momentum
                self.running_var = self.running_var * (1 - self.momentum) + xvar * self.momentum

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

n_emb = 10
n_hidden = 100

C = torch.randn(vocab_size, n_emb)

layers = [
    Linear(block_size * n_emb, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(          n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(          n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(          n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(          n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(          n_hidden, vocab_size), BatchNorm1D(vocab_size)
]

with torch.no_grad():
    layers[-1].w *= 0.1

    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.w *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]

for p in parameters:
    p.requires_grad = True

epochs = 10000
batch_size = 32
ud = []

for i in range(epochs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    Xb = Xtr[ix] # batch_size, block_size
    Yb = Ytr[ix] # batch_size, 1

    emb = C[Xb] # batch_size, block_size, emb_size
    x = emb.view(emb.shape[0], -1) # batch_size, block_size * emb_size

    for layer in layers:
        x = layer(x)
    
    loss = F.cross_entropy(x, Yb)

    for layer in layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 # if i < 100000 else 0.01

    with torch.no_grad():
        ud.append([(p.grad * lr).std()/(p.data.std()).log10().item() for p in parameters])

    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000:
        print(f"{i}/{epochs}: {loss.item()}")

plt.figure(figsize = (20,4))
legends1 = []

for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out

        hy, hx = torch.histogram(t, density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends1.append(f"layer {i} {layer.__class__.__name__}")

plt.legend(legends1)
plt.title("activation distribution")

plt.figure(figsize = (20,4))
legends2 = []

for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
        t = layer.out.grad

        hy, hx = torch.histogram(t, density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends2.append(f"{i} {layer.__class__.__name__}")

plt.legend(legends2)
plt.title("gradients distribution")

plt.figure(figsize = (20,4))
legends3 = []

for i, p in enumerate(parameters):
    if p.ndim == 2:
        t = p.grad

        hy, hx = torch.histogram(t, density = True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends3.append(f"{i} {tuple(p.shape())}")

plt.legend(legends3)
plt.title("gradients distribution")

plt.figure(figsize = (20,4))
legends4 = []

for i, p in enumerate(parameters):
    if p.ndim == 2:
        plt.plot(ud[j][i] for j in len(ud))

        
        legends4.append(f"param {i}")

plt.plot((0, len(ud)), [-3,-3], "k") 
plt.legend(legends4)
plt.title("ud distribution")

@torch.no_grad
def split_loss(split):
    x, y = {
        "train" : (Xtr, Ytr),
        "dev"   : (Xdev, Ydev),
        "test"  : (Xte, Yte)

    }[split]

    emb = C[x]
    x = emb.view(emb.shape[0], -1)
    
    for layer in layers:
        x = layer(x)

    loss = F.cross_entropy(x, y)

    print(split, loss.item())

for layer in layers:
    layers.training = False
split_loss("train")
split_loss("dev")

for i in range(20):
    out = []

    context = [0] * block_size

    while True:
        emb = C[torch.tensor(context)]
        x = emb.view(1, -1)
        
        for layer in layers:
            x = layer(x)

        probs = F.softmax(x, dim = 1)

        ix = torch.multinomial(probs, num_samples = 1).item()

        if ix == 0:
            break

        out.append(ix)
        context = context[1:] + [ix]

    print(''.join(itos[i] for i in out))  