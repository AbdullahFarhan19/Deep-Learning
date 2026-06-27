import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

words = open("makemore/names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {s:(i+1) for i,s in enumerate(chars)}
itos = {(i+1):s for i,s in enumerate(chars)}

itos[0] = "."
stoi["."] = 0

block_size = 8
vocab_size = len(itos)

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
        self.b = torch.zeros(fan_out) if bias else None
    
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
            if x.ndim == 2:
                dim = 0
            else:
                dim = (0, 1)

            xmean = x.mean(dim, keepdim = True)
            xvar = x.var(dim, keepdim = True, unbiased = True)

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

class Embedding:
    def __init__(self, vocab_size, n_emb):
        self.weight = torch.randn(vocab_size, n_emb)
    
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]
    
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, C, E = x.shape
        x = x.view(B, E//self.n, C * self.n)

        if x.shape[1] == 1:
            x.squeeze(dim = 1)
        
        self.out = x
        return self.out
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

n_emb = 10
n_hidden = 100

model = Sequential([
    Embedding(vocab_size, n_emb),
    FlattenConsecutive(2), Linear(2 * n_emb, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2 * n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2 * n_hidden, n_hidden), BatchNorm1D(n_hidden), Tanh(),
    Linear(          n_hidden, vocab_size), 
])

with torch.no_grad():
    model.layers[-1].w *= 0.1

parameters = model.parameters()

for p in parameters:
    p.requires_grad = True

epochs = 10000
batch_size = 32
lossi = []
ud = []

for i in range(epochs):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    Xb = Xtr[ix] # batch_size, block_size
    Yb = Ytr[ix] # batch_size, 1

    logits = model(Xb)
    
    loss = F.cross_entropy(logits, Yb)

    for layer in model.layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 # if i < 100000 else 0.01

    with torch.no_grad():
        ud.append([(p.grad * lr).std()/(p.data.std()).log10().item() for p in parameters])

    for p in parameters:
        p.data += -lr * p.grad

    if i % 1000 == 0:
        print(f"{i}/{epochs}: {loss.item()}")

    lossi.append(loss.log10().item())

plt.plot(lossi.view(-1, 1000).mean(dim = 1))
plt.show()

@torch.no_grad
def split_loss(split):
    x, y = {
        "train" : (Xtr, Ytr),
        "dev"   : (Xdev, Ydev),
        "test"  : (Xte, Yte)

    }[split]

    logits = model(x)

    loss = F.cross_entropy(logits, y)

    print(split, loss.item())

for layer in model.layers:
    layer.training = False
split_loss("train")
split_loss("dev")

for i in range(20):
    out = []

    context = [0] * block_size

    while True:
        x = model(torch.tensor(context))

        probs = F.softmax(x, dim = 1)

        ix = torch.multinomial(probs, num_samples = 1).item()

        if ix == 0:
            break

        out.append(ix)
        context = context[1:] + [ix]

    print(''.join(itos[i] for i in out))  