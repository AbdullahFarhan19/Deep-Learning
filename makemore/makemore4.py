import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

words = open("makemore/names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))

stoi = {s:(i+1) for i,s in enumerate(chars)}
itos = {(i+1):s for i,s in enumerate(chars)}

block_size = 3

itos[0] = "."
stoi["."] = 0

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

def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()

    print(f"{s:15s} | Exact {str(ex):15s} | approximate {str(app):5s} | maxdiff {maxdiff}")

n_embd = 10 
n_hidden = 64 

C  = torch.randn((vocab_size, n_embd)) # 27, 10
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden) * 0.1 

W2 = torch.randn((n_hidden, vocab_size)) * 0.1
b2 = torch.randn(vocab_size) * 0.1

bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) 
for p in parameters:
  p.requires_grad = True

batch_size = 32
n = batch_size

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

ix = torch.randint(0, Xtr.shape[0], (batch_size,))

Xb = Xtr[ix] # batch_size, block_size
Yb = Ytr[ix] # batch_size, 1

emb = C[Xb] # 32, 3, 10 
embcat = emb.view(batch_size, -1) #32, 30

hprebn = embcat @ W1 + b1 # 32, 64

bnmeani = hprebn.sum(dim = 0, keepdim = True) * (1/n) # dim = 0: what is the mean for every neuron along all training examples
bndiff = hprebn - bnmeani # 32, 64
bndiff2 = bndiff ** 2
bnvar = bndiff2.sum(dim = 0, keepdim = True) * (1/(n-1)) # 1, 64
bnvar_inv = (bnvar + 1e-5) ** -0.5 # 1, 64
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

h = hpreact.tanh()

logits = h @ W2 + b2 # 32, 27

logit_maxes = logits.max(dim = 1, keepdim = True).values # dim = 1: for each training example, what are the possible next characters
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(dim = 1, keepdim = True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log() # log probs
loss = -logprobs[range(n), Yb].mean()

for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, 
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()
loss.backward()

# Exercise 1

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n

dprobs = dlogprobs * (1 / probs)

dcounts_sum_inv = (dprobs * counts).sum(dim = 1, keepdim = True) # broadcasting in the forward pass always leads to sum during back prop

dcounts_sum = dcounts_sum_inv * (-1.0 * (counts_sum ** -2))

dcounts = dcounts_sum * 1 + dprobs * counts_sum_inv

dnorm_logits = dcounts * counts

dlogit_maxes = (dnorm_logits * (-1.0)).sum(dim = 1, keepdim = True)

dlogits = dlogit_maxes * (F.one_hot(logits.max(dim = 1).indices, num_classes = logits.shape[1])) + dnorm_logits * 1

dh = dlogits @ W2.T

dW2 = h.T @ dlogits

db2 = dlogits.sum(dim = 0)

dhpreact = dh * (1 - h**2)

dbnraw = dhpreact * bngain

dbngain = (dhpreact * bnraw).sum(dim = 0, keepdim = True)

dbnbias = (dhpreact * 1.0).sum(dim = 0, keepdim = True)

dbnvar_inv = (dbnraw * bndiff).sum(dim = 0, keepdim = True)

dbnvar = dbnvar_inv * (-1.0/2 * (bnvar + 1e-5) ** (-3.0/2))

dbndiff2 = dbnvar * (1.0/(n-1)) * torch.ones_like(bndiff2)

dbndiff = dbndiff2 * 2 * bndiff + dbnraw * bnvar_inv

dbnmeani = (dbndiff * (-1.0)).sum(dim = 0, keepdim = True)

dhprebn = (dbnmeani * torch.ones_like(hprebn)) * 1.0/n + dbndiff

dembcat = dhprebn @ W1.T

dW1 = embcat.T @ dhprebn

db1 = dhprebn.sum(dim = 0)

demb = dembcat.view(emb.shape)

dC = torch.zeros_like(C)

for i in range(Xb.shape[0]):
   for j in range(Xb.shape[1]):
      ix = Xb[i, j]

      dC[ix] += demb[i, j]

cmp('logprobs', dlogprobs, logprobs)
cmp('probs', dprobs, probs)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
cmp('counts_sum', dcounts_sum, counts_sum)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
cmp('logits', dlogits, logits)
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
cmp('hpreact', dhpreact, hpreact)
cmp('bngain', dbngain, bngain)
cmp('bnbias', dbnbias, bnbias)
cmp('bnraw', dbnraw, bnraw)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bnvar', dbnvar, bnvar)
cmp('bndiff2', dbndiff2, bndiff2)
cmp('bndiff', dbndiff, bndiff)
cmp('bnmeani', dbnmeani, bnmeani)
cmp('hprebn', dhprebn, hprebn)
cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)
cmp('emb', demb, emb)
cmp('C', dC, C)

# Exercise 2:

dlogits = F.softmax(logits, dim = 1)
dlogits[range(n), Yb] -= 1
dlogits /= n # loss = -logits[range(n), Yb].mean() = dL/dli = sum(py)/n = dlogits / n

# Exercise 3:

dhprebn = (bngain * bnvar_inv)/n * (n * dhpreact - dhpreact.sum(dim = 0) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(dim = 0)) 

cmp('hprebn', dhprebn, hprebn)

# Exercise 4: putting it all together!
# Train the MLP neural net with your own backward pass

# init
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

# same optimization as last time
max_steps = 200000
batch_size = 32
n = batch_size # convenience
lossi = []

# use this context manager for efficiency once your backward pass is written (TODO)
with torch.no_grad():

  # kick off optimization
  for i in range(max_steps):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer
    hprebn = embcat @ W1 + b1 # hidden layer pre-activation
    # BatchNorm layer
    # -------------------------------------------------------------
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # -------------------------------------------------------------
    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
      p.grad = None
    #loss.backward() # use this for correctness comparisons, delete it later!

    # manual backprop! #swole_doge_meme
    # -----------------
    
    dlogits = F.softmax(logits, dim = 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n

    # 2nd layer backprop
  
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(dim = 0)

    # tanh

    dhpreact = dh * (1 - h**2)

    # batchnorm backprop

    dhprebn = (bngain * bnvar_inv)/n * (n * dhpreact - dhpreact.sum(dim = 0) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(dim = 0)) 
    dbngain = (dhpreact * bnraw).sum(dim = 0, keepdim = True)
    dbnbias = (dhpreact * 1.0).sum(dim = 0, keepdim = True)

    # 1st layer

    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(dim = 0)

    # embedding

    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for i in range(Xb.shape[0]):
      for j in range(Xb.shape[1]):
        ix = Xb[i, j]

        dC[ix] += demb[i, j]
  
    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
    # -----------------

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p, grad in zip(parameters, grads):
      #p.data += -lr * p.grad # old way of cheems doge (using PyTorch grad from .backward())
      p.data += -lr * grad # new way of swole doge TODO: enable

    # track stats
    if i % 10000 == 0: # print every once in a while
      print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

  #   if i >= 100: # TODO: delete early breaking when you're ready to train the full net
  #     break

with torch.no_grad():
  # pass the training set through
  emb = C[Xtr]
  embcat = emb.view(emb.shape[0], -1)
  hpreact = embcat @ W1 + b1
  # measure the mean/std over the entire training set
  bnmean = hpreact.mean(0, keepdim=True)
  bnvar = hpreact.var(0, keepdim=True, unbiased=True)

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  hpreact = embcat @ W1 + b1
  hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
  h = torch.tanh(hpreact) # (N, n_hidden)
  logits = h @ W2 + b2 # (N, vocab_size)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      # ------------
      # forward pass:
      # Embedding
      emb = C[torch.tensor([context])] # (1,block_size,d)      
      embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
      hpreact = embcat @ W1 + b1
      hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
      h = torch.tanh(hpreact) # (N, n_hidden)
      logits = h @ W2 + b2 # (N, vocab_size)
      # ------------
      # Sample
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))