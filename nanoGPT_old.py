# program inspired by Andrej Karpathy (https://github.com/karpathy/ng-video-lecture) (https://github.com/karpathy/micrograd)
# data set taken from https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

import numpy as np
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, dtypes
from tinygrad.helpers import getenv, colored, trange
import matplotlib.pyplot as plt
import math
import time
import random

class Transformer:
  def __init__(self, syms, maxlen, embed_dim, ff_dim):

    self.maxlen, self.syms = maxlen, syms
    self.tok_emb_table = Tensor.normal(syms, embed_dim, mean = 0.0, std = 0.02)
    self.pos_emb_table = Tensor.normal(maxlen, embed_dim, mean = 0.0, std = 0.02)
    self.query = Tensor.normal(embed_dim, embed_dim, mean = 0.0, std = 0.02)
    self.key = Tensor.normal(embed_dim, embed_dim, mean = 0.0, std = 0.02)
    self.value = Tensor.normal(embed_dim, embed_dim, mean = 0.0, std = 0.02)
    self.out = Tensor.normal(embed_dim, embed_dim, mean = 0.0, std = 0.02)
    self.ff1 = Tensor.normal(embed_dim, ff_dim, mean = 0.0, std = 0.02)
    self.ff2 = Tensor.normal(ff_dim, embed_dim, mean = 0.0, std = 0.02)
    self.final = Tensor.normal(embed_dim, syms, mean = 0.0, std = 0.02)

  def __call__(self, x):

    onehot_tok = x.one_hot(self.syms)
    onehot_pos = Tensor.eye(self.maxlen).unsqueeze(0).expand([x.shape[0], self.maxlen, self.maxlen])
    tok_emb = onehot_tok.dot(self.tok_emb_table)
    pos_emb = onehot_pos.dot(self.pos_emb_table)
    x = tok_emb + pos_emb
    query, key, value = x.dot(self.query), x.dot(self.key), x.dot(self.value)
    qk = query.matmul(key.transpose(-2,-1)) / math.sqrt(query.shape[-1])
    attn_mask = qk.ones_like().tril()
    attn_mask = attn_mask.where(0, -float("inf"))
    qk = qk + attn_mask
    attention = qk.softmax(-1) @ value
    x = x + attention.dot(self.out)
    x = x + x.dot(self.ff1).relu().dot(self.ff2)
    return x.dot(self.final)
  
###### [ MAIN FUNCTION ] ######

with open('littleshakespeare/input.txt') as file:
    text = file.read()    # data loading
vocab = sorted(list(set(text)))    # finding and sorting all unique charaters in the data
ctoi = {c:i for i,c in enumerate(vocab)}    # map characters to integers
itoc = {i:c for i,c in enumerate(vocab)}    # map integers to characters
encode = lambda clist:         [ctoi[c] for c in clist]     # converts string to list of integers
decode = lambda ilist: ''.join([itoc[i] for i in ilist])    # converts list of integers to string

n = int(0.9*len(text))    # first 90% of data is the training set, rest is test set
train_data, val_data = encode(text[:n]), encode(text[n:])    # creates training set and test set

batch_size = 64    # number of training examples per forward pass
block_size = 128    # max context length for predictions
max_iters = 1000
n_embed = 128
n_ff = 32
learning_rate = 1e-3
sample_num = 200
sample_run = False
vocab_size = len(vocab)
model = Transformer(syms = vocab_size, maxlen = block_size, embed_dim = n_embed, ff_dim = n_ff)
params = nn.state.get_parameters(model)
optim = nn.optim.Adam(params, lr = learning_rate)
print('TRAINING BEGINS (with', sum([t.numel() for t in params]), 'parameters)')

def get_batch(data):
    xi = [random.randint(0, len(data)-block_size-1) for i in range(batch_size)]
    xs = [data[i:i+block_size] for i in xi]
    ys = [data[i+1:i+1+block_size] for i in xi]
    return Tensor(xs), Tensor(ys) # (batch_size, block_size)

@TinyJit
@Tensor.test()
def sample(model, sample_num):
    if sample_run == False: return
    samples = Tensor.ones((1, block_size), dtype=dtypes.int)
    for i in range(sample_num):
        probs = model(samples[:, -block_size:])[-1][-1].softmax(-1)
        samples = samples.cat(probs.multinomial().unsqueeze(0), dim = 1)
    print('MODEL SAMPLING:')
    print(decode(samples.tolist()[0][block_size:]))

@TinyJit
@Tensor.train()
def train_step() -> Tensor:
    Xb, yb = get_batch(train_data)
    optim.zero_grad()
    loss = model(Xb).reshape(-1, vocab_size).sparse_categorical_crossentropy(yb.flatten()).backward()
    optim.step()
    return loss

@TinyJit
@Tensor.test()
def eval() -> Tensor:
    Xb, yb = get_batch(train_data)
    return (model(Xb).argmax(axis=-1) == yb).mean()

sample(model, sample_num)
acc, startTime = float('nan'), time.time()
for k in (t:=trange(max_iters)):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if k % 10 == 9: acc = eval().item()*100
    t.set_description(f"loss: {loss.item():6.2f} accuracy: {acc:5.2f}%")

sample(model, sample_num)
print('TRAINING COMPLETE (in', time.time() - startTime, 'sec)')