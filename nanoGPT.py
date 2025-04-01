# program inspired by Andrej Karpathy (https://github.com/karpathy/ng-video-lecture) (https://github.com/karpathy/micrograd)
# data set taken from https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

import numpy as np
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, dtypes
from tinygrad.helpers import getenv, colored, trange
import matplotlib.pyplot as plt
import time
import random

###### [ 1/3 : MODEL INITIALIZATION ] ######

class GPTLanguageModel:

    def __init__(self):
        self.emb_size = 64
        self.head_size = 64
        
        self.tok_emb_table = Tensor(np.random.randn(vocab_size, self.emb_size) * 0.02, dtype=dtypes.float32)
        self.pos_emb_table = Tensor(np.random.randn(block_size, self.emb_size) * 0.02, dtype=dtypes.float32)

        self.w_q = Tensor(np.random.randn(1, self.emb_size, self.head_size) * 0.02, dtype=dtypes.float32)
        self.w_k = Tensor(np.random.randn(1, self.emb_size, self.head_size) * 0.02, dtype=dtypes.float32)
        self.w_v = Tensor(np.random.randn(1, self.emb_size, self.head_size) * 0.02, dtype=dtypes.float32)
        self.w_o = Tensor(np.random.randn(1, self.head_size, self.emb_size) * 0.02, dtype=dtypes.float32)

        self.w1 = Tensor(np.random.randn(1, self.emb_size, self.emb_size*4) * 0.02, dtype=dtypes.float32)
        self.b1 = Tensor(np.zeros((1, 1, self.emb_size*4)), dtype = self.w1.dtype)
        self.w2 = Tensor(np.random.randn(1, self.emb_size*4, self.emb_size) * 0.02, dtype=dtypes.float32)
        self.b2 = Tensor(np.zeros((1, 1, self.emb_size)), dtype = self.w2.dtype)

        self.w_final = Tensor(np.random.randn(1, self.emb_size, vocab_size) * 0.02, dtype=dtypes.float32)

    def __call__(self, x) -> Tensor:

        tok_emb = self.tok_emb_table[x].reshape((-1, block_size, self.emb_size))    # (batch_size, block_size, self.emb_size)
        pos_emb = self.pos_emb_table[Tensor.arange(block_size)]    # (1, block_size, self.emb_size)
        x = tok_emb + pos_emb    # (batch_size, block_size, self.emb_size)
        Q, K, V = x @ self.w_q, x @ self.w_k, x @ self.w_v    # (batch_size, block_size, self.head_size)
        qk = Q @ K.transpose(-2, -1) * self.head_size**-0.5    # (batch_size, block_size, block_size)
        #np.putmask(attention.data, np.tile(np.tri(block_size), (attention.data.shape[0], 1, 1)) == 0, float('-inf'))
        qk_masked = qk + Tensor.full(qk.shape, float("-inf")).triu().realize()
        attention = qk_masked.softmax(-1) @ V    # (batch_size, block_size, self.head_size)
        y = attention @ self.w_o + x    # (batch_size, block_size, self.emb_size)
        z = (y @ self.w1 + self.b1).relu() @ self.w2 + self.b2 + y    # (batch_size, block_size, self.emb_size)
        return z @ self.w_final    # (batch_size, block_size, self.vocab_size)

###### [ 2/3 : HELPER FUNCTIONS ] ######

def get_batch(data):
    xi = [random.randint(0, len(data)-block_size-1) for i in range(batch_size)]
    xs = [data[i:i+block_size] for i in xi]
    ys = [data[i+1:i+1+block_size] for i in xi]
    return Tensor(xs), Tensor(ys) # (batch_size, block_size)

#@TinyJit
@Tensor.test()
def sample(model, sample_num):
    samples = Tensor.ones((1, block_size), dtype=dtypes.int)
    for i in range(sample_num):
        probs = model(samples[:, -block_size:])[-1][-1].softmax(-1)
        samples = samples.cat(probs.multinomial().unsqueeze(0), dim = 1)
    print('MODEL SAMPLING:')
    print(decode(samples.tolist()))

###### [ 3/3 : MAIN FUNCTION ] ######

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
block_size = 256    # max context length for predictions
max_iters = 5000
learning_rate = 9e-3
sample_num = 200
vocab_size = len(vocab)
model = GPTLanguageModel()
params = nn.state.get_parameters(model)
optim = nn.optim.Adam(params)
print('TRAINING BEGINS (with', sum([t.numel() for t in params]), 'parameters)')

@TinyJit
@Tensor.train()
def train_step() -> Tensor:
    Xb, yb = get_batch(train_data)
    optim.zero_grad()
    loss = model(Xb).sparse_categorical_crossentropy(yb).backward()
    optim.step()
    return loss

@TinyJit
@Tensor.test()
def eval() -> Tensor:
    Xb, yb = get_batch(val_data)
    return (model(Xb).argmax(axis=-1) == yb).mean()

#sample(model, sample_num)
acc, startTime = float('nan'), time.time()
for k in (t:=trange(max_iters)):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if k % 10 == 9: acc = eval().item()*100
    t.set_description(f"loss: {loss.item():6.2f} accuracy: {acc:5.2f}%")

#sample(model, sample_num)
print('TRAINING COMPLETE (in', time.time() - startTime, 'sec)')