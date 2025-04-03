# NOTE: based on tinygrad/examples/transformer.py

#!/usr/bin/env python3
import numpy as np
import random
import math

from tinygrad.nn.state import get_parameters
from tinygrad.nn.optim import Adam

from tinygrad.tensor import Tensor
from tinygrad.helpers import CI, trange
from tinygrad.engine.jit import TinyJit
  
class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = Tensor.scaled_dot_product_attention(query, key, value, is_causal=True).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x

class Transformer:
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.scaled_uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)]
    self.final = Tensor.scaled_uniform(embed_dim, syms)

  def forward(self, x):
    bs = x.shape[0]

    maxlen_eye = Tensor.eye(x.shape[1])
    maxlen_eye = maxlen_eye.unsqueeze(0).expand([bs, *maxlen_eye.shape])

    onehot_feat = x.int().one_hot(self.syms)

    onehot = maxlen_eye.cat(onehot_feat, dim=2).flatten(end_dim=1)

    x = onehot.dot(self.embed).reshape((bs, x.shape[1], -1))
    x = x.sequential(self.tbs)
    # x = x.reshape((-1, x.shape[-1])).dot(self.final).log_softmax()
    x = x.reshape((-1, x.shape[-1])).dot(self.final)
    return x.reshape((bs, -1, x.shape[-1]))
  
###

def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=lambda out,y: out.sparse_categorical_crossentropy(y),
        transform=lambda x: x, target_transform=lambda x: x, noloss=False, allow_jit=True):

  def train_step(x, y):
    # network
    out = model.forward(x) if hasattr(model, 'forward') else model(x)
    loss = lossfn(out, y)
    optim.zero_grad()
    loss.backward()
    if noloss: del loss
    optim.step()
    if noloss: return (None, None)
    cat = out.argmax(axis=-1)
    accuracy = (cat == y).mean()
    return loss.realize(), accuracy.realize()

  if allow_jit: train_step = TinyJit(train_step)

  with Tensor.train():
    losses, accuracies = [], []
    for i in (t := trange(steps, disable=CI)):
      samp = np.random.randint(0, X_train.shape[0], size=(BS))
      x = Tensor(transform(X_train[samp]), requires_grad=False)
      y = Tensor(target_transform(Y_train[samp]))
      loss, accuracy = train_step(x, y)
      # printing
      if not noloss:
        loss, accuracy = loss.numpy(), accuracy.numpy()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
  return [losses, accuracies]

def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x,
             target_transform=lambda y: y):
  Tensor.training = False
  def numpy_eval(Y_test, num_classes):
    Y_test_preds_out = np.zeros(list(Y_test.shape)+[num_classes])
    for i in trange((len(Y_test)-1)//BS+1, disable=CI):
      x = Tensor(transform(X_test[i*BS:(i+1)*BS]))
      out = model.forward(x) if hasattr(model, 'forward') else model(x)
      Y_test_preds_out[i*BS:(i+1)*BS] = out.numpy()
    Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
    Y_test = target_transform(Y_test)
    return (Y_test == Y_test_preds).mean(), Y_test_preds

  if num_classes is None: num_classes = Y_test.max().astype(int)+1
  acc, Y_test_pred = numpy_eval(Y_test, num_classes)
  print("test set accuracy is %f" % acc)
  return (acc, Y_test_pred) if return_predict else acc

###

with open('littleshakespeare/input.txt') as file:
    text = file.read()    # data loading
vocab = sorted(list(set(text)))    # finding and sorting all unique charaters in the data
ctoi = {c:i for i,c in enumerate(vocab)}    # map characters to integers
itoc = {i:c for i,c in enumerate(vocab)}    # map integers to characters
encode = lambda clist:         [ctoi[c] for c in clist]     # converts string to list of integers
decode = lambda ilist: ''.join([itoc[i] for i in ilist])    # converts list of integers to string

n = int(0.9*len(text))    # first 90% of data is the training set, rest is test set
train_data, val_data = encode(text[:n]), encode(text[n:])    # creates training set and test set

batch_size = 8000    # number of training examples per forward pass
block_size = 128    # max context length for predictions
steps = 5000

def get_batch(data):
    xi = [random.randint(0, len(data)-block_size-1) for i in range(batch_size)]
    xs = [data[i:i+block_size] for i in xi]
    ys = [data[i+1:i+1+block_size] for i in xi]
    return np.array(xs), np.array(ys) # (batch_size, block_size)

model = Transformer(syms = len(vocab), maxlen = block_size, layers = 4, embed_dim = 64, num_heads = 4, ff_dim = 4*64)
X_train, Y_train = get_batch(train_data)
optim = Adam(get_parameters(model), lr=0.003)
train(model, X_train, Y_train, optim, steps, BS=64, allow_jit=True)
acc, Y_test_preds = evaluate(model, X_train, Y_train, num_classes=len(vocab), return_predict=True)

@TinyJit
@Tensor.test()
def sample(model, sample_num):
    samples = [1]*block_size
    for i in range(sample_num): samples.append(model.forward(Tensor([samples[-block_size:]]))[-1][-1].softmax(-1).multinomial().item())
    print('MODEL SAMPLING:')
    print(decode(samples[block_size:]))
sample(model, 200)