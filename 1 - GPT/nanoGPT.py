# NOTE: based on tinygrad/examples/transformer.py, inspiration and dataset from Andrej Karpathy's Github repos

import numpy as np, random, math, time, os
from tinygrad import Tensor, TinyJit, nn, GlobalCounters, dtypes
from tinygrad.helpers import CI, trange

###### [ 1/3 : MODEL INITIALIZATION ] ######

class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, act=lambda x: x.relu()):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.act = act

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x): # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]

    qk = query.matmul(key.transpose(-2,-1)) / math.sqrt(query.shape[-1])
    attn_mask = qk.ones_like(requires_grad=False).tril()
    attn_mask = attn_mask.where(0, -float("inf"))
    qk = qk + attn_mask
    attention = qk.softmax(-1) @ value
    return attention.transpose(1,2).reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    x = x + self.attn(x.layernorm().linear(*self.ln1))
    return x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2)

class Transformer:
  def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
    self.maxlen, self.syms = maxlen, syms
    self.embed = Tensor.scaled_uniform(maxlen+syms, embed_dim, requires_grad=False)
    self.tbs = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)]
    self.final = Tensor.scaled_uniform(embed_dim, syms)

  def forward(self, x):
    maxlen_eye = Tensor.eye(x.shape[1]).unsqueeze(0).expand(x.shape[0], x.shape[1], x.shape[1])
    onehot_feat = x.int().one_hot(self.syms)
    onehot = maxlen_eye.cat(onehot_feat, dim=2).flatten(end_dim=1)

    x = onehot.dot(self.embed).reshape((x.shape[0], x.shape[1], -1))
    x = x.sequential(self.tbs)
    return x.dot(self.final)
  
###### [ 2/3 : HELPER FUNCTIONS ] ######

# taken from tinygrad/extra/training.py
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

def get_batch(data):
    xi = [random.randint(0, len(data)-block_size-1) for i in range(batch_size)]
    xs = [data[i:i+block_size] for i in xi]
    ys = [data[i+1:i+1+block_size] for i in xi]
    return np.array(xs), np.array(ys) # (batch_size, block_size)

@TinyJit
@Tensor.test()
def sample(model, sample_num):
    samples = [1]*block_size
    for i in range(sample_num): samples.append(model.forward(Tensor([samples[-block_size:]]))[-1][-1].softmax(-1).multinomial().item())
    print('MODEL SAMPLING:')
    print(decode(samples[block_size:]))

###### [ 3/3 : MAIN FUNCTION ] ######

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/shakespeare/input.txt')
with open(data_dir) as file: text = file.read()             # data loading
vocab = sorted(list(set(text)))                             # finding and sorting all unique charaters in the data
ctoi = {c:i for i,c in enumerate(vocab)}                    # map characters to integers
itoc = {i:c for i,c in enumerate(vocab)}                    # map integers to characters
encode = lambda clist:         [ctoi[c] for c in clist]     # converts string to list of integers
decode = lambda ilist: ''.join([itoc[i] for i in ilist])    # converts list of integers to string

n = int(0.9*len(text))                                      # first 90% of data is the training set, rest is test set
train_data, val_data = encode(text[:n]), encode(text[n:])   # creates training set and test set

batch_size = 8000    # number of training examples per forward pass
block_size = 128     # max context length for predictions
steps = 5000

model = Transformer(syms = len(vocab), maxlen = block_size, layers = 4, embed_dim = 64, num_heads = 4, ff_dim = 4*64)
X_train, Y_train = get_batch(train_data)
optim, startTime = nn.optim.Adam(nn.state.get_parameters(model), lr = 0.003), time.time()

print('TRAINING BEGINS (with', sum([t.numel() for t in nn.state.get_parameters(model)]), 'parameters)')
train(model, X_train, Y_train, optim, steps, BS=64, allow_jit=True)
print('TRAINING COMPLETE (in', time.time() - startTime, 'sec)')
sample(model, 200)