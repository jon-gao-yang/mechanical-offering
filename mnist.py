# model based off tinygrad/examples/beautiful_mnist.py
from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
import numpy as np
import matplotlib.pyplot as plt
import time

###### [ 1/3 : MODEL INITIALIZATION ] ######

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)
    
###### [ 2/3 : HELPER FUNCTIONS ] ######

# based on code from Andrew Ng's "Advanced Learning Algorithms" Coursera course
@Tensor.test()
def plot_kaggle_data(X, y, model, predict=False):
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):

        # Select random image
        random_index = Tensor.randint(high=X.shape[0])
        
        # Display random image
        ax.imshow(X[random_index].numpy().reshape(28, 28), cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            yhat = model(X[random_index].reshape(-1, 1, 28, 28)).argmax(axis=1)
        
        # Display the label above the image
        ax.set_title(f"{y[random_index].item()},{yhat.item()}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

@Tensor.test()
def write_kaggle_submission(model):
    print('BEGINNING TEST SET INFERENCE')

    X = np.loadtxt('digit-recognizer/test.csv', dtype = int, delimiter = ',', skiprows = 1) # data loading
    X = Tensor(X).reshape(-1, 1, 28, 28)

    out = np.concatenate((np.arange(1, X.shape[0]+1).reshape((-1, 1)), model(X).argmax(axis=1).reshape((-1, 1)).numpy()), axis = 1)
    np.savetxt('digit-recognizer/submission.csv', out, delimiter = ',', fmt = '%s', header = 'ImageId,Label', comments = '')

    print('TEST SET INFERENCE COMPLETE')

###### [ 3/3 : MAIN FUNCTION ] ######

# hyperparameters
model = Model()
steps = 70
batch_size = 512
optim = nn.optim.Adam(nn.state.get_parameters(model))
print('TRAINING BEGINS (with', sum([t.numel() for t in nn.state.get_parameters(model)]), 'parameters)')

# loading data from file, then splitting into labels (first col) and pixel vals
[y, X] = np.split(np.loadtxt('digit-recognizer/train.csv', dtype = int, delimiter = ',', skiprows = 1), [1], axis = 1)
y, X = Tensor(np.squeeze(y)), Tensor(X).reshape(-1, 1, 28, 28)

@TinyJit
@Tensor.train()
def train_step() -> Tensor:
    samples = Tensor.randint(batch_size, high=X.shape[0])
    optim.zero_grad()
    loss = model(X[samples]).sparse_categorical_crossentropy(y[samples]).backward()
    optim.step()
    return loss

@TinyJit
@Tensor.test()
def eval() -> Tensor: return (model(X[:batch_size]).argmax(axis=1) == y[:batch_size]).mean()

acc, startTime = float('nan'), time.time()
for k in (t:=trange(steps)):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if k % 10 == 9: acc = eval().item()*100
    t.set_description(f"loss: {loss.item():6.2f} accuracy: {acc:5.2f}%")

print('TRAINING COMPLETE (in', time.time() - startTime, 'sec)')
plot_kaggle_data(X, y, model, predict = True)
write_kaggle_submission(model)