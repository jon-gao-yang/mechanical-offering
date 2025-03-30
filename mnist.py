import tinygrad
import numpy as np
import matplotlib.pyplot as plt
import time

###### [ 1/4 : MODEL INITIALIZATION ] ######

class LinearNet:
    def __init__(self):
        self.w1 = tinygrad.Tensor.kaiming_uniform(28*28, 160)
        self.b1 = tinygrad.Tensor.kaiming_uniform(1, 160)
        self.w2 = tinygrad.Tensor.kaiming_uniform(160, 80)
        self.b2 = tinygrad.Tensor.kaiming_uniform(1, 80)
        self.w3 = tinygrad.Tensor.kaiming_uniform(80, 40)
        self.b3 = tinygrad.Tensor.kaiming_uniform(1, 40)
        self.w4 = tinygrad.Tensor.kaiming_uniform(40, 20)
        self.b4 = tinygrad.Tensor.kaiming_uniform(1, 20)
        self.w5 = tinygrad.Tensor.kaiming_uniform(20, 10)
        self.b5 = tinygrad.Tensor.kaiming_uniform(1, 10)

    def __call__(self, x:tinygrad.Tensor) -> tinygrad.Tensor:
        l1 = ((x @ self.w1) + self.b1).relu()
        l2 = ((l1 @ self.w2) + self.b2).relu()
        l3 = ((l2 @ self.w3) + self.b3).relu()
        l4 = ((l3 @ self.w4) + self.b4).relu()
        return (l4 @ self.w5) + self.b5
    
###### [ 2/4 : HELPER FUNCTIONS ] ######

# based on code from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(X.shape[0])
        
        # Display the image
        ax.imshow(X[random_index].numpy().reshape(28, 28), cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            yhat = model(X[random_index]).argmax(axis=1).item()
        
        # Display the label above the image
        ax.set_title(f"{y[random_index].item()},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

def write_kaggle_submission(model):
    print('BEGINNING TEST SET INFERENCE')

    X = np.loadtxt('digit-recognizer/test.csv', dtype = int, delimiter = ',', skiprows = 1) # data loading
    X = tinygrad.Tensor((X - np.mean(X))/np.std(X)) # data normalization

    out = np.concatenate((np.arange(1, X.shape[0]+1).reshape((-1, 1)), model(X).argmax(axis=1).reshape((-1, 1))), axis = 1)
    np.savetxt('digit-recognizer/submission.csv', out, delimiter = ',', fmt = '%s', header = 'ImageId,Label', comments = '')

    print('TEST SET INFERENCE COMPLETE')

###### [ 3/4 : MAIN FUNCTION ] ######

@tinygrad.TinyJit
def kaggle_training(model, epochs = 10, batch_size = None, learning_rate = 0.0001):
    [y, X] = np.split(np.loadtxt('digit-recognizer/train.csv', dtype = int, delimiter = ',', skiprows = 1), [1], axis = 1)
    # ^ NOTE: loading data from file, then splitting into labels (first col) and pixel vals
    y = tinygrad.Tensor(np.squeeze(y)) # 2D -> 1D

    X = tinygrad.Tensor((X - np.mean(X))/np.std(X)) # data normalization
    optim = tinygrad.nn.optim.Adam(tinygrad.nn.state.get_parameters(model), lr=learning_rate)
    print('TRAINING BEGINS (with', sum([t.numel() for t in tinygrad.nn.state.get_parameters(model)]), 'parameters)')
    startTime = time.time()

    # optimization
    for k in range(epochs):
        tinygrad.Tensor.training = True
        samples = tinygrad.Tensor.randint(batch_size, high=X.shape[0])
        xb, yb = X[samples], y[samples]

        optim.zero_grad()
        #acc = (model(xb).argmax(axis=1) == yb).mean().item()
        loss = model(xb).sparse_categorical_crossentropy(yb).backward()
        # loss = model(tinygrad.Tensor(xb)).sparse_categorical_crossentropy(tinygrad.Tensor(yb)).backward()
        optim.step()

        if k % 100 == 0 or k == epochs-1:
            tinygrad.Tensor.training = False
            acc = (model(xb).argmax(axis=1) == yb).mean().item()
            print(f"step {k} loss {loss.item()}, accuracy {acc*100}%") # NOTE: COMMENT OUT THIS LINE FOR FASTER TRAINING

    print('TRAINING COMPLETE (in', time.time() - startTime, 'sec)')
    plot_kaggle_data(X, y, model, predict = True)
    #write_kaggle_submission(model) # NOTE: UNCOMMENT THIS LINE TO CREATE KAGGLE SUBMISSION FILE

###### [ 4/4 : MAIN FUNCTION EXECUTION ] ######
kaggle_training(model = LinearNet(), epochs = 600, batch_size = 128, learning_rate = 0.0025799)
