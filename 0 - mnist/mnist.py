import numpy as np, matplotlib.pyplot as plt, time, os, torch

###### [ 1/3 : MODEL INITIALIZATION ] ######

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x:torch.Tensor):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

###### [ 2/3 : HELPER FUNCTIONS ] ######

# based on code from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):

        # Select random image
        random_index = torch.randint(low=0, high=X.shape[0], size = ())
        
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

def write_kaggle_submission(model):
    print('BEGINNING TEST SET INFERENCE')

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/digit-recognizer/test.csv')
    X = np.loadtxt(data_dir, dtype = int, delimiter = ',', skiprows = 1) # data loading
    X = torch.Tensor(X).reshape(-1, 1, 28, 28)

    out = np.concatenate((np.arange(1, X.shape[0]+1).reshape((-1, 1)), model(X).argmax(axis=1).reshape((-1, 1)).numpy()), axis = 1)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/digit-recognizer/submission.csv')
    np.savetxt(data_dir, out, delimiter = ',', fmt = '%s', header = 'ImageId,Label', comments = '')

    print('TEST SET INFERENCE COMPLETE')

###### [ 3/3 : MAIN FUNCTION ] ######

model = Model()
steps = 1024
batch_size = 512
optimizer = torch.optim.AdamW(model.parameters())
lossFunc = torch.nn.CrossEntropyLoss()

# loading data from file, then splitting into labels (first col) and pixel vals
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/digit-recognizer/train.csv')
[y, X] = np.split(np.loadtxt(data_dir, dtype = int, delimiter = ',', skiprows = 1), [1], axis = 1)
y, X = torch.Tensor(np.squeeze(y)).long(), torch.Tensor(X).reshape(-1, 1, 28, 28)

print('TRAINING BEGINS (with', sum(p.numel() for p in model.parameters()), 'parameters)')
startTime = time.time()

# optimization
for k in range(steps):

    randint = torch.randint(low = 0, high = X.shape[0], size = (batch_size,))
    
    # forward
    # acc = sum(model(X[randint]).argmax(axis=1) == y[randint]) * 100 / batch_size # skip line for speed
    total_loss = lossFunc(model(X[randint]), y[randint])

    # backward
    optimizer.zero_grad()
    total_loss.backward()
    
    # update parameters
    optimizer.step()

    # if k % 100 == 0: # skip line for speed
    #     print(f"loss: {total_loss.item():6.2f} accuracy: {acc:5.2f}%") # skip line for speed

print('TRAINING COMPLETE (in', time.time() - startTime, 'sec)')
plot_kaggle_data(X, y, model, predict = True) # model demo
write_kaggle_submission(model) # model usage