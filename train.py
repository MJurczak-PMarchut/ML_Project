import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import PreparePulses
from pulse_loader import PulseDataset, pulse_transform, cal_vol_transform
from matplotlib import pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

learning_rate = 1e-5
batch_size = 512
no_of_layers = 10
epochs = 50
no_of_features = 500
layers = [nn.Linear(no_of_features, no_of_features), nn.ReLU()]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30, no_of_features),
            nn.ReLU(),
            *(layers*no_of_layers),
            nn.Linear(no_of_features, 1),
        )
        print(self.linear_relu_stack)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

loss_fn = nn.MSELoss()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, last_iter):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    prediction, value = [], []
    with torch.no_grad():
        for X, y in dataloader:
            # print(X[1].shape)
            pred = model(X)
            # print(y)
            test_loss += loss_fn(pred, y.unsqueeze(-1)).item()
            diff = pred.squeeze(1) - y
            for d in diff:
                correct += 1 if abs(d) < 0.002 else 0
            for pr, result in zip(pred.squeeze(1), y):
                prediction.append(pr.item())
                value.append(result.item())
        # if last_iter:
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return prediction, value


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

predd, vall = [], []
lengths = [1000]
for length in lengths:
    print(f"Length: {length}")
    training_data = PulseDataset(
        pulse_dir="C:\\Users\\jurcz\\Desktop\\CalRead",
        transform=pulse_transform,
        target_transform=cal_vol_transform,
        max_len=length
    )

    test_data = PulseDataset(
        pulse_dir="C:\\Users\\jurcz\\Desktop\\CalTest",
        transform=pulse_transform,
        target_transform=cal_vol_transform,
        max_len=length
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    last_iter = False
    start_time = time.time()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        pr, vl = test_loop(test_dataloader, model, loss_fn, last_iter)
        if t == epochs - 1:
            last_iter = True
            predd.append(pr)
            vall.append(vl)
    plt.plot(range(0, len(predd[0])), predd[0])
    plt.plot(range(0, len(vall[0])), vall[0])
    plt.show()
    end_time = time.time()
    print(f'Took {end_time - start_time}')

# print("Done!")
