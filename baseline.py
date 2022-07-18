import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError  # MAPE


from model import model
from load_data import get_data


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Device Check
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"The device is {device}")

# Model
n_input = 1
n_output = 1
n_hidden = 128

lstm = model(n_input, n_output, n_hidden).to(device)
print(lstm)

# loss_func = MeanAbsolutePercentageError()
loss_func = nn.MSELoss()
optimizer = optim.Adam(lstm.parameters(), lr=0.001)
loss_history = []

# Generate Data
timestep = 12
batch_size = 24
trainset, testset = get_data(timestep=timestep)
trainloader = DataLoader(trainset, batch_size=batch_size)
testloader = DataLoader(testset, batch_size=batch_size)


# Training
epochs = 250
for i in range(epochs):
    for b, tup in enumerate(trainloader):
        X, y = tup
        optimizer.zero_grad()

        y_pred = lstm(X.unsqueeze(dim=-1))

        single_loss = loss_func(y_pred.squeeze(), y)
        single_loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# Test
err = 0
err_history = []
data = testloader.__iter__().next()
with torch.no_grad():
    X, y = data
    pred = lstm(X.unsqueeze(dim=-1))
    error = loss_func(pred.squeeze(), y)
    err += error.item()
    err_history.append(error.item())
print(f"mean of test error: {err/len(err_history)}")
plt.plot(range(len(y)), y.cpu().detach().numpy(), color="red")
plt.plot(range(len(y)), pred.cpu().squeeze().detach().numpy(), color="green")
plt.show()
