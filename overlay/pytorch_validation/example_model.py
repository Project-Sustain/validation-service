import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# convert y to a column vector
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape
print(f'n_samples: {n_samples}')
print(f'n_features: {n_features}')

# Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss and Optimizer
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    # foward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pas
    loss.backward()

    # update
    optimizer.step()

    # empty the gradients for the next iteration
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, loss = {loss.item():.4f}')

# save model
filename = 'pytorch_linear_regression.pth'
torch.save(model, filename)  # using TorchScript