import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchmetrics import ROC
from sklearn import metrics

data = load_breast_cancer()
X, Y = data.data, data.target

print(data.feature_names)
print(data.target_names)

"""let's preprocess, normalize and create the model"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_test.shape)


class BinaryClassification(torch.nn.Module):
    def __init__(self, input_dimension_):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension_, 1)

    def forward(self, input_dimension_):
        return self.linear(input_dimension_)


_, input_dimension = X_train.shape

model = torch.nn.Linear(input_dimension, 1)

"""train the model"""


def configure_loss_function():
    return torch.nn.BCEWithLogitsLoss()


def configure_optimizer(model_):
    return torch.optim.Adam(model_.parameters())


def full_gd(model, criterion, optimizer, X_train, y_train, n_epochs=2000):
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    for it in range(n_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs_test = model(X_test)
        loss_test = criterion(outputs_test, y_test)

        train_losses[it] = loss.item()
        test_losses[it] = loss_test.item()

        if (it + 1) % 50 == 0:
            print(
                f'In this epoch {it + 1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

    return train_losses, test_losses


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

criterion = configure_loss_function()
optimizer = configure_optimizer(model)
train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train)

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
# plt.show()

"""evaluate model"""

with torch.no_grad():
    p_train = model(X_train)
    p_train = (p_train.numpy() > 0)

    train_acc = np.mean(y_train.numpy() == p_train)

    p_test = model(X_test)
    p_probs = torch.nn.functional.softmax(p_test, dim=1)
    p_test = (p_test.numpy() > 0)

    test_acc = np.mean(y_test.numpy() == p_test)

print(f'train_acc: {train_acc}')
print(f'test_acc: {test_acc}')
print()
print(f'type(p_test): {type(p_test)}')
print(f'type(y_test): {type(y_test)}')

print(f'p_probs: {p_probs}')
# p_test_tensor = torch.from_numpy(p_test)
y_test_numpy = y_test.numpy()

# Calculate ROC
# roc = ROC(pos_label=1)

# fpr, tpr, thresholds = roc(p_test_tensor, y_test)
# print(f'fpr: {fpr}')
# print(f'tpr: {tpr}')
# print(f'thresholds: {thresholds}')

# Save model
model_scripted = torch.jit.script(model)
# model_scripted.save('model.pt')

# Loading
# model = torch.jit.load('model.pt')
# model.eval()
