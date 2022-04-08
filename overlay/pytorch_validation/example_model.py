import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn import datasets


# --- Global Variables ---

# MongoDB Stuff
URI = "mongodb://lattice-100.cs.colostate.edu:27018/"
DATABASE = "sustaindb"
COLLECTION = "noaa_nam"
GIS_JOIN = "G3500170"
REGRESSION_FEATURE_FIELDS = [
    "PRESSURE_REDUCED_TO_MSL_PASCAL",
    "VISIBILITY_AT_SURFACE_METERS",
    "VISIBILITY_AT_CLOUD_TOP_METERS",
    "WIND_GUST_SPEED_AT_SURFACE_METERS_PER_SEC",
    "PRESSURE_AT_SURFACE_PASCAL",
    "TEMPERATURE_AT_SURFACE_KELVIN",
    "DEWPOINT_TEMPERATURE_2_METERS_ABOVE_SURFACE_KELVIN",
    "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT",
    "ALBEDO_PERCENT",
    "TOTAL_CLOUD_COVER_PERCENT"
]

CLASSIFICATION_FEATURE_FIELDS = [
        "PRESSURE_AT_SURFACE_PASCAL",
        "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"
]

REGRESSION_LABEL_FIELD = "SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN"
CLASSIFICATION_LABEL_FIELD = "CATEGORICAL_RAIN_SURFACE_BINARY"

# Modeling Stuff
LEARNING_RATE = 0.01
EPOCHS = 3
BATCH_SIZE = 128

# X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=1)
#
# X = torch.from_numpy(X_numpy.astype(np.float32))
# y = torch.from_numpy(y_numpy.astype(np.float32))
#
# # convert y to a column vector
# y = y.view(y.shape[0], 1)
#
# n_samples, n_features = X.shape
# print(f'n_samples: {n_samples}')
# print(f'n_features: {n_features}')
#
# # Model
# input_size = n_features
# output_size = 1
# model = nn.Linear(input_size, output_size)
#
# # Loss and Optimizer
# criterion = nn.MSELoss()
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # Training Loop
# num_epochs = 100
# for epoch in range(num_epochs):
#     # foward pass and loss
#     y_predicted = model(X)
#     loss = criterion(y_predicted, y)
#
#     # backward pas
#     loss.backward()
#
#     # update
#     optimizer.step()
#
#     # empty the gradients for the next iteration
#     optimizer.zero_grad()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch: {epoch + 1}, loss = {loss.item():.4f}')
#
# # save model
# filename = 'pytorch_linear_regression.pth'
# torch.save(model, filename)  # using TorchScript


class DeepModel(nn.Module):

    def __init__(self, D_in, H=15, D_out=1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        #print(x.size())
        x = self.fc2(x)
        #print(x.size())
        x = self.relu(x)
        #print(x.size())
        return x.squeeze()


def main():
    cuda = torch.cuda.is_available()




    pass


if __name__ == "__main__":
    main()
