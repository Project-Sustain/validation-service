import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import torch.optim as optim
import numpy as np
from sklearn import datasets
import os


# --- Global Variables ---

# MongoDB Stuff
URI = "mongodb://lattice-100.cs.colostate.edu:27018/"
DATABASE = "sustaindb"
COLLECTION = "noaa_nam"

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

REGRESSION_LABEL_FIELD = "SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN"

# Modeling Stuff
LEARNING_RATE = 0.001
EPOCHS = 3
BATCH_SIZE = 128


def exports():
    # Set CUDA and CUPTI paths
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PATH']= '/usr/local/cuda/bin:$PATH'
    os.environ['CPATH'] = '/usr/local/cuda/include:$CPATH'
    os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LIBRARY_PATH'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LD_LIBRARY_PATH'


def load_from_disk() -> (pd.DataFrame, pd.DataFrame):
    path_to_noaa_csv: str = "~/noaa_nam_normalized.csv"
    all_df: pd.DataFrame = pd.read_csv(path_to_noaa_csv, header=0)
    features: pd.DataFrame = all_df[REGRESSION_FEATURE_FIELDS]
    labels: pd.DataFrame = all_df[REGRESSION_LABEL_FIELD]

    return features, labels


class NoaaNamDataset(Dataset):

    def __init__(self, features_df: pd.DataFrame, label_df: pd.DataFrame):

        features_numpy = features_df.to_numpy(dtype=np.float32)
        label_numpy = label_df.to_numpy(dtype=np.float32)

        self.x_train: torch.Tensor = torch.tensor(features_numpy, dtype=torch.float32)
        self.y_train: torch.Tensor = torch.tensor(label_numpy, dtype=torch.float32)
        self.y_train = self.y_train.unsqueeze(-1)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        return self.x_train[index], self.y_train[index]


def train_linear_regression_model(dataloader: DataLoader):
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    input_size = len(REGRESSION_FEATURE_FIELDS)
    output_size = 1
    model: nn.Linear = nn.Linear(input_size, output_size)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    model = model.to(device)
    criterion.to(device)

    for epoch in range(EPOCHS):

        # Forward pass, optimize, and loss
        for i, (data, labels) in enumerate(dataloader):
            #print(data.shape, labels.shape)
            #print(data, labels)
            data = data.to(device)
            labels = labels.to(device)

            #print(f"Model is on ?, data are on {data.device}, labels are on {labels.device}")

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f'Epoch: {epoch + 1}, loss={loss.item():.4f}')

    # save model
    filename = '../../testing/test_models/pytorch/linear_regression/model.pth'
    torch.save(model, filename)  # using TorchScript



def train_deep_model(dataloader: DataLoader):
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    input_size = len(REGRESSION_FEATURE_FIELDS)
    output_size = 1

    model: DeepModel = DeepModel(D_in=input_size, H=20, D_out=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    model = model.to(device)

    for epoch in range(EPOCHS):

        # Forward pass, optimize, and loss
        for i, (data, labels) in enumerate(dataloader):
            #print(data.shape, labels.shape)
            #print(data, labels)
            data = data.to(device)
            labels = labels.to(device)

            #print(f"Model is on ?, data are on {data.device}, labels are on {labels.device}")

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f'Epoch: {epoch + 1}, loss={loss.item():.4f}')

    # save model
    filename = '../../testing/test_models/pytorch/neural_network/model.pth'
    torch.save(model, filename)  # using TorchScript


class DeepModel(nn.Module):

    def __init__(self, D_in, H=20, D_out=1):
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
    # cuda = torch.cuda.is_available()
    features_df, label_df = load_from_disk()
    noaa_ds = NoaaNamDataset(features_df, label_df)
    sample_subset = torch.utils.data.Subset(noaa_ds, [0])
    sample_loader = DataLoader(sample_subset, batch_size=1, shuffle=False)
    #train_loader = DataLoader(noaa_ds, batch_size=BATCH_SIZE, shuffle=True)
    #train_linear_regression_model(train_loader)
    #train_deep_model(train_loader)

    loaded_model: DeepModel = torch.load('../../testing/test_models/pytorch/linear_regression/model.pth')
    cpu_model = loaded_model.cpu()
    for data, label in sample_loader:
        sample_input_cpu = data.cpu()
        traced_cpu = torch.jit.trace(cpu_model, sample_input_cpu)

    model_scripted = torch.jit.script(loaded_model)
    model_scripted.save('../../testing/test_models/pytorch/linear_regression/model.pt')  # Save


if __name__ == "__main__":
    main()
