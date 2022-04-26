import torch
import os
import json
import pandas as pd
import numpy as np
import gc
import time
import pymongo
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from torch.utils.data import Dataset, DataLoader

lattice_157_gis_joins = [
    "G0100630",
    "G0100650",
    "G0100670",
    "G0100690",
    "G0100710",
    "G0100730",
    "G0100750",
    "G0100770",
    "G0100790",
    "G0100810",
    "G0100830",
    "G0100850",
    "G0100870",
    "G0100890",
    "G0100910",
    "G0100930",
    "G0100950",
    "G0100970",
    "G0100990",
    "G0101010",
    "G0101030",
    "G0101050",
    "G0101070",
    "G0101090",
    "G0101110",
    "G0101130",
    "G0101150",
    "G0101170",
    "G0101190",
    "G0101210",
    "G0101230",
    "G0101250",
    "G0101310",
    "G0101330",
    "G0201000",
    "G0201300",
    "G0201950",
    "G0201980",
    "G0202200",
    "G0202750",
    "G0400010",
    "G0400050",
    "G0400090",
    "G0400110",
    "G0400120",
    "G0400130",
    "G0400150",
    "G0400170",
    "G0400190",
    "G0400210",
    "G0400250",
    "G0400270",
    "G0500010",
    "G0500030",
    "G0500050",
    "G0500070",
    "G0500090",
    "G0500110",
    "G0500290",
    "G0500310",
    "G0500330",
    "G0500350",
    "G0500590",
    "G0500610",
    "G0500630",
    "G0500650",
    "G0500670",
    "G0500690",
    "G0500710",
    "G0500730",
    "G0500950",
    "G0500970",
    "G0501150",
    "G0501170",
    "G0501290",
    "G0501310",
    "G0501330",
    "G0501350",
    "G0501370",
    "G0501450",
    "G0600090",
    "G0600110",
    "G0600130",
    "G0600210",
    "G0600270",
    "G0600290",
    "G0600650",
    "G0600710",
    "G0600930",
    "G1600490",
    "G1600730",
    "G2300030",
    "G2701370",
    "G3100310",
    "G3200030",
    "G3200070",
    "G3200130",
    "G3200170",
    "G3200230",
    "G3200310",
    "G3200330",
    "G3500030",
    "G3500350",
    "G3500530",
    "G4100250",
    "G4100350",
    "G4100370",
    "G4100450",
    "G4900030",
    "G4900270",
    "G4900370",
    "G4900450",
    "G5000010",
    "G5000030",
    "G5000050",
    "G5000070",
    "G5000090",
    "G5000110",
    "G5000130",
    "G5000150",
    "G5000170",
    "G5000190",
    "G5000210",
    "G5000230",
    "G5000250",
    "G5000270",
    "G5100010",
    "G5100030",
    "G5100050",
    "G5100070",
    "G5100090",
    "G5100110",
    "G5100130",
    "G5100150",
    "G5100170",
    "G5100190",
    "G5100210",
    "G5100230",
    "G5100250",
    "G5100270",
    "G5100290",
    "G5100310",
    "G5100330",
    "G5100350",
    "G5100360",
    "G5100370",
    "G5100410",
    "G5100430",
    "G5100450",
    "G5100470",
    "G5100490",
    "G5100510",
    "G5100530",
    "G5100570",
    "G5100590",
    "G5100610",
    "G5100630",
    "G5100650",
    "G5100670",
    "G5100690",
    "G5100710",
    "G5100730",
    "G5100750",
    "G5100770",
    "G5100790",
    "G5100810",
    "G5100830",
    "G5100850",
    "G5100870",
    "G5100890",
    "G5100910",
    "G5100930",
    "G5100950",
    "G5100970",
    "G5100990",
    "G5101010",
    "G5101030",
    "G5101050",
    "G5101070",
    "G5101090",
    "G5101110",
    "G5101130",
    "G5101150",
    "G5101170",
    "G5101190",
    "G5101210",
    "G5101250",
    "G5101270",
    "G5101310",
    "G5101330",
    "G5101350",
    "G5101370",
    "G5101390",
    "G5101410",
    "G5101430",
    "G5101450",
    "G5101470",
    "G5101490",
    "G5101530",
    "G5101550",
    "G5101570",
    "G5101590",
    "G5101610",
    "G5101630",
    "G5101650",
    "G5101670",
    "G5101690",
    "G5101710",
    "G5101730",
    "G5101750",
    "G5101770",
    "G5101790",
    "G5101810",
    "G5101830",
    "G5101850",
    "G5101870",
    "G5101910",
    "G5101930",
    "G5101950",
    "G5101970",
    "G5101990",
    "G5105500",
    "G5106500",
    "G5106600",
    "G5106700",
    "G5106800",
    "G5107000",
    "G5107100",
    "G5107300",
    "G5107400",
    "G5107600",
    "G5108000",
    "G5108100",
    "G5300010",
    "G5300030",
    "G5300050",
    "G5300070",
    "G5300090",
    "G5300110",
    "G5300130",
    "G5300150",
    "G5300170",
    "G5300190",
    "G5300210",
    "G5300230",
    "G5300250",
    "G5300270",
    "G5300290",
    "G5300310",
    "G5300330",
    "G5300350",
    "G5300370",
    "G5300390",
    "G5300410",
    "G5300430",
    "G5300450",
    "G5300470",
    "G5300490",
    "G5300510",
    "G5300530",
    "G5300550",
    "G5300570",
    "G5300590",
    "G5300610",
    "G5300630",
    "G5300650",
    "G5300670",
    "G5300690",
    "G5300710",
    "G5300730",
    "G5300750",
    "G5300770",
    "G5400010",
    "G5400030",
    "G5400050",
    "G5400070",
    "G5400090",
    "G5400110",
    "G5400130",
    "G5400150",
    "G5400170",
    "G5400190",
    "G5400210",
    "G5400230",
    "G5400250",
    "G5400270",
    "G5400290",
    "G5400310",
    "G5400330",
    "G5400350",
    "G5400370",
    "G5400390",
    "G5400410",
    "G5400430",
    "G5400450",
    "G5400470",
    "G5400490",
    "G5400510",
    "G5400530",
    "G5400550",
    "G5400570",
    "G5400590",
    "G5400610",
    "G5400630",
    "G5400650",
    "G5400670",
    "G5400690",
    "G5400710",
    "G5400730",
    "G5400750",
    "G5400770",
    "G5400790",
    "G5400810",
    "G5400830",
    "G5400850",
    "G5400870",
    "G5400890",
    "G5400910",
    "G5400930",
    "G5400950",
    "G5400970",
    "G5400990",
    "G5401010",
    "G5401030",
    "G5401050",
    "G5401070",
    "G5401090",
    "G5500010",
    "G5500030",
    "G5500050",
    "G5500070",
    "G5500090",
    "G5500110",
    "G5500130",
    "G5500150",
    "G5500170",
    "G5500190",
    "G5500210",
    "G5500230",
    "G5500250",
    "G5500270",
    "G5500290",
    "G5500310",
    "G5500330",
    "G5500350",
    "G5500370",
    "G5500390",
    "G5500410",
    "G5500430",
    "G5500450",
    "G5500470",
    "G5500490",
    "G5500510",
    "G5500530",
    "G5500550",
    "G5500570",
    "G5500590",
    "G5500610",
    "G5500630",
    "G5500650",
    "G5500670",
    "G5500690",
    "G5500710",
    "G5500730",
    "G5500750",
    "G5500770",
    "G5500780",
    "G5500790",
    "G5500810",
    "G5500830",
    "G5500850",
    "G5500870",
    "G5500890",
    "G5500910",
    "G5500930",
    "G5500950",
    "G5500970",
    "G5500990",
    "G5501010",
    "G5501030",
    "G5501050",
    "G5501070",
    "G5501090",
    "G5501110",
    "G5501130",
    "G5501150",
    "G5501170",
    "G5501190",
    "G5501210",
    "G5501230",
    "G5501250",
    "G5501270",
    "G5501290",
    "G5501310",
    "G5501330",
    "G5501350",
    "G5501370",
    "G5501390",
    "G5501410",
    "G5600010",
    "G5600030",
    "G5600050",
    "G5600070",
    "G5600090",
    "G5600110",
    "G5600130",
    "G5600150",
    "G5600170",
    "G5600190",
    "G5600210",
    "G5600230",
    "G5600250",
    "G5600270",
    "G5600290",
    "G5600310",
    "G5600330",
    "G5600350",
    "G5600370",
    "G5600390",
    "G5600410",
    "G5600430",
    "G5600450"
]


class Timer:

    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    # starting the module
    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    # stopping the timer
    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    # resetting the timer
    def reset(self):
        self.elapsed = 0.0

    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def run(gis_join):
    print(f"GISJOIN: {gis_join}")

    profiler: Timer = Timer()
    profiler.start()
    features = [
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
    label = "SOIL_TEMPERATURE_0_TO_01_M_BELOW_SURFACE_KELVIN"
    model_path = "../../../testing/test_models/pytorch/neural_network/model.pt"
    model = torch.jit.load(model_path)
    model.eval()

    profiler.stop()
    print(f">>> Loading model: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()
    client = MongoClient("mongodb://localhost:27017")
    db = client["sustaindb"]
    coll = db["noaa_nam"]
    query = {"GISJOIN": gis_join}
    # Build projection
    projection = {"_id": 0}
    for feature in features:
        projection[feature] = 1
    projection[label] = 1

    documents = coll.find(query, projection)

    features_df = pd.DataFrame(list(documents))
    client.close()

    profiler.stop()
    print(f">>> Loading data into pandas df: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()

    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(features_df)
    features_df = pd.DataFrame(scaled, columns=features_df.columns)
    print(f"Normalized Pandas DataFrame")

    label_df = features_df.pop(label)

    profiler.stop()
    print(f">>> Normalizing pandas df: {profiler.elapsed} sec")
    profiler.reset()

    profiler.start()

    #inputs_numpy = features_df.values.astype(np.float32)
    #y_true_numpy = label_df.values.astype(np.float32)

    inputs_tensor = torch.tensor(features_df.values, dtype=torch.float32, requires_grad=False)
    y_true_tensor = torch.tensor(label_df.values, dtype=torch.float32, requires_grad=False)

    #inputs: torch.Tensor = torch.from_numpy(inputs_numpy)
    #y_true: torch.Tensor = torch.from_numpy(y_true_numpy)
    y_true_tensor = y_true_tensor.view(y_true_tensor.shape[0], 1).squeeze(-1)  # convert y to a column vector

    profiler.stop()
    print(f">>> Getting tensors from pandas df: {profiler.elapsed} sec")
    profiler.reset()

    n_samples, n_features = inputs_tensor.shape
    print(f'n_samples: {n_samples}, n_features: {n_features}')

    with torch.no_grad():
        criterion = torch.nn.MSELoss()
        y_predicted = model(inputs_tensor)
        # y_predicted_numpy = y_predicted.detach().numpy()
        loss = criterion(y_predicted, y_true_tensor)
        # squared_residuals = np.square(y_predicted_numpy - y_true_numpy)
        print(loss)


def main():
    for gis_join in lattice_157_gis_joins:
        run(gis_join)


if __name__ == "__main__":
    main()
