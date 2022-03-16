import numpy as np
import torch
import pandas as pd
import os
import time

import validation

def main():
    print(f"PyTorch version: {torch.__version__}")
    tf_validator = validation.PyTorchValidator(
        "test_request_id"
        "/tmp/validation-service/saved_models",
        "Linear Regression",
        "noaa_nam",
        "GISJOIN",
        ["PRESSURE_AT_SURFACE_PASCAL", "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"],
        "TEMPERATURE_AT_SURFACE_KELVIN",
        "RMSE",
        True,  # normalize
        0,
        0.0
    )

    tf_validator.validate_gis_joins_synchronous(
        [
            "G2000190",
            "G2000090",
            "G2000670",
            "G2000610",
            "G2000250",
            "G2000070",
            "G2000030",
            "G2000470"
        ]
    )


if __name__ == '__main__':
    main()