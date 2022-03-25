# Experiment 2

## Description

- Master job mode: Asynchronous
- Worker job mode: Multiprocessing
- Model used: Tensorflow HDF5 file, type: Sequential regression, inputs: 2 (Pressure at Surface, Relative Humidity at Surface), outputs: 1 (Temperature 2m above Surface),
         learning rate: 0.001, epochs: 3, batch size 32, hidden layers: 2 (first hidden layer 16 units relu, second hidden layer 4 units relu) 
- Number of GISJOINs: all 3192
- Number of workers: 21 (roughly 152 GISJOINs per worker)
- Loss function: MSE
- Validation budget: Total limit, 20M observations (roughly 6k per GISJOIN)
- MongoDB configuration: Local mongod daemon

## Request

```json
{
  "master_job_mode": "ASYNCHRONOUS",
  "worker_job_mode": "MULTIPROCESSING",
  "model_framework": "TENSORFLOW",
  "model_category": "REGRESSION",
  "mongo_host": "localhost",
  "mongo_port": 27017,
  "read_config": {
    "read_preference": "primary",
    "read_concern": "local"
  },
  "database": "sustaindb",
  "collection": "noaa_nam",
  "feature_fields": [
    "PRESSURE_AT_SURFACE_PASCAL",
    "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"
  ],
  "label_field": "TEMPERATURE_AT_SURFACE_KELVIN",
  "normalize_inputs": true,
  "validation_budget": {
    "budget_type": "STATIC_BUDGET",
    "static_budget": {
      "total_limit": 20000000,
      "strata_limit": 0,
      "sample_rate": 0.0
    }
  },
  "loss_function": "MEAN_SQUARED_ERROR",
  "gis_joins": [
    "G5600170",
    "G5600030",
    "G5600050"
  ]
}
```
