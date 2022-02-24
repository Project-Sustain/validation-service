import validation
import sklearn


def main():
    print(f"sklearn version: {sklearn.__version__}")

    sklearn_validator = validation.ScikitLearnValidator(
        job_id="test_request_id",
        models_dir="/tmp/validation-service/saved_models",
        model_type="Linear Regression",
        collection="noaa_nam",
        gis_join_key="GISJOIN",
        feature_fields=["PRESSURE_AT_SURFACE_PASCAL", "RELATIVE_HUMIDITY_2_METERS_ABOVE_SURFACE_PERCENT"],
        label_field="TEMPERATURE_AT_SURFACE_KELVIN",
        validation_metric="RMSE",
        normalize=True,
        limit=0,
        sample_rate=0.0
    )

    sklearn_validator.validate_gis_joins_synchronous(
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


if __name__ == "__main__":
    main()
