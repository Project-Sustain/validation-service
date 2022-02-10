import tensorflow as tf
import pandas as pd
from logging import info, error
from sklearn.preprocessing import MinMaxScaler


# Normalizes all the columns of a Pandas DataFrame using sklearn's Min-Max Feature Scaling.
def normalize_dataframe(dataframe):
    scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)


def validate_model(models_dir, job_id, model_type, documents, feature_fields, label_field, validation_metric, normalize=True):
    # Load MongoDB Documents into Pandas DataFrame
    features_df = pd.DataFrame(list(documents))
    info(f"Loaded Pandas DataFrame from MongoDB, raw data:\n{features_df}")

    if len(features_df.index) == 0:
        error("DataFrame is empty! Returning None")
        return None

    # Normalize features, if requested
    if normalize:
        features_df = normalize_dataframe(features_df)
        info(f"Pandas DataFrame after normalization:\n{features_df}")

    # Pop the label column off into its own DataFrame
    label_df = features_df.pop(label_field)

    # Load model from disk
    model_path = f"{models_dir}/{job_id}"
    model = tf.keras.models.load_model(model_path)
    model.summary()
    validation_results = model.evaluate(features_df, label_df, batch_size=128, return_dict=True)
    info(f"Model validation results: {validation_results}")
    return validation_results['loss']
