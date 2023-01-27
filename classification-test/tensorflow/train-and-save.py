import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
data = load_breast_cancer()
X, Y = data.data, data.target

print(f'data.feature_names: {data.feature_names}')
print(f'data.target_names: {data.target_names}')

# Preprocess, normalize
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

print(f'X_test.shape: {X_test.shape}')
print(f'y_test.shape: {y_test.shape}')
print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')

_, input_dim = X_test.shape

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

num_epochs = 10
batch_size = 20
# Train the model
# history = model.fit(train_data, train_labels, epochs=num_epochs,
                    # batch_size=batch_size, validation_data=(val_data, val_labels))

history = model.fit(X_train, y_train, epochs=num_epochs,
                    batch_size=batch_size, validation_data=(X_test, y_test))

# Save the trained model
model.save('model.h5')
