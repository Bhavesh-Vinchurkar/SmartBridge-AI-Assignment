# AI Assignment 2 - House Price Prediction using ANN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------------
# 1. Load and Preprocess the Dataset
# -----------------------------

df = pd.read_csv("House Price India.csv")

# Drop columns that won't help in prediction
df.drop(['id', 'Date'], axis=1, inplace=True)

# Convert 'number of bedrooms' and 'number of bathrooms' to int
# (already should be int, but just in case)
df['number of bedrooms'] = df['number of bedrooms'].astype(int)
df['number of bathrooms'] = df['number of bathrooms'].astype(int)

# Drop any missing values (simplest approach)
df.dropna(inplace=True)

# Drop 'Label' and 'Count' if they exist
if 'Label' in df.columns:
    df.drop(['Label'], axis=1, inplace=True)
if 'Count' in df.columns:
    df.drop(['Count'], axis=1, inplace=True)

# Separate features and target
X = df.drop(['Price'], axis=1)
y = df['Price']

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# 2. Build the ANN Model
# -----------------------------

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation because this is regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# -----------------------------
# 3. Test the Model
# -----------------------------

y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot Loss Curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
