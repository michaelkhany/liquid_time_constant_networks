#
# Dataset prepration
#
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
dataframe = pd.read_csv(url, usecols=[1], engine="python")
dataset = dataframe.values.astype("float32")

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split the dataset into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Convert time series data into a supervised learning problem
def create_dataset(data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        dataX.append(data[i:(i + look_back), 0])
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 3
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape the input data to be compatible with LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#
# ML using LTC Model
#

import tensorflow as tf
from tensorflow.keras.layers import Input, RNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from LTC4 import LTCCell

# Define hyperparameters
num_units = 32
input_shape = X_train.shape[1:]
output_shape = 1

# Build the model
inputs = Input(shape=input_shape)
x = RNN(LTCCell(num_units))(inputs)
outputs = tf.keras.layers.Dense(output_shape)(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["mae"])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.1, verbose=1)

#
# Let's evaluate the model
#
import matplotlib.pyplot as plt
from LTC4 import LTCCell

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(y_train[0], train_predict[:, 0]))
test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
print(f"Train Score: {train_score:.2f} RMSE")
print(f"Test Score: {test_score:.2f} RMSE")

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset), label="True Data")
plt.plot(range(look_back, len(train_predict) + look_back), train_predict, label="Train Predictions")
plt.plot(range(len(train_predict) + 2 * look_back, len(dataset) - 2), test_predict, label="Test Predictions")  # Subtract 2 instead of 1
plt.xlabel("Months")
plt.ylabel("Airline Passengers")
plt.legend()
plt.show()
