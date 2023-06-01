
# This example demonstrates how to use the LTC model to predict the daily closing price of a stock (in this case, Apple Inc.). It loads historical stock data, preprocesses it, creates the LTC model, trains the model, and then makes predictions. Finally, it plots the true closing prices alongside the train and test predictions to visually evaluate the model's performance.

# Note that the performance of the model may vary based on the stock, data, and hyperparameters used. Additionally, keep in mind that predicting stock prices is inherently difficult due to the influence of various factors, and this example should not be considered as financial advice.

#
# Install required libraries:
#

# !pip install pandas_datareader
# !pip install yfinance

#
# Install required libraries:
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RNN, Dense
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from LTC4 import LTCCell


#
# Load and preprocess the data:
#
yf.pdr_override()
stock = 'AAPL'
start_date = '2010-01-01'
end_date = '2021-09-01'
df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)

# Use only the 'Close' column
data = df[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Train-test split
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train, test = data[0:train_size, :], data[train_size:len(data), :]


#
# Define helper functions to create datasets:
#
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


#
# Define and train the LTC model:
#
num_units = 50
batch_size = 32
epochs = 100

model = Sequential()
model.add(RNN(LTCCell(num_units), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, Y_test))


#
# Make predictions:
#
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Calculate RMSE
train_score = np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0]))
test_score = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))
print('Train Score: {:.2f} RMSE'.format(train_score))
print('Test Score: {:.2f} RMSE'.format(test_score))


#
# Plot the results:
#
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(data), label="True Data")
plt.plot(range(look_back + 1, look_back + 1 + len(train_predict)), train_predict, label="Train Predictions")
plt.plot(range(len(train_predict) + 2 * look_back + 1, len(data) - 1), test_predict, label="Test Predictions")
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Stock Price Predictions using LTC Model')
plt.legend()
plt.show()


