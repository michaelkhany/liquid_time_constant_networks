#The code is to train the model on the historical USDT data with Ichimoku Cloud values and predict the closing price of the trading pair.

import numpy as np
import pandas as pd
import requests
#pip install finta
from finta import TA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, RNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from LTC4 import LTCCell
import matplotlib.dates as mdates

# Fetch historical klines data from Binance public API
symbol = "BTCUSDT"
interval = "1h"
url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=720"

response = requests.get(url)
klines = response.json()

# Convert klines data to pandas dataframe
columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
dataframe = pd.DataFrame(klines, columns=columns)

print(dataframe.columns)
print(dataframe.head())

# Convert open_time to datetime and set as index
dataframe["open_time"] = pd.to_datetime(dataframe["open_time"], unit="ms")
dataframe.set_index("open_time", inplace=True)

# Calculate Ichimoku Cloud values using finta library
dataframe["close"] = pd.to_numeric(dataframe["close"])
dataframe["high"] = pd.to_numeric(dataframe["high"])
dataframe["low"] = pd.to_numeric(dataframe["low"])

dataframe = TA.ICHIMOKU(dataframe)

# Drop missing values
dataframe.dropna(inplace=True)

# Prepare the dataset
dataframe.columns = dataframe.columns.str.strip()
# dataset = dataframe['close'].values.reshape(-1, 1).astype('float32')
dataset = dataframe.iloc[:, 4].values.reshape(-1, 1).astype('float32')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Prepare the dataset
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Create and fit the LTC network
num_units = 10
model = Sequential()
model.add(RNN(LTCCell(num_units), input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)

# Make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform([trainY])
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(trainY[0], train_predict[:, 0]))
print(f'Train Score: {train_score:.2f} RMSE')
test_score = np.sqrt(mean_squared_error(testY[0], test_predict[:, 0]))
print(f'Test Score: {test_score:.2f} RMSE')

# Save the trained model
model.save('LTC_IMusdt_model.h5')

print("Model saved successfully.")

# Plot the results
data = scaler.inverse_transform(dataset)

fig, ax = plt.subplots()
ax.plot(dataframe.index, data, label="True Data")
ax.plot(dataframe.index[look_back:look_back + len(train_predict)], train_predict, label="Train Predictions")
ax.plot(dataframe.index[len(train_predict) + 2 * look_back:len(data) - 2], test_predict.reshape(-1), label="Test Predictions")

# Format the x-axis with date and time
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)

plt.xlabel('Date and Time')
plt.ylabel('USDT')
plt.legend()
plt.show()

