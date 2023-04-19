import numpy as np
import matplotlib.pyplot as plt
import yfinance as yt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import pandas as pd

crypto_currency = "VRSC-USD"
prediction_days = 60
future_prediction = 60
chart_predictionstart = dt.datetime(2021, 2, 1)
single_predictionstart = dt.datetime(2018, 6, 1)
end = dt.datetime.now()
data = yt.download(crypto_currency, chart_predictionstart, end)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)-future_prediction):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x+future_prediction, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0, 2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0, 2))
model.add(LSTM(units=50))
model.add(Dropout(0, 2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32)

test_data = yt.download(crypto_currency, single_predictionstart, end)
actual_prices = test_data["Close"].values
total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
prediction = model.predict(real_data)
predictions = scaler.inverse_transform(prediction)
filterprediction = str(predictions).replace("[[", "")
finalprediction = str(filterprediction).replace("]]", "")
print(f"The Future price of {crypto_currency} will be {finalprediction} USD")
plt.plot(actual_prices, color="blue", label="Actual Coin Price")
plt.plot(prediction_prices, color="green", label="Predicted Coin Price")
plt.title(f"{crypto_currency} Price Prediction Model")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()