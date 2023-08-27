# import libraries
import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.constraints import maxnorm
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import sys




# receive stock symbol from user
stock_ticker = input("What is the stock symbol?\n")
info_ticker = yf.Ticker(stock_ticker)

end_date = date.today()

# Getting the data for every single day since the start of company
stock_history = info_ticker.history(period='max', end=end_date, interval='1d')

# Storing it into a database for manipulation
df = pd.DataFrame(stock_history)
if len(df) == 0:
    print("Incorrect Stock Ticker")
    sys.exit(1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))



prediction_days = 100
x = []
y = []
for z in range(prediction_days, len(scaled_data) - 1):  # -1 to avoid out of bounds
    x.append(scaled_data[z - prediction_days:z, 0])
    y.append(scaled_data[z, 0])  # Predicting the next day's closing price

x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2, random_state=42)

# Reshape data for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

def create_model(neurons=10, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=neurons, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


model = KerasRegressor(model=create_model, verbose=0)
#print(list(model.get_params().keys()))
validation_split = [0.2, 0.4, 0.6, 0.8]
batch_size = [100, 200, 400, 600]
epochs = [50, 75, 100, 125]


param_grid = {"batch_size":batch_size, "epochs":epochs, "validation_split":validation_split}

grid = GridSearchCV(model, param_distributions=param_grid, n_jobs=1, verbose=2, scoring="neg_mean_squared_error"
                    , cv=3)
grid.fit(x_train, y_train)
best_clf = grid.best_estimator_


y_pred = best_clf.predict(x_test)
mse = np.mean(np.square(y_test - y_pred))
print(f"Mean Squared Error after optimization: {mse}")



# Get the most recent 100 days of stock prices
recent_100_days = scaled_data[-100:]

# Ensure the shape is (1, 100, 1) i.e., 1 sequence, of length 100, with 1 feature
recent_100_days = recent_100_days.reshape(1, 100, 1)

# Predict the stock price for the next day
tmmrw_price = best_clf.predict(recent_100_days)

# If you want to convert this predicted value back to its original scale
tmmrw_price_original_scale = scaler.inverse_transform(np.array(tmmrw_price).reshape(1, -1))

print(f"${tmmrw_price_original_scale[0][0]}")

