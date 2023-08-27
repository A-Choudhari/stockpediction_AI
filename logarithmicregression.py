# import libraries

import yfinance as yf
from datetime import date
import pandas as pd
# from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def main():
    # Call functions here
    stock_info = user_input()
    train_data(stock_info)


def user_input():
    # receive stock symbol from user
    try:
        stock_ticker = input("What is the stock symbol?\n")
    except:
        print("Stock Ticker Not Found")

    end_date = date.today()

    # Getting the data for every single day since the start of company
    info_ticker = yf.Ticker(stock_ticker)
    stock_history = info_ticker.history(period='max', end=end_date, interval='1d')

    # Storing it into a database for manipulation
    final_data = pd.DataFrame(stock_history)

    final_data["Tommorow"] = final_data["Close"].shift(-1)

    final_data["Gain"] = (final_data["Tommorow"] > final_data["Close"]).astype(int)

    del final_data["Stock Splits"]

    # Testing Data in final_data
    # print(final_data)
    return final_data


def train_data(stock_data):
    # Getting length of dataframe
    x = stock_data.drop(["Gain", "Tommorow"], axis=1)
    y = stock_data["Gain"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    y_pred = log_reg.predict(x_test)
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()