import yfinance as yf
import mplfinance as mpf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def plot_candlestick_chart(stock_data, ticker):
    mpf.plot(stock_data, type='candle', title=f'{ticker} Japanese Candlestick Chart',
             ylabel='Stock Price (USD)', show_nontrading=True, mav=(10, 20))

def predict_future_prices(stock_data, ticker, days_to_predict):
    # Create a new column 'Prediction' shifted 'days_to_predict' days into the future
    stock_data['Prediction'] = stock_data['Close'].shift(-days_to_predict)

    # Drop rows with missing values
    stock_data.dropna(inplace=True)

    # Features (X) and target variable (y)
    X = np.array(stock_data[['Close']])
    y = np.array(stock_data['Prediction'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the future prices
    stock_data['Predicted'] = model.predict(X)

    # Plot the candlestick chart with predicted prices
    mpf.plot(stock_data, type='candle', title=f'{ticker} Japanese Candlestick Chart with Predicted Prices',
             ylabel='Stock Price (USD)', show_nontrading=True,
             addplot=[mpf.make_addplot(stock_data['Predicted'], color='blue', secondary_y=False)],
             mav=(10, 20), savefig='chart_with_predictions.png')  # You can save the chart if needed

if __name__ == "__main__":
    # Example: Fetch historical data for Apple Inc. (AAPL) from 2020-01-01 to 2022-01-01
    ticker_symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2022-01-01'

    stock_data = get_stock_data(ticker_symbol, start_date, end_date)

    # Plot the initial candlestick chart with zooming and panning enabled
    plot_candlestick_chart(stock_data, ticker_symbol)

    # Predict future prices for the next 5 days (adjust as needed) and plot them on the same chart
    days_to_predict = 5
    predict_future_prices(stock_data, ticker_symbol, days_to_predict)
