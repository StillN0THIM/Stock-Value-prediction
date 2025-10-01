import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, group_by="ticker")
        if data.empty:
            print(f"No data found for ticker: {ticker} in the given range.")
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Preprocess data
def preprocess(data):
    if data['Close'].isnull().any():
        print("⚠️ Missing values found. Filling with forward fill.")
        data['Close'].fillna(method='ffill', inplace=True)
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    return scaled_prices, scaler

# Create dataset with look-back
def create_dataset(scaled_data, look_back=15):
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i+look_back, 0])
        y.append(scaled_data[i+look_back, 0])
    return np.array(X), np.array(y)
