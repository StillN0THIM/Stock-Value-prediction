import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from utils import fetch_stock_data, preprocess, create_dataset

# User inputs
ticker = input("Enter Stock Ticker (e.g., TSLA): ") or "TSLA"
start_date = input("Enter Start Date (YYYY-MM-DD): ") or "2022-01-01"
end_date = input("Enter End Date (YYYY-MM-DD): ") or "2023-01-01"

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

print(f"Using ticker: {ticker}")
print(f"Start date: {start_date}")
print(f"End date: {end_date}")

# Fetch stock data
data = fetch_stock_data(ticker, start_date, end_date)

if data is not None and not data.empty:
    print(f"✅ Successfully fetched data! Shape: {data.shape}")
    print(data.head())
else:
    print("❌ Data empty. Cannot continue.")
    exit()

# Preprocess data
scaled_prices, scaler = preprocess(data)
X, y = create_dataset(scaled_prices, look_back=15)

# Train-test split (80-20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Preprocessing successful ✅")

# Train Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions
y_pred = regressor.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📊 Mean Squared Error: {mse}")
print(f"📊 R² Score: {r2}")

# Rescale back to original price
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(15, 6))
plt.plot(y_test_rescaled, label="Actual Price", linewidth=2)
plt.plot(y_pred_rescaled, label="Predicted Price (Regression)", linestyle="--")
plt.title(f"{ticker} Stock Price Prediction using Linear Regression")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
