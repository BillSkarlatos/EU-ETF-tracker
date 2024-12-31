import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def get_top_etfs_by_volume(region="Europe", num_etfs=20):
    # Predefined list of popular ETFs for Europe
    return [
        "IWDA.AS", "EQQQ.AS", "SPY5.DE", "XMRV.DE", "XDWD.DE", "IS3N.DE", "EXXT.DE", "XD9U.DE", 
        "XLYE.DE", "SPYD.L", "VEUR.DE", "VGVE.DE", "IWMO.DE", "SXR8.DE", "XDJP.DE", "EUNL.DE", 
        "XRS2.DE", "IUSA.L", "VWRL.L", "UB43.L"
    ]

# Automatically fetch ETF tickers
def fetch_etf_tickers():
    try:
        tickers = get_top_etfs_by_volume()
        return tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []

# Fetch ETFs dynamically
etf_tickers = fetch_etf_tickers()
if not etf_tickers:
    print("No ETFs available to analyze. Exiting.")
    quit()

print(f"Attempting to access the following ETFs: {etf_tickers}")

# Step 2: Fetch and process data
def prepare_dataset(data):
    data['Return'] = data['Close'].pct_change()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data.dropna(inplace=True)
    return data

processed_data = {}
valid_tickers = []
for ticker in etf_tickers:
    try:
        data = yf.Ticker(ticker).history(period="max")
        if not data.empty:
            processed_data[ticker] = prepare_dataset(data)
            valid_tickers.append(ticker)
        else:
            print(f"Ticker {ticker} has no data and is likely delisted.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

if not valid_tickers:
    print("No valid ETFs found. Exiting.")
    quit()

print(f"Valid ETFs: {valid_tickers}")

# Step 3: Train ML model
def train_ml_model(data):
    features = data[['Close', 'SMA_10', 'SMA_30', 'Volatility']]
    target = (data['Close'].shift(-20) - data['Close']) / data['Close']  # 20 trading days ahead
    features, target = features[:-20], target[:-20]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Model MSE: {mse}")
    return model

ml_models = {}
for ticker, data in processed_data.items():
    ml_models[ticker] = train_ml_model(data)

# Step 4: Predict next month's gain/loss
predictions = {}
for ticker, model in ml_models.items():
    last_row = processed_data[ticker][['Close', 'SMA_10', 'SMA_30', 'Volatility']].iloc[-1]
    last_row = last_row.values.reshape(1, -1)  # Reshape for single prediction
    predictions[ticker] = model.predict(last_row)[0]

sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
top_gainer_ticker = sorted_predictions[0][0]
print(f"Top gainer: {top_gainer_ticker}")

# Step 5: Graph real vs predicted prices
top_gainer_data = processed_data[top_gainer_ticker]
last_close_price = top_gainer_data['Close'].iloc[-1]
predicted_prices = [last_close_price * (1 + predictions[top_gainer_ticker]) for _ in range(20)]
real_prices = top_gainer_data['Close'][-50:]  # Last 50 data points

plt.figure(figsize=(12, 6))
plt.plot(real_prices.index, real_prices.values, label="Real Prices", color='blue', linewidth=2)
plt.plot(
    pd.date_range(start=real_prices.index[-1], periods=20, freq='B'),
    predicted_prices, label="Predicted Prices", color='orange', linestyle='--', linewidth=2
)
plt.title(f"Real vs Predicted Prices for {top_gainer_ticker}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print Predictions
print("Predicted Gain/Loss for the next month:")
for ticker, prediction in sorted_predictions:
    print(f"{ticker}: {prediction * 100:.2f}%")
