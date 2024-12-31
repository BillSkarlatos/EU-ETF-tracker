import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Define the top 20 EU ETFs by trading volume (you can expand or automate fetching this list)
etf_tickers = [
    "IWDA.AS", "EQQQ.AS", "SPY5.DE", "XMRV.DE", "XDWD.DE", "IS3N.DE", "EXXT.DE", "XD9U.DE", "XLYE.DE", "SPYD.L", 
    "VEUR.DE", "VGVE.DE", "IWMO.DE", "SXR8.DE", "XDJP.DE", "EUNL.DE", "XRS2.DE", "IUSA.L", "VWRL.L", "UB43.L"
]

# Step 2: Fetch historical data for each ETF and calculate average daily prices
historical_data = {}
for ticker in etf_tickers:
    try:
        etf = yf.Ticker(ticker)
        data = etf.history(period="1y")
        if not data.empty:
            historical_data[ticker] = data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Step 3: Prepare the dataset for ML modeling
def prepare_dataset(data):
    data['Return'] = data['Close'].pct_change()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data.dropna(inplace=True)
    return data

processed_data = {}
for ticker, data in historical_data.items():
    processed_data[ticker] = prepare_dataset(data)

# Step 4: Train ML model to predict next month's return
def train_ml_model(data):
    # Define features and target
    features = data[['Close', 'SMA_10', 'SMA_30', 'Volatility']]
    target = (data['Close'].shift(-20) - data['Close']) / data['Close']
    target = target[:-20]  # Remove NaN values
    features = features[:-20]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE for ticker: {mse}")

    return model

ml_models = {}
for ticker, data in processed_data.items():
    ml_models[ticker] = train_ml_model(data)

# Step 5: Predict next month's performance
def predict_next_month(model, data):
    last_data = data.iloc[-1][['Close', 'SMA_10', 'SMA_30', 'Volatility']]
    prediction = model.predict([last_data])[0]
    return prediction

predictions = {}
for ticker, model in ml_models.items():
    predictions[ticker] = predict_next_month(model, processed_data[ticker])

# Step 6: Visualize the results
sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
top_gainer_ticker = sorted_predictions[0][0]
top_gainer_data = processed_data[top_gainer_ticker]

# Graph: Real vs Predicted Price Graph
plt.figure(figsize=(10, 5))
plt.plot(top_gainer_data['Close'], label="Real Prices", color='blue')

predicted_prices = [top_gainer_data['Close'].iloc[-1] * (1 + predictions[top_gainer_ticker])] * 20
plt.plot(range(len(top_gainer_data), len(top_gainer_data) + 20), predicted_prices, label="Predicted Prices", color='orange')

plt.title(f"Real vs Predicted Price for {top_gainer_ticker}")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# Print Predictions
print("Predicted Gain/Loss for the next month:")
for ticker, prediction in predictions.items():
    print(f"{ticker}: {prediction * 100:.2f}%")
