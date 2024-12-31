import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ETF_Fetching import *

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fetch ETF tickers
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
    data.dropna(inplace=True)
    return data

processed_data = {}
valid_tickers = []
for ticker in etf_tickers:
    try:
        data = yf.Ticker(ticker).history(period="1y")  # Use last 5 years of data
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

# Step 3: Apply Exponential Smoothing and Make Predictions
predictions = {}
monthly_averages = {}
percentage_changes = {}

for ticker, data in processed_data.items():
    try:
        # Fit the Exponential Smoothing model
        model = ExponentialSmoothing(data['Close'], trend='add', seasonal=None, initialization_method="estimated")
        fitted_model = model.fit()

        # Predict for the next 20 trading days
        forecast = fitted_model.forecast(steps=20)

        # Calculate the monthly average of the forecast
        monthly_avg = np.mean(forecast)

        # Calculate percentage changes for predictions
        last_close_price = data['Close'].iloc[-1]
        percentage_change = ((forecast.values[-1] - last_close_price) / last_close_price) * 100

        predictions[ticker] = forecast
        monthly_averages[ticker] = monthly_avg
        percentage_changes[ticker] = percentage_change

        print(f"Forecast for {ticker}: {forecast.values}")
        print(f"Expected Monthly Average for {ticker}: {monthly_avg:.2f}")
        print(f"Predicted Percentage Change for {ticker}: {percentage_change:.2f}%")
    except Exception as e:
        print(f"Error applying Exponential Smoothing for {ticker}: {e}")

# Step 4: Identify the Top Gainer
sorted_changes = sorted(percentage_changes.items(), key=lambda x: x[1], reverse=True)
top_gainer_ticker = sorted_changes[0][0]
top_gainer_forecast = predictions[top_gainer_ticker]
print(f"Top gainer: {top_gainer_ticker}")

# Step 5: Visualization
# Plot the real prices and forecast for the top gainer
real_prices = processed_data[top_gainer_ticker]['Close'][-50:]  # Last 50 data points
forecast_dates = pd.date_range(start=real_prices.index[-1], periods=20, freq='B')

plt.figure(figsize=(12, 6))
plt.plot(real_prices.index, real_prices.values, label="Real Prices", color='blue', linewidth=2)
plt.plot(forecast_dates, top_gainer_forecast.values, label="Predicted Prices", color='orange', linestyle='--', linewidth=2)

# Add the monthly average line
monthly_avg = monthly_averages[top_gainer_ticker]
plt.axhline(y=monthly_avg, color='green', linestyle=':', linewidth=2, label=f"Monthly Average: {monthly_avg:.2f}")

# Add title and labels
plt.title(f"Real vs Predicted Prices for {top_gainer_ticker} with Monthly Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print Predictions
print("Predicted Prices for the next month:")
for ticker, forecast in predictions.items():
    print(f"{ticker}: {forecast.values}")

# Print Monthly Averages
print("Expected Monthly Averages:")
for ticker, avg in monthly_averages.items():
    print(f"{ticker}: {avg:.2f}")

# Print Percentage Changes
print("Predicted Percentage Changes:")
for ticker, change in percentage_changes.items():
    print(f"{ticker}: {change:.2f}%")
