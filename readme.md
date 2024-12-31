# ETF Prediction Project

## Overview
This project aims to predict the potential price gain or loss of Exchange Traded Funds (ETFs) within the next month based on historical data. Using machine learning, specifically a Random Forest Regressor, the model forecasts the percentage change in price. The project dynamically fetches a list of ETFs and processes their data to generate predictions and visualize results.

---

## Mathematical Model

### 1. **Data Preparation**

For each ETF, the following features are calculated:

- **Close**: The daily closing price of the ETF.
- **SMA (Simple Moving Average)**:
  ```math\text{SMA}_{10} = \frac{1}{10} \sum_{i=1}^{10} P_{i}
  ```
  ```math
  \text{SMA}_{30} = \frac{1}{30} \sum_{i=1}^{30} P_{i}
  ```
  Where $P_{i}$ represents the closing price at day $i$.

- **Volatility**: The standard deviation of closing prices over a 10-day window:
  ```math
  \text{Volatility} = \sqrt{\frac{1}{10} \sum_{i=1}^{10} (P_i - \mu)^2}
  ```
  Where $\mu$ is the mean closing price over the 10-day window.

- **Return**: The daily percentage change in price:
  ```math
  \text{Return} = \frac{P_t - P_{t-1}}{P_{t-1}}
  ```


### 2. **Machine Learning Model**
- **Target Variable**:
  The percentage change in price over the next 20 trading days:
  ```math
  \text{Target} = \frac{P_{t+20} - P_t}{P_t}
  ```
- **Feature Set**:
  ```math
   \{\text{Close}, \text{SMA}_{10}, \text{SMA}_{30}, \text{Volatility}\}
   ```
- **Model**:
  - A Random Forest Regressor is trained on the feature set to predict the target variable.
  - Mean Squared Error (MSE) is used to evaluate model performance.

### 3. **Prediction and Visualization**
- Predicted Price:
  ```math
  \text{Predicted Price} = P_t \times (1 + \text{Predicted Change})
  ```
- Visualization compares the actual historical prices and the predicted prices for the top-gaining ETF.

---

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r dependencies.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
4. The script fetches ETFs, processes their data, trains the model, and outputs predictions.

---

## Disclaimer
This project is for educational purposes only. It is not financial advice and was not developed by a professional in finance. Predictions generated by this project should not be used for investment decisions. Use at your own risk.

---

## Bibliography
1. Breiman, L. (2001). "Random Forests." Machine Learning 45(1): 5-32.
2. Yahoo Finance API - Historical Market Data: [https://finance.yahoo.com](https://finance.yahoo.com)
3. Moving Average Calculation: [Investopedia - Moving Averages](https://www.investopedia.com/terms/m/movingaverage.asp)
4. Python Data Science Handbook by Jake VanderPlas.

