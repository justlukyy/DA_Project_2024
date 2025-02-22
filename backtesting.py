import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  Load trained model & scaler
model = joblib.load("models/stock_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

#  Load dataset & filter for VNM
df = pd.read_csv("data/stock_prices_features.csv")
df["Date"] = pd.to_datetime(df["Date"])
df_vnm = df[df["Ticker"] == "VNM"].copy()

#  Define Backtesting Period
backtest_start = "2020-03-01"
backtest_end = "2020-09-01"

df_backtest = df_vnm[(df_vnm["Date"] >= backtest_start) & (df_vnm["Date"] <= backtest_end)].copy()

#  Check if Data Exists
if df_backtest.empty:
    print(" No data found for backtesting! Check the date range.")
    exit()

#  Sort Dates Before Processing
df_backtest = df_backtest.sort_values(by="Date")

#  Prepare features & target variable
X_test = df_backtest[["MA_5", "MA_20", "MA_50", "Normalized_Close"]]
y_actual = df_backtest["Close"].values

#  Scale the input features
X_test_scaled = scaler.transform(X_test)

#  Predict stock prices
y_pred = model.predict(X_test_scaled)

#  Print Evaluation Metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

print(f" Mean Absolute Error (MAE): {mae:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.2f}")

# 🔹 Visualize Predictions
plt.figure(figsize=(12, 6))
plt.plot(df_backtest["Date"], y_actual, label="Actual Price", color="blue", linewidth=2)
plt.plot(df_backtest["Date"], y_pred, label="Predicted Price", color="red", linestyle="dashed", linewidth=2)

plt.title("🔍 Backtesting: Actual vs Predicted Stock Prices")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

plt.show()
