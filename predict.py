import pandas as pd
import joblib
import numpy as np

# Load trained model & scaler
model = joblib.load("models/stock_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset & filter VNM stock
df = pd.read_csv("data/stock_prices_features.csv")

df_vnm = df[df["Ticker"] == "VNM"].copy()
df_vnm["Date"] = pd.to_datetime(df_vnm["Date"])
df_vnm.sort_values(by="Date", inplace=True)

# Get last known data
last_date = pd.to_datetime(df_vnm["Date"].max())  # Ensure it's datetime
latest_data = df_vnm.iloc[-5:].copy()  # Keep last 5 days for moving averages

# Prepare predictions
future_predictions = []
num_days = 126  # Approx. 6 months (trading days)

for i in range(num_days):
    # Prepare feature input (last row)
    latest_features = latest_data[["MA_5", "MA_20", "MA_50", "Normalized_Close"]].iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_features)

    # Predict next day's stock price
    predicted_price = model.predict(latest_scaled)[0]
    
    # Get next valid trading date
    next_date = last_date + pd.DateOffset(days=1)
    while next_date.weekday() >= 5:  # Skip weekends
        next_date += pd.DateOffset(days=1)

    # Store prediction
    future_predictions.append({"Date": next_date, "Predicted_Close": predicted_price})

    # Add prediction to latest_data for future calculations
    new_row = pd.DataFrame({"Date": [next_date], "Close": [predicted_price]})
    latest_data = pd.concat([latest_data, new_row], ignore_index=True).iloc[-50:]  # Keep only last 50 rows

    # Update moving averages
    latest_data["MA_5"] = latest_data["Close"].rolling(window=5, min_periods=1).mean()
    latest_data["MA_20"] = latest_data["Close"].rolling(window=20, min_periods=1).mean()
    latest_data["MA_50"] = latest_data["Close"].rolling(window=50, min_periods=1).mean()

    # Update last_date
    last_date = next_date

    # Debugging: Print progress
    print(f"Day {i+1}: Predicted Close = {predicted_price:.2f} on {next_date}")

# Convert predictions to DataFrame
future_df = pd.DataFrame(future_predictions)


# Save predictions
future_df.to_csv("predicted_stock_prices_vnm.csv", index=False)

print("Predictions saved to 'predicted_stock_prices_vnm.csv'")
