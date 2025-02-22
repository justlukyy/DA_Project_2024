import pandas as pd
import numpy as np

# Technical Indicators
def calculate_moving_averages(df, window=5):
    df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, short=12, long=26, signal=9):
    df["EMA_12"] = df["Close"].ewm(span=short, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df

def calculate_bollinger_bands(df, window=20):
    df["Middle_BB"] = df["Close"].rolling(window=window).mean()
    df["Upper_BB"] = df["Middle_BB"] + 2 * df["Close"].rolling(window=window).std()
    df["Lower_BB"] = df["Middle_BB"] - 2 * df["Close"].rolling(window=window).std()
    return df

# Load dataset
df = pd.read_csv("data/cleaned_stock_data.csv")


# Apply Feature Engineering
df = calculate_moving_averages(df, window=5)
df = calculate_rsi(df, window=14)
df = calculate_macd(df)
df = calculate_bollinger_bands(df)

# Save the processed dataset
df.to_csv("data/stock_prices_features.csv", index=False)

print("Feature engineering completed. Processed dataset saved!")
