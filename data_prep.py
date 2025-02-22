import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("stock_data.csv")

# Format change for better analysis
df["DTYYYYMMDD"] = pd.to_datetime(df["DTYYYYMMDD"], format="%Y%m%d")
# Rename the name of Column
df.rename(columns={"DTYYYYMMDD": "Date"}, inplace=True)
# Sort the data by ticker,date
df.sort_values(by=["Ticker","Date"], ascending=[True, True], inplace=True)

# Check the data
print(df.head(5))
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())

# Create a new column for the analysis
df["Daily_Return"] = df["Close"].pct_change()
# Calculate moving average to know short-terms changes
df["MA_5"] = df["Close"].rolling(window=5).mean()  # 5-day moving average
df["MA_20"] = df["Close"].rolling(window=20).mean()  # 20-day moving average
df["MA_50"] = df["Close"].rolling(window=50).mean()  # 50-day moving average

# Normalize stock prize 
scaler = MinMaxScaler()
df["Normalized_Close"] = scaler.fit_transform(df[["Close"]])

print(df.head(5))
print(df.isnull().sum())

# Load the new data into a new file 
df.to_csv("cleaned_stock_data.csv", index=False)