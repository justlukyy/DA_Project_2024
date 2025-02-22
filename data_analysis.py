import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

df = pd.read_csv("cleaned_stock_data.csv")

# Quick check
print(df.describe())  
print(df.info())   

# Visualize plot closing price over time of FPT for example 
df_fpt = df[df["Ticker"] == "FPT"].copy()
df_fpt["Date"] = pd.to_datetime(df_fpt["Date"])
df_fpt.sort_values(by="Date", inplace=True)

plt.figure(figsize=(12, 6))
plt.plot( df_fpt["Date"], df_fpt["Close"], label="Close Price", color="#4DBEEE")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("FPT Stock Closing Price Over Time")
plt.legend()
plt.show()

# Create a Histogram of Daily Returns of VNM from 2020 to 2021 for example
df_vnm = df[df["Ticker"] == "VNM"].copy()  
df_vnm["Date"] = pd.to_datetime(df_vnm["Date"])
df_vnm.sort_values(by="Date", inplace=True)
df_vnm_filtered = df_vnm[(df_vnm["Date"] >= "2019-01-01") & (df_vnm["Date"] <= "2020-12-31")]

plt.figure(figsize=(10, 5))
plt.plot(df_vnm_filtered["Date"], df_vnm_filtered["Daily_Return"], color="#D95319", linewidth=0.8)
plt.axhline(y=0, color="blue", linestyle="--", linewidth=1)
plt.title("Daily Returns of VNM (2019-2020)")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.show()

# Moving avg to identify trends for BIDV stock
df_bid = df[df["Ticker"] == "BID"].copy()
df_bid["Date"] = pd.to_datetime(df_bid["Date"])
df_bid.sort_values(by="Date", inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df_bid["Date"], df_bid["Close"], label="Close Price", color="blue")
plt.plot(df_bid["Date"], df_bid["MA_5"], label="5-day MA", color="orange")
plt.plot(df_bid["Date"], df_bid["MA_20"], label="20-day MA", color="green")
plt.plot(df_bid["Date"], df_bid["MA_50"], label="50-day MA", color="red")
plt.xlabel("Date")
plt.ylabel("BIDV Stock Price")
plt.title("BIDV Stock Price with Moving Averages")
plt.legend()
plt.show()


# Pivot data for correlation between PV Gas and Petrolimex
df_filtered = df[df["Ticker"].isin(["PLX", "GAS"])]
df_pivot = df_filtered.pivot(index="Date", columns="Ticker", values="Close")
corr_matrix = df_pivot.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, annot_kws={"size": 14})
plt.title("Stock Price Correlation Heatmap")
plt.show() 