import pandas as pd
import joblib
from model_training import train_model
from predict import predict_stock_price
from backtesting import backtest_model

def main():
    print("Stock Price Prediction Pipeline Started...\n")

    # Step 1: Data Preprocessing
    print("Preprocessing Data...")
    df = pd.read_csv("data/stock_prices_features.csv")
    df.to_csv("data/stock_prices_features.csv", index=False)
    print("Data Preprocessing Done!\n")

    # Step 2: Train the Model
    print(" Training the Model...")
    model, scaler = train_model(df)
    joblib.dump(model, "models/stock_price_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model Training Done!\n")

    # Step 3: Predict Future Prices
    print("Predicting Future Stock Prices...")
    predictions_df = predict_stock_price(df, model, scaler)
    predictions_df.to_csv("data/predicted_stock_prices.csv", index=False)
    print("Predictions Saved!\n")

    # Step 4: Backtest Model
    print("Running Backtesting...")
    backtest_results = backtest_model(df, model, scaler)
    print("Backtesting Done!\n")

    print("All steps completed successfully!")

if __name__ == "__main__":
    main()
