from prophet import Prophet
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_prophet(df):
    """
    Runs a Prophet model with a regressor, including robust data cleaning
    and column name handling.
    """
    # --- 1. Prepare data for Prophet ---
    prophet_df = df[['Close', 'Volume']].copy()
    
    # --- THE FIX: Flatten MultiIndex columns to simple strings ---
    # yfinance can return columns as a MultiIndex. This converts them
    # from ('Close', 'AAPL') to 'Close', which Prophet can handle.
    if isinstance(prophet_df.columns, pd.MultiIndex):
        prophet_df.columns = prophet_df.columns.get_level_values(0)

    # Resample to business day frequency and forward-fill missing values.
    prophet_df = prophet_df.asfreq('B').ffill()
    
    # Robustly clean data before fitting
    prophet_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    prophet_df.dropna(inplace=True)
    
    if prophet_df.empty:
        raise ValueError("Prophet model failed: DataFrame is empty after cleaning. Check initial data quality.")

    prophet_df.reset_index(inplace=True)
    prophet_df.rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'volume'}, inplace=True)

    # --- 2. Split data into training and testing sets ---
    train_size = int(len(prophet_df) * 0.8)
    train = prophet_df.iloc[:train_size]
    test = prophet_df.iloc[train_size:]

    if test.empty:
        raise ValueError("Prophet model failed: The test set is empty. The input data might be too short.")

    # --- 3. Initialize and fit the model ---
    model = Prophet(daily_seasonality=True)
    model.add_regressor('volume')
    model.fit(train)

    # --- 4. Predict directly on the test set ---
    forecast = model.predict(test)

    # --- 5. Align and calculate metrics ---
    actual_values = test['y'].values
    predicted_values = forecast['yhat'].values

    rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)

    # --- 6. Prepare final forecast DataFrame for plotting ---
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_df = forecast_df.set_index('ds')
    forecast_df.index.name = 'Date'

    return forecast_df, model, rmse, mae
