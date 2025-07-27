import pandas as pd
import math
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_sarima(df):
    """
    Runs an auto-SARIMA model to find the best parameters and forecast.
    """
    # Resample to business day frequency, forward-filling any missing values (holidays)
    data = df['Close'].asfreq('B').ffill()

    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # Use auto_arima to find the best SARIMA model.
    # - seasonal=True: We are building a seasonal model.
    # - m=5: Assumes a 5-day weekly seasonality for business day data.
    #   This is a common and effective assumption for stock prices.
    model = auto_arima(
        train,
        start_p=1, start_q=1,
        test='adf',
        max_p=3, max_q=3,
        m=5,  # Weekly seasonality
        start_P=0,
        seasonal=True,
        d=None,
        D=1,  # Start with one seasonal difference
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("SARIMA Model Summary:")
    print(model.summary())

    # Generate forecasts
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)

    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=test.index)
    forecast_df.index.name = 'Date'

    return forecast_df, rmse, mae
