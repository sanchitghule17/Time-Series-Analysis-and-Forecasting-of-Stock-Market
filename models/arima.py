import pandas as pd
import math
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_arima(df):
    """
    Runs an auto-ARIMA model to find the best parameters and forecast.
    """
    # Use the 'Close' column for forecasting
    data = df['Close']

    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # Use auto_arima to find the best ARIMA model.
    # - seasonal=False: We are building a non-seasonal ARIMA model.
    # - stepwise=True: Speeds up the search process.
    # - suppress_warnings=True: Hides convergence warnings.
    # - error_action='ignore': Skips models that fail to fit.
    model = auto_arima(
        train,
        start_p=1, start_q=1,
        test='adf',       # use adftest to find optimal 'd'
        max_p=5, max_q=5, # maximum p and q
        m=1,              # frequency of series (1 for non-seasonal)
        d=None,           # let model determine 'd'
        seasonal=False,   # No Seasonality for ARIMA
        start_P=0,
        D=0,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("ARIMA Model Summary:")
    print(model.summary())

    # Generate forecasts for the test period
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)

    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    # Create a forecast DataFrame
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=test.index)
    forecast_df.index.name = 'Date'

    return forecast_df, rmse, mae
