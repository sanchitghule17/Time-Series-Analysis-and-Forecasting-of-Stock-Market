import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import sys
import os

# Ensure the models directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the improved model functions
from models.arima import run_arima
from models.sarima import run_sarima
from models.lstm import run_lstm
from models.prophet_model import run_prophet

st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide")

st.title("📈 Stock Market Forecasting Dashboard")
st.markdown("""
This dashboard uses several time-series models to forecast stock prices.
- **ARIMA & SARIMA**: Now use `auto_arima` to find the best parameters automatically.
- **Prophet & LSTM**: Now use additional features like 'Volume' to improve accuracy.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["ARIMA", "SARIMA", "Prophet", "LSTM", "Compare All"]
)

# --- Data Loading ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
    # Forward-fill to handle any missing values, then drop any remaining NaNs
    data.ffill(inplace=True)
    data.dropna(inplace=True)
    return data

try:
    with st.spinner(f"Loading data for {symbol}..."):
        df = load_data(symbol)
    st.success(f"Successfully loaded data for {symbol}.")
except Exception as e:
    st.error(f"Could not load data for symbol '{symbol}'. Please check the ticker. Error: {e}")
    st.stop()


# Show historical prices chart
st.subheader(f"Historical Prices for {symbol}")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_hist.add_trace(go.Scatter(x=df.index, y=df['Volume'], mode='lines', name='Volume', yaxis='y2'))
fig_hist.update_layout(
    title='Historical Close Price and Volume',
    yaxis_title='Price (USD)',
    yaxis2=dict(title='Volume', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig_hist, use_container_width=True)


# --- Model Execution ---
model_results = {}
if st.sidebar.button("Run Forecast"):
    with st.spinner("Running selected model(s)... This may take a few minutes, especially for LSTM."):
        if model_choice == "ARIMA" or model_choice == "Compare All":
            try:
                forecast_df, rmse, mae = run_arima(df)
                model_results["ARIMA"] = (forecast_df, rmse, mae)
            except Exception as e:
                st.error(f"ARIMA model failed: {e}")

        if model_choice == "SARIMA" or model_choice == "Compare All":
            try:
                forecast_df, rmse, mae = run_sarima(df)
                model_results["SARIMA"] = (forecast_df, rmse, mae)
            except Exception as e:
                st.error(f"SARIMA model failed: {e}")

        if model_choice == "Prophet" or model_choice == "Compare All":
            try:
                # Prophet returns the model object as well
                forecast_df, model, rmse, mae = run_prophet(df)
                # For comparison, we only need the forecast values
                prophet_forecast = forecast_df[['yhat']] if 'yhat' in forecast_df else forecast_df
                model_results["Prophet"] = (prophet_forecast, rmse, mae)
            except Exception as e:
                st.error(f"Prophet model failed: {e}")

        if model_choice == "LSTM" or model_choice == "Compare All":
            try:
                forecast_df, rmse, mae = run_lstm(df)
                model_results["LSTM"] = (forecast_df[['Forecast']], rmse, mae)
            except Exception as e:
                st.error(f"LSTM model failed: {e}")

# --- Display Results ---
if model_results:
    st.header("📊 Forecast Results")

    # Determine the test set for plotting
    train_size = int(len(df) * 0.8)
    test_df = df.iloc[train_size:]

    # Plotting
    fig_forecast = go.Figure()
    # Plot actual test data
    fig_forecast.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], mode='lines', name='Actual Test Data', line=dict(color='black')))

    # Plot forecasts from all run models
    for model_name, (forecast_df, _, _) in model_results.items():
        col_name = 'yhat' if model_name == 'Prophet' else 'Forecast'
        fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[col_name], mode='lines', name=f'{model_name} Forecast'))

    fig_forecast.update_layout(
        title=f"{symbol} Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Show metric comparison table
    st.subheader("📝 Model Evaluation")
    eval_data = {
        "Model": [model for model in model_results.keys()],
        "RMSE": [round(metrics[1], 2) for _, metrics in model_results.items()],
        "MAE": [round(metrics[2], 2) for _, metrics in model_results.items()]
    }
    eval_df = pd.DataFrame(eval_data).sort_values(by="RMSE").set_index("Model")
    st.dataframe(eval_df, use_container_width=True)

    # Export all forecasts
    st.subheader("📥 Export Forecasts")
    for model_name, (df_out, _, _) in model_results.items():
        col_name = 'yhat' if model_name == 'Prophet' else 'Forecast'
        export_df = df_out[[col_name]].reset_index()
        export_df.rename(columns={col_name: 'Forecast'}, inplace=True)

        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            f"Download {model_name} Forecast",
            csv,
            file_name=f"{symbol}_{model_name}_forecast.csv",
            mime='text/csv',
            key=f"download_{model_name}"
        )

