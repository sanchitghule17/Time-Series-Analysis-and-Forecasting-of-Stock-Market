<!-- README.md – Time-Series Analysis & Forecasting of Stock Market -->
# Time-Series Analysis & Forecasting of Stock Market 📈🔮

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit Cloud](https://img.shields.io/badge/Streamlit-Cloud-Deploy-green?logo=streamlit&logoColor=white)](https://time-series-analysis-and-forecasting-of-stock-market-nwchl7aot.streamlit.app/)
[![License](https://img.shields.io/github/license/sanchitghule17/Time-Series-Analysis-and-Forecasting-of-Stock-Market)](LICENSE)

An end-to-end academic project that shows how to download equity-price data, explore temporal patterns, train both classical and deep-learning models, and **serve interactive forecasts on Streamlit Cloud**.

> ⚠️ Educational use only – **not** financial advice.


---

## Table of Contents
1. [Key Features](#key-features)  
2. [Project Structure](#project-structure)  
3. [Quick Start](#quick-start)  
4. [Usage](#usage)  
5. [Implementation Details](#implementation-details)  
6. [Data & Assumptions](#data--assumptions)  
7. [Extending the Project](#extending-the-project)  
8. [Requirements](#requirements)  
9. [Testing](#testing)  
10. [Results & Screenshots](#results--screenshots)  
11. [Contributing](#contributing)  
12. [License](#license)  
13. [Acknowledgements](#acknowledgements)  

---

## Key Features
* Automated data ingestion via **yfinance**  
* Exploratory analysis: ADF test, ACF/PACF, seasonal decomposition  
* Model zoo: **ARIMA/SARIMA**, **Prophet**, **LSTM / Stacked-LSTM**  
* Unified evaluation: MAE, RMSE, MAPE, Diebold-Mariano test  
* Streamlit UI with zoomable candlestick and forecast plots  
* Modular codebase; easy to plug in new models or data sources  
* Reproducible: `requirements.txt`, CLI flags, cached artefacts  

---

## Project Structure
```
.
├── app/                           # Streamlit application
│   ├── main.py                    # entry-point: streamlit run
│   └── utils.py                   # plotting & caching helpers
├── data/                          # raw & processed data (auto-created)
├── models/
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── lstm_model.py
│   ├── evaluate.py                # common metrics & plots
│   └── outputs/                   # saved models, figures, csvs
├── notebooks/                     # exploratory Jupyter notebooks
├── tests/                         # pytest unit tests
├── assets/                        # screenshots / GIFs for README
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md                      # ← you are here
```

---

## Quick Start

### 1. Clone & create a virtual environment
```
git clone https://github.com/sanchitghule17/Time-Series-Analysis-and-Forecasting-of-Stock-Market.git
cd Time-Series-Analysis-and-Forecasting-of-Stock-Market

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. (Optional) Install Jupyter extras
```
pip install notebook jupyterlab
```

---

## Usage

### Launch the Streamlit app
```
streamlit run app/main.py
```
1. Enter a ticker symbol (e.g., **AAPL**).  
2. Choose a forecast horizon (days).  
3. Inspect interactive plots and download the forecast CSV.

### Run model training & evaluation
```
# Train ARIMA
python models/arima_model.py    --ticker AAPL --p 5 --d 1 --q 0

# Train Prophet
python models/prophet_model.py  --ticker AAPL --periods 60

# Train LSTM
python models/lstm_model.py     --ticker AAPL --epochs 50 --batch_size 32

# Compare all saved forecasts
python models/evaluate.py       --ticker AAPL
```
Outputs (plots, metrics tables, serialized models) are stored under `models/outputs/`.

---

## Implementation Details

| Technique          | File                | Highlights                                                                                         |
|--------------------|---------------------|-----------------------------------------------------------------------------------------------------|
| **ARIMA / SARIMA** | `arima_model.py`    | Differencing ➜ AIC/BIC grid search ➜ dynamic forecast with confidence intervals                     |
| **Prophet**        | `prophet_model.py`  | Holiday & seasonality regressors ➜ rolling-window cross-validation                                  |
| **LSTM**           | `lstm_model.py`     | MinMax scaling ➜ sliding-window tensors ➜ stacked LSTM (64-32) ➜ early stopping & LR scheduler      |
| **Evaluation**     | `evaluate.py`       | MAE, RMSE, MAPE + Diebold-Mariano test for statistical significance                                 |

---

## Data & Assumptions
* Source – Yahoo Finance (`yfinance`).  
* Frequency – Daily market days (weekends/holidays skipped).  
* Missing values – Forward-fill then back-fill.  
* Time-zone – All timestamps converted to UTC.  

---

## Extending the Project
1. **Add a new model**  
   Create `models/my_model.py` implementing  
   `train(args)` and `predict(df, horizon)` ➜ returns a `pd.Series`.  
2. **Back-test multiple tickers**  
   Loop over tickers inside `evaluate.py`, aggregate error stats.  
3. **Dockerise**  
   ```
   FROM python:3.10-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```  
4. **CI/CD** – Add a GitHub Actions workflow that runs `pytest` and `black .` on every push.  

---

## Requirements
```
pandas>=2.0
numpy
matplotlib
seaborn
yfinance
statsmodels
pmdarima
prophet
tensorflow-cpu>=2.15
scikit-learn
streamlit
pytest                 # for tests
```

---

## Testing
```
pip install pytest
pytest -q
```

---

## Results & Screenshots

| LSTM vs. ARIMA – AAPL (RMSE)          | Streamlit Forecast View                          |
|---------------------------------------|--------------------------------------------------|
| ![RMSE table](assets/metrics.png)     | ![Forecast page](assets/forecast_aapl.png)       |

<p align="center">
  <img src="assets/loss_curve.png" alt="LSTM training loss curve" width="550">
</p>

> Place the files below in the **assets/** folder (names must match):  
> • `metrics.png` – comparison table  
> • `forecast_aapl.png` – Streamlit page after selecting AAPL & horizon  
> • `loss_curve.png` – training/validation loss curve exported by LSTM  

---

## Contributing
Pull requests are welcome!  
1. Fork the repo and create a feature branch.  
2. Ensure `pytest` passes and `black .` produces no diff.  
3. Open a PR describing your changes.

---

## License
Distributed under the **GPL-3.0** License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements
* Course material from IIT Bombay  
* Open-source examples by the Python finance community  
* Icons by [Font Awesome](https://fontawesome.com/)  

---

Happy forecasting! ⭐
```
