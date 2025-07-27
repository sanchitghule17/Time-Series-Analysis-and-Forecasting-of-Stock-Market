import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def run_lstm(df, feature_columns=['Close', 'Volume'], time_step=60):
   
    # --- 1. Data Preparation ---
    data_to_use = df[feature_columns].copy().ffill()
    target_column = 'Close'
    target_col_idx = feature_columns.index(target_column)

    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_use)

    # --- 2. Create Sequential Data ---
    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(time_step, len(dataset)):
            X.append(dataset[i - time_step:i, :])
            y.append(dataset[i, target_col_idx])
        return np.array(X), np.array(y)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - time_step:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    n_features = len(feature_columns)

    # --- 3. Build the LSTM Model ---
    model = Sequential([
        Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(time_step, n_features))),
        Dropout(0.3),
        LSTM(units=100, return_sequences=False),
        Dropout(0.3),
        Dense(units=50),
        Dense(units=1)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.summary()

    # --- 4. Train the Model ---
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # --- 5. Evaluate the Model ---
    predictions_scaled = model.predict(X_test)
    
    # Create a dummy array to inverse transform predictions 
    dummy_array = np.zeros((len(predictions_scaled), n_features))
    dummy_array[:, target_col_idx] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, target_col_idx]

    
    dummy_array = np.zeros((len(y_test), n_features))
    dummy_array[:, target_col_idx] = y_test.flatten()
    actual = scaler.inverse_transform(dummy_array)[:, target_col_idx]

    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)

    # Create forecast DataFrame
    predict_dates = df.index[-len(actual):]
    forecast_df = pd.DataFrame({'Forecast': predictions, 'Actual': actual})
    forecast_df.set_index(predict_dates, inplace=True)
    forecast_df.index.name = 'Date'

    return forecast_df, rmse, mae
