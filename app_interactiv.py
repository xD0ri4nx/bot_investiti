import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
from datetime import datetime, date
import tensorflow as tf
import warnings
import os
import io

# ---- Environment Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
tf.config.run_functions_eagerly(True)

st.set_page_config(page_title="LSTM Investment Backtest", layout="wide")
st.title("Historical Validation Bot (Backtesting)")

st.markdown("""
This application trains an LSTM model on a portion of the selected data and validates performance by comparing predictions against real prices that have already occurred.
""")

# ---- Sidebar for Parameters
st.sidebar.header("Configuration Parameters")

# 1. Symbol Selection
tickers = ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "BTC-USD", "NVDA", "META"]
symbol = st.sidebar.selectbox("Select stock symbol:", tickers)

# 2. Period Selection (Constraint: Max Dec 1, 2025)
st.sidebar.subheader("Analysis Interval")
max_allowed_date = date(2025, 12, 1)

start_date = st.sidebar.date_input(
    "Start Date", 
    value=date(2020, 1, 1),
    max_value=max_allowed_date
)

end_date = st.sidebar.date_input(
    "End Date", 
    value=date(2025, 1, 1),
    max_value=max_allowed_date
)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date!")

# 3. Technical Parameters
st.sidebar.subheader("AI Settings")
epochs = st.sidebar.slider("Number of epochs (Training)", min_value=5, max_value=50, value=25)
time_step = st.sidebar.slider("Time window (days)", min_value=10, max_value=90, value=60)
split_percent = 0.80 # 80% training, 20% testing

# ---- Utility Functions
def calculate_accuracy(y_true, y_pred):
    """Calculates accuracy based on MAPE (Mean Absolute Percentage Error)"""
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return accuracy, mape

# ---- Main Logic
if st.button("Run Comparison (Backtest)"):
    if start_date >= end_date:
        st.error("Invalid date range. Please correct the dates in the sidebar.")
        st.stop()

    with st.spinner(f"Downloading historical data for {symbol} ({start_date} -> {end_date})..."):
        try:
            # Download data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Check if 'Close' or 'Adj Close' column exists
            if "Adj Close" in df.columns:
                target_col = "Adj Close"
            elif "Close" in df.columns:
                target_col = "Close"
            else:
                st.error("Downloaded data does not contain price column ('Close').")
                st.stop()
                
            # If df is empty
            if df.empty or len(df) < time_step * 2:
                st.error(f"Not enough data for the selected interval. (Minimum required: {time_step*2} days)")
                st.stop()
                
            prices = df[[target_col]].values

        except Exception as e:
            st.error(f"Critical error retrieving data: {e}")
            st.stop()

    # ---- Preprocessing
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(prices)

    # Split Train (80%) / Test (20%)
    training_size = int(len(scaled_data) * split_percent)
    test_size = len(scaled_data) - training_size
    train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size-time_step:len(scaled_data), :]

    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    # Create sets X, y
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # ---- Build Model
    model = Sequential([
        Input(shape=(time_step, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # ---- Training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Training LSTM model (this may take a few seconds)...")
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    progress_bar.progress(100)
    status_text.text("Training complete!")

    # ---- Predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Inverse transform to get real prices
    train_predict = scaler.inverse_transform(train_predict)
    y_train_real = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_real = scaler.inverse_transform([y_test])

    # ---- Calculate Accuracy (only on Test area)
    accuracy, mape = calculate_accuracy(y_test_real[0], test_predict[:,0])
    
    # ---- Prepare data for Visualization
    # Create clean DataFrame for Test vs Real comparison
    test_indices = df.index[training_size + 1 : len(df) - 1] 
    
    # Fine tune length
    min_len = min(len(test_indices), len(y_test_real[0]), len(test_predict))
    
    comparison_df = pd.DataFrame({
        "Date": test_indices[:min_len],
        "Real Price": y_test_real[0][:min_len],
        "Predicted Price (AI)": test_predict[:,0][:min_len]
    }).set_index("Date")

    comparison_df["Difference"] = comparison_df["Real Price"] - comparison_df["Predicted Price (AI)"]

    # ---- DISPLAY RESULTS ----
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Symbol", symbol)
    with col2:
        st.metric("Estimated Accuracy", f"{accuracy:.2f}%")
    with col3:
        st.metric("Mean Error (MAPE)", f"{mape:.2f}%")
        
    st.subheader(f"Comparative Chart: {symbol}")
    st.caption("Blue Line: Real Price | Red Line: AI Prediction (on data unknown during training)")

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot Full Real Price
    ax.plot(df.index, scaler.inverse_transform(scaled_data), label="Full Real History", color='blue', alpha=0.6)
    
    # Plot Test Prediction
    ax.plot(comparison_df.index, comparison_df["Predicted Price (AI)"], label="AI Prediction (Test)", color='red', linewidth=2)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ---- Detailed Table
    st.subheader("Detailed Data (Last 10 days of interval)")
    st.dataframe(comparison_df.tail(10).style.format("{:.2f}"))

    # Download CSV
    csv = comparison_df.to_csv().encode('utf-8')
    st.download_button(
        "Download Comparison Report (CSV)",
        csv,
        "prediction_report.csv",
        "text/csv",
        key='download-csv'
    )