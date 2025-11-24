# ================================================
# BOT DE RECOMANDARE INVESTIȚII CU LSTM - Versiune Interactiva
# ================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
from datetime import datetime, timedelta
import tensorflow as tf
import warnings
import os
import io

# ---- Configurare mediu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
tf.config.run_functions_eagerly(True)

st.set_page_config(page_title="Bot Investiții LSTM Interactiv", layout="wide")
st.title("🤖 Bot de Recomandare Investiții cu LSTM (Interactiv)")

st.markdown("Aplicație educațională pentru predicția prețurilor acțiunilor cu ajustare interactivă a parametrilor modelului.")

# ---- Sidebar pentru parametri
st.sidebar.header("Parametri Model")
years_back = st.sidebar.slider("Număr ani analiză istorică", min_value=1, max_value=20, value=10)
epochs = st.sidebar.slider("Număr epoci LSTM", min_value=5, max_value=50, value=20)
time_step = st.sidebar.slider("Lungime fereastră (time step)", min_value=10, max_value=120, value=60)

# Dropdown simboluri
tickers = ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "BTC-USD"]
symbol = st.selectbox("Selectați simbolul acțiunii:", tickers)

if st.button("Rulează Predicția"):

    with st.spinner(f"Se descarcă datele pentru {symbol}..."):
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=365*years_back)
        try:
            data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if data.empty:
                st.error("Simbol invalid sau date indisponibile!")
                st.stop()
        except Exception as e:
            st.error(f"Eroare la descărcarea datelor: {e}")
            st.stop()

    prices = data["Close"].values.reshape(-1,1)

    # ---- Preprocesare
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(prices)

    def create_dataset(dataset, time_step):
        X, y = [], []
        for i in range(len(dataset)-time_step-1):
            X.append(dataset[i:(i+time_step),0])
            y.append(dataset[i+time_step,0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # ---- Model LSTM
    model = Sequential([
        Input(shape=(time_step,1)),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")

    with st.spinner("Se antrenează modelul LSTM..."):
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    # ---- Predicție viitor
    forecast_target = datetime(2025,12,31).date()
    forecast_days = (forecast_target - end_date).days
    if forecast_days <= 0:
        st.warning("Data finală deja trecută. Se vor afișa doar date istorice.")
        forecast_days = 0

    last_window = scaled[-time_step:]
    future_predictions = []
    current_input = last_window.copy()

    for _ in range(forecast_days):
        pred = model.predict(current_input.reshape(1,time_step,1), verbose=0)
        future_predictions.append(pred[0,0])
        current_input = np.append(current_input[1:], pred)
        current_input = current_input.reshape(time_step,1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
    future_dates = pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=forecast_days, freq="D") if forecast_days>0 else []
    future_df = pd.DataFrame(future_prices, index=future_dates, columns=["Predicted_Close"]) if forecast_days>0 else pd.DataFrame()

    # ---- Grafic cu trend colorat
    st.subheader(f"Grafic Istoric + Predicție {symbol}")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(data["Close"], label="Valori istorice", linewidth=2, color="blue")

    if forecast_days>0:
        trend_change = future_prices[-1] - prices[-1][0]
        color = "green" if trend_change > 0 else "red" if trend_change < 0 else "gold"
        ax.plot(future_df["Predicted_Close"], label="Predicții viitoare", linestyle='--', color=color)

    ax.set_xlabel("Dată")
    ax.set_ylabel("Preț (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ---- Recomandare colorată
    last_real = prices[-1][0]
    predicted_mean = np.mean(future_prices[-30:]) if forecast_days>0 else last_real

    if predicted_mean > last_real * 1.05:
        recommendation = "📈 BUY (tendință de creștere estimată)"
        rec_color = "green"
    elif predicted_mean < last_real * 0.95:
        recommendation = "📉 SELL (tendință de scădere estimată)"
        rec_color = "red"
    else:
        recommendation = "⚖️ HOLD (tendință stabilă estimată)"
        rec_color = "gold"

    st.subheader("💡 Recomandare finală")
    st.markdown(f"<h3 style='color:{rec_color}'>{recommendation}</h3>", unsafe_allow_html=True)
    st.markdown(f"**Preț actual:** {last_real:.2f} USD  \n**Preț mediu estimat (ultimele 30 zile):** {predicted_mean:.2f} USD")

    # ---- Descarcă grafic
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="⬇️ Descarcă graficul PNG",
        data=buf,
        file_name=f"{symbol}_prediction.png",
        mime="image/png"
    )
