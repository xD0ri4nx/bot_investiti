# ================================================
# BOT DE TESTARE ISTORICĂ INVESTIȚII (BACKTESTING)
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
from datetime import datetime, date
import tensorflow as tf
import warnings
import os
import io

# ---- Configurare mediu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
tf.config.run_functions_eagerly(True)

st.set_page_config(page_title="Backtest Investiții LSTM", layout="wide")
st.title("📉 Bot de Validare Istorică (Backtesting)")

st.markdown("""
Această aplicație antrenează un model LSTM pe o porțiune din datele selectate și **verifică performanța** comparând predicțiile cu prețurile reale care au avut loc deja.
""")

# ---- Sidebar pentru parametri
st.sidebar.header("Parametri Configurare")

# 1. Selecție Simbol
tickers = ["AAPL", "GOOG", "TSLA", "MSFT", "AMZN", "BTC-USD", "NVDA", "META"]
symbol = st.sidebar.selectbox("Selectați simbolul acțiunii:", tickers)

# 2. Selecție Perioadă (Constrângere: Max 1 Dec 2025)
st.sidebar.subheader("Interval de Analiză")
max_allowed_date = date(2025, 12, 1)

start_date = st.sidebar.date_input(
    "Data de Început", 
    value=date(2020, 1, 1),
    max_value=max_allowed_date
)

end_date = st.sidebar.date_input(
    "Data de Final", 
    value=date(2025, 1, 1),
    max_value=max_allowed_date
)

if start_date >= end_date:
    st.sidebar.error("Data de început trebuie să fie anterioară datei de final!")

# 3. Parametri Tehnici
st.sidebar.subheader("Setări AI")
epochs = st.sidebar.slider("Număr epoci (Antrenament)", min_value=5, max_value=50, value=25)
time_step = st.sidebar.slider("Fereastra de timp (zile)", min_value=10, max_value=90, value=60)
split_percent = 0.80 # 80% antrenament, 20% testare

# ---- Funcții Utilitare
def calculate_accuracy(y_true, y_pred):
    """Calculează acuratețea bazată pe MAPE (Mean Absolute Percentage Error)"""
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return accuracy, mape

# ---- Main Logic
if st.button("Rulează Comparatia (Backtest)"):
    if start_date >= end_date:
        st.error("Interval de date invalid. Vă rugăm corectați datele în sidebar.")
        st.stop()

    with st.spinner(f"Descărcare date istorice pentru {symbol} ({start_date} -> {end_date})..."):
        try:
            # Descărcăm datele
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Verificăm dacă coloana 'Close' sau 'Adj Close' există (fix pentru versiuni noi yfinance)
            if "Adj Close" in df.columns:
                target_col = "Adj Close"
            elif "Close" in df.columns:
                target_col = "Close"
            else:
                st.error("Datele descărcate nu conțin coloana de preț ('Close').")
                st.stop()
                
            # Dacă df este gol
            if df.empty or len(df) < time_step * 2:
                st.error(f"Nu există suficiente date pentru intervalul selectat. (Minim necesar: {time_step*2} zile)")
                st.stop()
                
            prices = df[[target_col]].values

        except Exception as e:
            st.error(f"Eroare critică la preluarea datelor: {e}")
            st.stop()

    # ---- Preprocesare
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(prices)

    # Împărțire Train (80%) / Test (20%)
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

    # Creare seturi X, y
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape pentru LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # ---- Construire Model
    model = Sequential([
        Input(shape=(time_step, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # ---- Antrenare
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Antrenare model LSTM (poate dura câteva secunde)...")
    # Callback simplu nu e afișat în Streamlit ușor, deci antrenăm direct
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
    progress_bar.progress(100)
    status_text.text("Antrenare completă!")

    # ---- Predicții
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Inversare scalare pentru a ajunge la prețurile reale
    train_predict = scaler.inverse_transform(train_predict)
    y_train_real = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_real = scaler.inverse_transform([y_test])

    # ---- Calcul Acuratețe (doar pe zona de Test)
    accuracy, mape = calculate_accuracy(y_test_real[0], test_predict[:,0])
    
    # ---- Pregătire date pentru Vizualizare
    # Ajustăm indicii pentru a se alinia cu datele originale
    train_plot = np.empty_like(scaled_data)
    train_plot[:, :] = np.nan
    # Shift train predictions for plotting
    train_plot[time_step:len(train_predict)+time_step, :] = train_predict

    test_plot = np.empty_like(scaled_data)
    test_plot[:, :] = np.nan
    # Shift test predictions for plotting
    # Start index for test plot is len(train_predict) + (time_step * 2) + 1 roughly, 
    # but strictly based on how we sliced 'test_data' which included lookback
    test_start_idx = len(train_predict) + (time_step * 2) + 1
    # Simplificare aliniere folosind indicii originali din df
    
    # Creăm un DataFrame curat pentru comparatia Test vs Real
    test_indices = df.index[training_size + 1 : len(df) - 1] 
    # Ajustare fină lungime
    min_len = min(len(test_indices), len(y_test_real[0]), len(test_predict))
    
    comparison_df = pd.DataFrame({
        "Data": test_indices[:min_len],
        "Preț Real": y_test_real[0][:min_len],
        "Preț Predis (AI)": test_predict[:,0][:min_len]
    }).set_index("Data")

    comparison_df["Diferența"] = comparison_df["Preț Real"] - comparison_df["Preț Predis (AI)"]

    # ---- AFIȘARE REZULTATE ----
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Simbol", symbol)
    with col2:
        st.metric("Acuratețe Estimată", f"{accuracy:.2f}%")
    with col3:
        st.metric("Eroare Medie (MAPE)", f"{mape:.2f}%")
        
    st.subheader(f"Grafic Comparativ: {symbol}")
    st.caption("Linia Albastră: Preț Real | Linia Roșie: Predicția AI (pe date necunoscute la antrenare)")

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot Preț Real Complet
    ax.plot(df.index, scaler.inverse_transform(scaled_data), label="Istoric Real Complet", color='blue', alpha=0.6)
    
    # Plot Predicție Test (Doar zona portocalie/rosie)
    # Aliniem axa X a predicțiilor cu datele din comparison_df
    ax.plot(comparison_df.index, comparison_df["Preț Predis (AI)"], label="Predicție AI (Test)", color='red', linewidth=2)
    
    ax.set_xlabel("Data")
    ax.set_ylabel("Preț (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ---- Tabel Detaliat
    st.subheader("📊 Date Detaliate (Ultimele 10 zile din interval)")
    st.dataframe(comparison_df.tail(10).style.format("{:.2f}"))

    # Descărcare CSV
    csv = comparison_df.to_csv().encode('utf-8')
    st.download_button(
        "⬇️ Descarcă Raportul Comparativ (CSV)",
        csv,
        "raport_predictie.csv",
        "text/csv",
        key='download-csv'
    )