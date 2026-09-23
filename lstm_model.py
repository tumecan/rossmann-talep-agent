# Rossman/lstm_model.py — Univariate LSTM (Store 1)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # metrics.py'yi bulmak için

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from metrics import tum_metrikler   # <-- metrics.py'deki fonksiyon adı buysa

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# ================== 1) VERİ HAZIRLAMA ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

s1 = train[(train["Store"] == 1) & (train["Open"] == 1)].copy()
s1 = s1.sort_values("Date").reset_index(drop=True)
seri = s1[["Date", "Sales"]].copy()

TEST_DAYS = 42
cutoff = seri["Date"].max() - pd.Timedelta(days=TEST_DAYS)
train_seri = seri[seri["Date"] <= cutoff].copy()
test_seri  = seri[seri["Date"] >  cutoff].copy()

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_seri[["Sales"]])
test_scaled  = scaler.transform(test_seri[["Sales"]])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 14
X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test,  y_test  = create_sequences(test_scaled,  SEQ_LEN)

# ================== 2) MODEL ==================
model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(50, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# ================== 3) EĞİTİM ==================
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    callbacks=[early],
    verbose=1
)

# ================== 4) TAHMİN + ORİJİNAL ÖLÇEĞE DÖNDÜR ==================
pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# ================== 5) METRİKLER ==================
sonuc_uni = tum_metrikler(y_test_actual, pred, "Univariate LSTM")

# ================== 6) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Eğitim / Doğrulama Kaybı (LSTM)")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/7_lstm_loss.png")

test_dates = test_seri["Date"].iloc[SEQ_LEN:].values
plt.figure(figsize=(11, 4))
plt.plot(test_dates, y_test_actual.flatten(), marker="o", label="Gerçek", color="black")
plt.plot(test_dates, pred.flatten(), marker="x", label="LSTM Tahmin", color="crimson")
plt.title("Store 1 — LSTM Tahmini vs Gerçek (Test)")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/8_lstm_pred.png")

print("\nGrafikler kaydedildi: figs/7_lstm_loss.png, figs/8_lstm_pred.png")
plt.show()