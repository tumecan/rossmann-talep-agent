# Rossman/lstm_multivariate.py — Multivariate LSTM (Store 1)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # metrics.py'yi bul

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from metrics import tum_metrikler   # merkezi metrik fonksiyonu

import tensorflow as tf
np.random.seed(42); tf.random.set_seed(42)

# ================== 1) VERİ + ÖZELLİK ÜRETİMİ ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

s1 = train[(train["Store"] == 1) & (train["Open"] == 1)].copy()
s1 = s1.sort_values("Date").reset_index(drop=True)

# Özellikler
s1["StateHolidayBin"] = (s1["StateHoliday"].astype(str) != "0").astype(int)
# Haftanın gününü cyclical kodla (Pzt-Paz-Pzt sürekliliği korunur)
s1["dow_sin"] = np.sin(2 * np.pi * s1["DayOfWeek"] / 7)
s1["dow_cos"] = np.cos(2 * np.pi * s1["DayOfWeek"] / 7)

# Model girdisi kolonları — Sales hep ilk sırada (hedef ondan üretiliyor)
feat_cols = ["Sales", "Promo", "SchoolHoliday", "StateHolidayBin", "dow_sin", "dow_cos"]
df = s1[["Date"] + feat_cols].copy()
print("Özellikler:", feat_cols)

# ================== 2) TARİHE GÖRE BÖL ==================
TEST_DAYS = 42
cutoff = df["Date"].max() - pd.Timedelta(days=TEST_DAYS)
train_df = df[df["Date"] <= cutoff].copy()
test_df  = df[df["Date"] >  cutoff].copy()
print(f"Train: {len(train_df)} gün | Test: {len(test_df)} gün")

# ================== 3) ÖLÇEKLE (scaler'lar SADECE train'e fit) ==================
sales_scaler = MinMaxScaler()
feat_scaler  = MinMaxScaler()

train_sales = sales_scaler.fit_transform(train_df[["Sales"]])
test_sales  = sales_scaler.transform(test_df[["Sales"]])

other_cols = feat_cols[1:]  # Sales dışındakiler
train_other = feat_scaler.fit_transform(train_df[other_cols])
test_other  = feat_scaler.transform(test_df[other_cols])

train_mat = np.hstack([train_sales, train_other])
test_mat  = np.hstack([test_sales,  test_other])

# ================== 4) SEQUENCE (bugünün bağlamıyla) ==================
def create_sequences_mv(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])   # (seq_length, n_features)
        y.append(data[i + seq_length, 0])  # sadece Sales
    return np.array(X), np.array(y)

SEQ_LEN = 14
X_train, y_train = create_sequences_mv(train_mat, SEQ_LEN)
X_test,  y_test  = create_sequences_mv(test_mat,  SEQ_LEN)
print("X_train:", X_train.shape, "| X_test:", X_test.shape)

# ================== 5) MODEL ==================
n_features = X_train.shape[2]
model = Sequential([
    Input(shape=(SEQ_LEN, n_features)),
    LSTM(50, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.1,
                    epochs=100, batch_size=16, callbacks=[early], verbose=1)

# ================== 6) TAHMİN + INVERSE ==================
pred_scaled = model.predict(X_test)
pred = sales_scaler.inverse_transform(pred_scaled)
y_test_actual = sales_scaler.inverse_transform(y_test.reshape(-1, 1))

# ================== 7) METRİKLER ==================
sonuc_mv = tum_metrikler(y_test_actual, pred, "Multivariate LSTM")

# ================== 8) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Eğitim / Doğrulama Kaybı (Multivariate LSTM)")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/9_mv_loss.png")

test_dates = test_df["Date"].iloc[SEQ_LEN:].values
plt.figure(figsize=(11, 4))
plt.plot(test_dates, y_test_actual.flatten(), marker="o", label="Gerçek", color="black")
plt.plot(test_dates, pred.flatten(), marker="x", label="MV-LSTM Tahmin", color="teal")
plt.title("Store 1 — Multivariate LSTM Tahmini vs Gerçek (Test)")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/10_mv_pred.png")

print("\nGrafikler: figs/9_mv_loss.png, figs/10_mv_pred.png")
plt.show()