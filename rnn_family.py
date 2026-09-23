# Rossman/rnn_family.py — SimpleRNN vs GRU vs LSTM kıyası (Store 1, univariate)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from metrics import tum_metrikler

np.random.seed(42); tf.random.set_seed(42)

# ================== 1) VERİ HAZIRLAMA (univariate) ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

s1 = train[(train["Store"] == 1) & (train["Open"] == 1)].copy()
s1 = s1.sort_values("Date").reset_index(drop=True)
seri = s1[["Date", "Sales"]].copy()

TEST_DAYS = 42
cutoff = seri["Date"].max() - pd.Timedelta(days=TEST_DAYS)
train_seri = seri[seri["Date"] <= cutoff]
test_seri  = seri[seri["Date"] >  cutoff]

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

# DİKKAT — PENCERE FARKI
# create_sequences, test setinin ilk SEQ_LEN gününü GİRDİ olarak tüketir.
# Bu yüzden dizi modelleri 36 değil (36 - SEQ_LEN) günde ölçülür.
# Prophet/XGBoost ise 36 günün tamamında ölçülüyor. Farklı pencerelerde
# ölçülmüş sMAPE değerleri doğrudan kıyaslanamaz; bu yüzden tahminler
# diske yazılıp model_karsilastirma.py ile ortak pencerede yeniden ölçülür.
print(f"Test seti: {len(test_seri)} gün | Dizi modeli tahmini: {len(X_test)} gün "
      f"(ilk {SEQ_LEN} gün girdi olarak tüketildi)")

# ================== 2) MODEL KURUCU (tek fark: hücre tipi) ==================
def model_kur(hucre_tipi):
    m = Sequential([Input(shape=(SEQ_LEN, 1))])
    if hucre_tipi == "SimpleRNN":
        m.add(SimpleRNN(50, activation="relu"))
    elif hucre_tipi == "GRU":
        m.add(GRU(50, activation="relu"))
    elif hucre_tipi == "LSTM":
        m.add(LSTM(50, activation="relu"))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mse")
    return m

# ================== 3) ÜÇ MODELİ DE EĞİT ==================
sonuclar = []
tahminler = {}

for tip in ["SimpleRNN", "GRU", "LSTM"]:
    print(f"\n{'='*50}\n{tip} eğitiliyor...\n{'='*50}")
    tf.random.set_seed(42)   # her modele aynı başlangıç şansı
    model = model_kur(tip)
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1,
              epochs=100, batch_size=16, callbacks=[early], verbose=0)

    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)
    y_actual = scaler.inverse_transform(y_test)

    sonuc = tum_metrikler(y_actual, pred, tip)
    sonuclar.append(sonuc)
    tahminler[tip] = pred.flatten()

# ================== 4) KIYAS TABLOSU ==================
tablo = pd.DataFrame(sonuclar)
print("\n\n########## RNN AİLESİ KIYAS TABLOSU ##########")
print(tablo.to_string(index=False))
tablo.to_csv("Rossman/rnn_family_results.csv", index=False)

# ================== 5) TAHMİNLERİ KAYDET (ortak pencere kıyası için) ==================
test_dates = test_seri["Date"].iloc[SEQ_LEN:].values
y_actual = scaler.inverse_transform(y_test).flatten()

os.makedirs("Rossman/tahminler", exist_ok=True)
dosya_adi = {"SimpleRNN": "simplernn", "GRU": "gru", "LSTM": "lstm_family"}
for tip, p in tahminler.items():
    pd.DataFrame({"ds": test_dates, "gercek": y_actual, "tahmin": p}).to_csv(
        f"Rossman/tahminler/{dosya_adi[tip]}.csv", index=False)
print(f"\nKiyas icin kaydedildi: {', '.join(dosya_adi.values())}.csv")

# ================== 6) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_actual, marker="o", label="Gerçek", color="black", linewidth=2)
renkler = {"SimpleRNN": "orange", "GRU": "green", "LSTM": "crimson"}
for tip, p in tahminler.items():
    plt.plot(test_dates, p, marker="x", label=tip, color=renkler[tip], alpha=0.8)
plt.title("RNN Ailesi — Tahmin Kıyası (Store 1, Test)")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/17_rnn_family_pred.png")

plt.figure(figsize=(7, 4))
plt.bar(tablo["Model"], tablo["sMAPE"], color=[renkler[m] for m in tablo["Model"]])
for i, v in enumerate(tablo["sMAPE"]):
    plt.text(i, v + 0.1, f"{v}", ha="center")
plt.title("RNN Ailesi — sMAPE Kıyası (düşük = iyi)")
plt.ylabel("sMAPE %"); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/18_rnn_family_smape.png")

print("\nGrafikler: figs/17_rnn_family_pred.png, figs/18_rnn_family_smape.png")
print("Sonuç tablosu: rnn_family_results.csv")