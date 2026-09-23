# Rossman/prophet_model.py — Prophet ile talep tahmini (Store 1)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # metrics.py'yi bul

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")            # plt.show() script'i kilitlemesin
import matplotlib.pyplot as plt
from prophet import Prophet

from metrics import tum_metrikler

# ================== 1) VERİ HAZIRLAMA ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

s1 = train[(train["Store"] == 1) & (train["Open"] == 1)].copy()
s1 = s1.sort_values("Date").reset_index(drop=True)

# Prophet ZORUNLU kolon isimleri: ds (tarih), y (hedef)
df = s1[["Date", "Sales", "Promo", "SchoolHoliday"]].rename(
    columns={"Date": "ds", "Sales": "y"}
)

# ================== 2) TARİHE GÖRE BÖL ==================
TEST_DAYS = 42
cutoff = df["ds"].max() - pd.Timedelta(days=TEST_DAYS)
train_df = df[df["ds"] <= cutoff].copy()
test_df  = df[df["ds"] >  cutoff].copy()
print(f"Train: {len(train_df)} gün | Test: {len(test_df)} gün")

# ================== 3) TATİL TABLOSU (StateHoliday'den) ==================
hol = s1[s1["StateHoliday"].astype(str) != "0"][["Date", "StateHoliday"]].copy()
holidays = pd.DataFrame({
    "holiday": "state_holiday",
    "ds": pd.to_datetime(hol["Date"]),
    "lower_window": 0,
    "upper_window": 1,   # tatil ertesi günü de etkili kabul et
})

# ================== 4) MODEL ==================
model = Prophet(
    yearly_seasonality=True,     # yıllık desen (Aralık zirvesi)
    weekly_seasonality=True,     # haftalık desen (Pzt/Paz yüksek)
    daily_seasonality=False,     # günlük içi desen yok (günlük veri)
    holidays=holidays,
    interval_width=0.90,         # %90 confidence interval
)
model.add_regressor("Promo")
model.add_regressor("SchoolHoliday")

model.fit(train_df)

# ================== 5) TAHMİN ==================
future = test_df[["ds", "Promo", "SchoolHoliday"]].copy()
forecast = model.predict(future)

pred = forecast["yhat"].values
y_test_actual = test_df["y"].values

# ================== 6) METRİKLER ==================
sonuc_prophet = tum_metrikler(y_test_actual, pred, "Prophet")

# ================== 6.5) AGENT İÇİN ÇIKTIYI KAYDET ==================
cikti = pd.DataFrame({
    "ds":          test_df["ds"].values,
    "yhat":        forecast["yhat"].values,
    "yhat_lower":  forecast["yhat_lower"].values,
    "yhat_upper":  forecast["yhat_upper"].values,
    "promo":       test_df["Promo"].values,
    "gercek":      y_test_actual,
})
cikti.to_csv("Rossman/prophet_forecast.csv", index=False)
print(f"\nAgent icin kaydedildi: Rossman/prophet_forecast.csv ({len(cikti)} gun)")

# ================== 6.6) ORTAK PENCERE KIYASI İÇİN TAHMİNİ KAYDET ==================
# Her model tahminini aynı formatta (ds, gercek, tahmin) yazar. model_karsilastirma.py
# bu dosyaları okuyup HEPSİNİN ortak olduğu günlerde metrikleri yeniden hesaplar.
# Gerekçe: dizi modelleri test setinin ilk SEQ_LEN gününü girdi olarak tükettiği
# için pencereleri farklı; farklı günlerde ölçülmüş sMAPE'ler kıyaslanamaz.
os.makedirs("Rossman/tahminler", exist_ok=True)
pd.DataFrame({"ds": test_df["ds"].values, "gercek": y_test_actual,
              "tahmin": pred}).to_csv("Rossman/tahminler/prophet.csv", index=False)
print("Kiyas icin kaydedildi: Rossman/tahminler/prophet.csv")

# ================== 7) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

plt.figure(figsize=(11, 4))
plt.plot(test_df["ds"], y_test_actual, marker="o", label="Gerçek", color="black")
plt.plot(test_df["ds"], pred, marker="x", label="Prophet Tahmin", color="darkorange")
plt.fill_between(test_df["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                color="orange", alpha=0.2, label="%90 Güven Aralığı")
plt.title("Store 1 — Prophet Tahmini vs Gerçek (Test)")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/11_prophet_pred.png")

fig2 = model.plot_components(forecast)
fig2.savefig("Rossman/figs/12_prophet_components.png")

print("\nGrafikler: figs/11_prophet_pred.png, figs/12_prophet_components.png")