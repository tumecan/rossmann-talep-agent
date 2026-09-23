# Rossman/prophet_tuning.py — Prophet hyperparameter tuning (grid search, Store 1)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from prophet import Prophet

from metrics import tum_metrikler

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# ================== 1) VERİ ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

s1 = train[(train["Store"] == 1) & (train["Open"] == 1)].copy()
s1 = s1.sort_values("Date").reset_index(drop=True)
df = s1[["Date", "Sales", "Promo", "SchoolHoliday"]].rename(
    columns={"Date": "ds", "Sales": "y"})

# ================== 2) TARİHE GÖRE 3'E BÖL: train / val / test ==================
TEST_DAYS = 42
VAL_DAYS  = 42
test_cut = df["ds"].max() - pd.Timedelta(days=TEST_DAYS)
val_cut  = test_cut - pd.Timedelta(days=VAL_DAYS)

train_df = df[df["ds"] <= val_cut]
val_df   = df[(df["ds"] > val_cut) & (df["ds"] <= test_cut)]
test_df  = df[df["ds"] > test_cut]
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ================== 3) GRID SEARCH (validation'a göre en iyiyi seç) ==================
param_grid = {
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
    "seasonality_prior_scale": [1.0, 5.0, 10.0],
}
kombinasyonlar = [dict(zip(param_grid.keys(), v))
                  for v in itertools.product(*param_grid.values())]
print(f"\n{len(kombinasyonlar)} kombinasyon denenecek...")

def prophet_kur(params):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, interval_width=0.90, **params)
    m.add_regressor("Promo")
    m.add_regressor("SchoolHoliday")
    return m

sonuclar_grid = []
for i, params in enumerate(kombinasyonlar):
    m = prophet_kur(params)
    m.fit(train_df)
    future = val_df[["ds", "Promo", "SchoolHoliday"]]
    fc = m.predict(future)
    rmse = np.sqrt(np.mean((val_df["y"].values - fc["yhat"].values) ** 2))
    sonuclar_grid.append({**params, "val_RMSE": round(rmse, 1)})
    print(f"  [{i+1}/{len(kombinasyonlar)}] {params} -> val RMSE: {rmse:.1f}")

grid_df = pd.DataFrame(sonuclar_grid).sort_values("val_RMSE").reset_index(drop=True)
print("\n===== GRID SEARCH SONUÇLARI (val RMSE'ye göre) =====")
print(grid_df.to_string(index=False))

en_iyi = grid_df.iloc[0]
best_params = {"changepoint_prior_scale": en_iyi["changepoint_prior_scale"],
               "seasonality_prior_scale": en_iyi["seasonality_prior_scale"]}
print(f"\nEn iyi parametreler: {best_params}")

# ================== 4) EN İYİ PARAMETRELERLE FİNAL (train+val ile eğit, test'te ölç) ==================
trval = pd.concat([train_df, val_df])
final = prophet_kur(best_params)
final.fit(trval)

future_test = test_df[["ds", "Promo", "SchoolHoliday"]]
fc_test = final.predict(future_test)
pred = fc_test["yhat"].values
y_test = test_df["y"].values

sonuc = tum_metrikler(y_test, pred, "Prophet (grid search ile optimize)")

# ================== 5) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

plt.figure(figsize=(11, 4))
plt.plot(test_df["ds"], y_test, marker="o", label="Gerçek", color="black")
plt.plot(test_df["ds"], pred, marker="x", label="Prophet (tuned)", color="darkorange")
plt.fill_between(test_df["ds"], fc_test["yhat_lower"], fc_test["yhat_upper"],
                 color="orange", alpha=0.2, label="%90 Güven Aralığı")
plt.title("Store 1 — Prophet (grid search optimize) vs Gerçek (Test)")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/29_prophet_tuned_pred.png")

# Grid search ısı haritası benzeri bar
plt.figure(figsize=(9, 4))
grid_df["etiket"] = grid_df.apply(
    lambda r: f"cp={r['changepoint_prior_scale']}\nsp={r['seasonality_prior_scale']}", axis=1)
plt.bar(range(len(grid_df)), grid_df["val_RMSE"], color="darkorange")
plt.xticks(range(len(grid_df)), grid_df["etiket"], fontsize=7, rotation=0)
plt.title("Prophet Grid Search — Kombinasyon Bazında Validation RMSE (düşük=iyi)")
plt.ylabel("Val RMSE"); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/30_prophet_grid.png")

print("\nGrafikler: figs/29_prophet_tuned_pred.png, figs/30_prophet_grid.png")
plt.show()