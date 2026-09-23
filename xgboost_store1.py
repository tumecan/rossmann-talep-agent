# Rossman/xgboost_store1.py — XGBoost baseline (Store 1)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from metrics import tum_metrikler

# ================== 1) VERİ + ÖZELLİKLER ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
train["Date"] = pd.to_datetime(train["Date"])

s1 = train[(train["Store"] == 1) & (train["Open"] == 1)].copy()
s1 = s1.sort_values("Date").reset_index(drop=True)

# --- Takvim özellikleri ---
s1["Year"]  = s1["Date"].dt.year
s1["Month"] = s1["Date"].dt.month
s1["Day"]   = s1["Date"].dt.day
s1["Week"]  = s1["Date"].dt.isocalendar().week.astype(int)
s1["dow_sin"] = np.sin(2 * np.pi * s1["DayOfWeek"] / 7)
s1["dow_cos"] = np.cos(2 * np.pi * s1["DayOfWeek"] / 7)
s1["StateHolidayBin"] = (s1["StateHoliday"].astype(str) != "0").astype(int)

# --- Lag / rolling özellikleri (LEAKAGE'sız: hepsi shift(1) ile) ---
# NOT: Customers kolonu BİLEREK kullanılmıyor. Müşteri sayısı satışla aynı anda
# oluşan bir sonuç değişkenidir; tahmin anında bilinemez. Feature olarak konsaydı
# R² sahte biçimde yükselirdi (hedef sızıntısı).
s1["Sales_Lag1"]    = s1["Sales"].shift(1)
s1["Sales_Lag7"]    = s1["Sales"].shift(7)     # geçen hafta aynı gün
s1["Sales_Roll7"]   = s1["Sales"].shift(1).rolling(7).mean()
s1["Sales_Roll14"]  = s1["Sales"].shift(1).rolling(14).mean()
s1["Sales_Roll30"]  = s1["Sales"].shift(1).rolling(30).mean()

s1 = s1.dropna().reset_index(drop=True)

feat_cols = ["Promo", "SchoolHoliday", "StateHolidayBin",
             "Year", "Month", "Day", "Week", "dow_sin", "dow_cos",
             "Sales_Lag1", "Sales_Lag7", "Sales_Roll7", "Sales_Roll14", "Sales_Roll30"]

# ================== 2) TARİHE GÖRE BÖL ==================
TEST_DAYS = 42
cutoff = s1["Date"].max() - pd.Timedelta(days=TEST_DAYS)
train_df = s1[s1["Date"] <= cutoff]
test_df  = s1[s1["Date"] >  cutoff]

X_train, y_train = train_df[feat_cols], train_df["Sales"]
X_test,  y_test  = test_df[feat_cols],  test_df["Sales"]
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Özellik sayısı: {len(feat_cols)}")

# ================== 3) MODEL (ilk hali, optimize edilmemiş) ==================
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

# ================== 4) METRİKLER ==================
sonuc_xgb = tum_metrikler(y_test.values, pred, "XGBoost (Store 1, ayarsız)")

# ================== 4.5) ORTAK PENCERE KIYASI İÇİN KAYDET ==================
os.makedirs("Rossman/tahminler", exist_ok=True)
pd.DataFrame({"ds": test_df["Date"].values, "gercek": y_test.values,
              "tahmin": pred}).to_csv("Rossman/tahminler/xgboost.csv", index=False)
print("Kiyas icin kaydedildi: Rossman/tahminler/xgboost.csv")

# ================== 5) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

plt.figure(figsize=(11, 4))
plt.plot(test_df["Date"].values, y_test.values, marker="o", label="Gerçek", color="black")
plt.plot(test_df["Date"].values, pred, marker="x", label="XGBoost Tahmin", color="purple")
plt.title("Store 1 — XGBoost Tahmini vs Gerçek (Test)")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/13_xgb_pred.png")

imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
plt.figure(figsize=(8, 5))
imp.plot(kind="barh", color="purple")
plt.title("XGBoost — Özellik Önemleri (Store 1)")
plt.tight_layout(); plt.savefig("Rossman/figs/14_xgb_importance.png")

print("\nGrafikler: figs/13_xgb_pred.png, figs/14_xgb_importance.png")