# Rossman/xgboost_all_stores.py — XGBoost, TÜM mağazalar (~800K satır)
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from metrics import tum_metrikler

# ================== 1) VERİ + STORE MERGE ==================
train = pd.read_csv("Rossman/data/train.csv", low_memory=False)
store = pd.read_csv("Rossman/data/store.csv")
train["Date"] = pd.to_datetime(train["Date"])

df = train.merge(store, on="Store", how="left")
df = df[df["Open"] == 1].copy()               # kapalı günleri at
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
print("Merge sonrası satır:", len(df))

# ================== 2) ÖZELLİKLER ==================
# --- Takvim ---
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"]   = df["Date"].dt.day
df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)
df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
df["StateHolidayBin"] = (df["StateHoliday"].astype(str) != "0").astype(int)

# --- Store bilgileri: kategorikleri sayıya çevir ---
for col in ["StoreType", "Assortment"]:
    df[col] = df[col].astype("category").cat.codes   # NaN -> -1

# --- Store kaynaklı NaN'ları güvenli doldur ---
df["Promo2"] = df["Promo2"].fillna(0)
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
    df["CompetitionDistance"].median()
)

# --- Lag / rolling: HER MAĞAZA İÇİN AYRI (groupby), leakage'sız (shift(1)) ---
df["Sales_Lag1"]   = df.groupby("Store")["Sales"].shift(1)
df["Sales_Lag7"]   = df.groupby("Store")["Sales"].shift(7)
df["Sales_Roll7"]  = df.groupby("Store")["Sales"].transform(lambda x: x.shift(1).rolling(7).mean())
df["Sales_Roll30"] = df.groupby("Store")["Sales"].transform(lambda x: x.shift(1).rolling(30).mean())

# --- SADECE lag NaN'larını at (ilk günler); diğerlerini yukarıda doldurduk ---
lag_kolonlari = ["Sales_Lag1", "Sales_Lag7", "Sales_Roll7", "Sales_Roll30"]
df = df.dropna(subset=lag_kolonlari).reset_index(drop=True)
print("Lag sonrası satır:", len(df))

feat_cols = ["Store", "Promo", "SchoolHoliday", "StateHolidayBin",
             "StoreType", "Assortment", "Promo2", "CompetitionDistance",
             "Year", "Month", "Day", "Week", "dow_sin", "dow_cos",
             "Sales_Lag1", "Sales_Lag7", "Sales_Roll7", "Sales_Roll30"]

# --- Güvenlik kontrolü: feature'larda NaN kaldı mı? ---
nan_kontrol = df[feat_cols].isnull().sum()
print("\n[KONTROL] feature'larda kalan NaN:")
print(nan_kontrol[nan_kontrol > 0] if nan_kontrol.sum() > 0 else "  NaN yok ✓")

# ================== 3) TARİHE GÖRE BÖL (son 42 gün test) ==================
TEST_DAYS = 42
cutoff = df["Date"].max() - pd.Timedelta(days=TEST_DAYS)
train_df = df[df["Date"] <= cutoff]
test_df  = df[df["Date"] >  cutoff]

X_train, y_train = train_df[feat_cols], train_df["Sales"]
X_test,  y_test  = test_df[feat_cols],  test_df["Sales"]
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,} | Özellik: {len(feat_cols)}")
print(f"Test mağaza sayısı: {test_df['Store'].nunique()}")

# ================== 4) MODEL ==================
model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",   # büyük veride hızlı
    random_state=42,
    n_jobs=-1,
)
print("\nEğitiliyor (birkaç dakika sürebilir)...")
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ================== 5) METRİKLER (genel — tüm mağazalar) ==================
sonuc_genel = tum_metrikler(y_test.values, pred, "XGBoost (TÜM mağazalar)")

# ================== 6) GRAFİKLER ==================
os.makedirs("Rossman/figs", exist_ok=True)

# --- Feature importance ---
imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
plt.figure(figsize=(8, 6))
imp.plot(kind="barh", color="teal")
plt.title("XGBoost (Tüm Mağazalar) — Özellik Önemleri")
plt.tight_layout(); plt.savefig("Rossman/figs/19_xgb_all_importance.png")

# --- Store 1'i bu modelden çekip tahmin kıyası ---
s1_test = test_df[test_df["Store"] == 1].sort_values("Date").copy()
print("\n[TEŞHİS] Store 1 test satır sayısı:", len(s1_test))

s1_pred = model.predict(s1_test[feat_cols])

plt.figure(figsize=(11, 4))
plt.plot(s1_test["Date"].values, s1_test["Sales"].values, marker="o", label="Gerçek", color="black")
plt.plot(s1_test["Date"].values, s1_pred, marker="x", label="XGBoost (all) tahmin", color="teal")
plt.title("Store 1 — Tüm Mağaza Modelinden Tahmin vs Gerçek (Test)")
plt.ylabel("Satış (€)"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("Rossman/figs/20_xgb_all_store1.png")

# Store 1 için ayrıca metrik (tek-mağaza modeliyle kıyas için)
sonuc_s1 = tum_metrikler(s1_test["Sales"].values, s1_pred, "XGBoost (all) — sadece Store 1")

print("\nGrafikler: figs/19_xgb_all_importance.png, figs/20_xgb_all_store1.png")
plt.show()