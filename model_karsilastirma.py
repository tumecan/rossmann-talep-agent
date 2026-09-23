"""
model_karsilastirma.py — Tum modelleri AYNI test penceresinde kiyaslar.

NEDEN GEREKLI
Ana kiyas tablosu her modelin kendi ciktisindan toplanmisti, ama pencereler
ayni degildi: Prophet ve XGBoost 36 gunde, LSTM/GRU/SimpleRNN/Transformer
22 gunde olculuyordu. Sebep, dizi modellerinde create_sequences'in test
setinin ilk SEQ_LEN gununu girdi olarak tuketmesi. Farkli gunlerde olculmus
iki sMAPE degeri yan yana konulamaz.

Bu dosya Rossman/tahminler/ altindaki tum model ciktilarini okur, HEPSININ
tahmin urettigi ortak tarihleri bulur ve metrikleri orada yeniden hesaplar.
Ayrica iki naive baseline ekler (rubrik gerekliligi).

NAIVE-7 NOTU
Magaza Pazar gunleri kapali oldugu icin veri seti takvimsel olarak
bosluklu. shift(7) satir bazli kaydirir ve haftanin FARKLI gunune denk
gelir. Bu yuzden naive-7 burada takvim tarihi uzerinden (Date - 7 gun)
eslestiriliyor; ancak o zaman gercekten "gecen haftanin ayni gunu" olur.

Calistirma:
    python model_karsilastirma.py
    python model_karsilastirma.py --kendi-penceresi   # her modeli kendi
                                                       # penceresinde de goster
"""

import sys
from pathlib import Path

import pandas as pd

KOK_DIZIN = Path(__file__).resolve().parent
TAHMIN_DIR = KOK_DIZIN / "Rossman" / "tahminler"
HAM_VERI = KOK_DIZIN / "Rossman" / "data" / "train.csv"
CIKTI_CSV = KOK_DIZIN / "Rossman" / "model_karsilastirma.csv"
MAGAZA = 1

# Dosya adi -> tabloda gorunecek isim
ETIKETLER = {
    "prophet": "Prophet",
    "prophet_tuned": "Prophet (grid search)",
    "xgboost": "XGBoost (ayarsiz)",
    "xgboost_optuna": "XGBoost (Optuna)",
    "xgboost_all_store1": "XGBoost (tum magazalar)",
    "lstm_uni": "LSTM (univariate)",
    "lstm_mv": "LSTM (multivariate)",
    "lstm_tuned": "LSTM (Keras Tuner)",
    "lstm_family": "LSTM (rnn_family)",
    "gru": "GRU",
    "simplernn": "SimpleRNN",
    "transformer": "Transformer",
}


# ---------------------------------------------------------------- metrikler
def smape(gercek: pd.Series, tahmin: pd.Series) -> float:
    pay = (tahmin - gercek).abs()
    payda = (gercek.abs() + tahmin.abs()) / 2
    ge = payda > 0
    return float((pay[ge] / payda[ge]).mean() * 100)


def mae(gercek: pd.Series, tahmin: pd.Series) -> float:
    return float((gercek - tahmin).abs().mean())


def r2(gercek: pd.Series, tahmin: pd.Series) -> float:
    kalinti = ((gercek - tahmin) ** 2).sum()
    toplam = ((gercek - gercek.mean()) ** 2).sum()
    return float(1 - kalinti / toplam) if toplam > 0 else float("nan")


def olc(ad: str, gercek: pd.Series, tahmin: pd.Series) -> dict:
    return {"model": ad, "gun": len(gercek),
            "sMAPE_%": round(smape(gercek, tahmin), 2),
            "MAE": round(mae(gercek, tahmin), 1),
            "R2": round(r2(gercek, tahmin), 4)}


# ---------------------------------------------------------------- yukleme
def naive_serileri_kur() -> pd.DataFrame:
    """Ham veriden Store 1 acik gunlerini alir, iki naive tahminci uretir.
    naive-7 TAKVIM tarihi uzerinden eslestirilir (satir kaydirma degil)."""
    if not HAM_VERI.exists():
        print(f"UYARI: {HAM_VERI} yok, naive baseline atlaniyor.")
        return pd.DataFrame(columns=["ds", "gercek", "naive_1", "naive_7"])

    ham = pd.read_csv(HAM_VERI, low_memory=False)
    ham["Date"] = pd.to_datetime(ham["Date"])
    s = ham[(ham["Store"] == MAGAZA) & (ham["Open"] == 1)][["Date", "Sales"]]
    s = s.sort_values("Date").reset_index(drop=True)
    s.columns = ["ds", "gercek"]

    # naive-1: bir onceki ACIK gunun gercek satisi
    s["naive_1"] = s["gercek"].shift(1)

    # naive-7: TAM 7 takvim gunu onceki tarihin gercek satisi
    gecmis = s[["ds", "gercek"]].rename(columns={"ds": "ds_7", "gercek": "naive_7"})
    gecmis["ds"] = gecmis["ds_7"] + pd.Timedelta(days=7)
    s = s.merge(gecmis[["ds", "naive_7"]], on="ds", how="left")
    return s


def tahminleri_yukle() -> dict:
    if not TAHMIN_DIR.exists():
        sys.exit(f"HATA: {TAHMIN_DIR} yok.\n"
                 "Once model scriptlerine tahmin kaydetme blogunu ekleyip "
                 "yeniden calistir.")
    tablolar = {}
    for yol in sorted(TAHMIN_DIR.glob("*.csv")):
        df = pd.read_csv(yol, parse_dates=["ds"])
        if not {"ds", "tahmin"} <= set(df.columns):
            print(f"UYARI: {yol.name} icinde ds/tahmin kolonu yok, atlandi.")
            continue
        ad = ETIKETLER.get(yol.stem, yol.stem)
        tablolar[ad] = df[["ds", "tahmin"]].dropna().sort_values("ds")
    if not tablolar:
        sys.exit(f"HATA: {TAHMIN_DIR} bos.")
    return tablolar


# ---------------------------------------------------------------- ana akis
def main() -> None:
    tahminler = tahminleri_yukle()
    naive = naive_serileri_kur()

    print(f"Bulunan model ciktisi: {len(tahminler)}")
    for ad, df in tahminler.items():
        print(f"  {ad:28} {len(df):3} gun  "
              f"({df['ds'].min().date()} - {df['ds'].max().date()})")

    # Ortak pencere: her modelin tahmin urettigi tarihlerin kesisimi
    ortak = None
    for df in tahminler.values():
        tarihler = set(df["ds"])
        ortak = tarihler if ortak is None else (ortak & tarihler)

    if naive.empty:
        naive_var = False
    else:
        naive_gecerli = naive.dropna(subset=["naive_1", "naive_7"])
        ortak = ortak & set(naive_gecerli["ds"])
        naive_var = True

    ortak = sorted(ortak)
    if len(ortak) < 5:
        sys.exit(f"HATA: ortak pencere cok kisa ({len(ortak)} gun).")

    print(f"\nORTAK PENCERE: {len(ortak)} gun "
          f"({ortak[0].date()} - {ortak[-1].date()})")
    print("Asagidaki tum metrikler BU gunlerde hesaplandi.\n")

    # Gercek degerleri tek bir kaynaktan al (ilk tahmin dosyasindaki gercek
    # kolonu veya ham veri). Ham veri onceliklidir, daha guvenilir.
    if naive_var:
        gercek_kaynak = naive.set_index("ds")["gercek"]
    else:
        ilk = next(iter(TAHMIN_DIR.glob("*.csv")))
        d = pd.read_csv(ilk, parse_dates=["ds"])
        gercek_kaynak = d.set_index("ds")["gercek"]

    y = gercek_kaynak.loc[ortak]

    satirlar = []
    if naive_var:
        n = naive.set_index("ds").loc[ortak]
        satirlar.append(olc("naive-1 (onceki acik gun)", y, n["naive_1"]))
        satirlar.append(olc("naive-7 (gecen hafta ayni gun)", y, n["naive_7"]))

    for ad, df in tahminler.items():
        p = df.set_index("ds")["tahmin"].loc[ortak]
        satirlar.append(olc(ad, y, p))

    tablo = pd.DataFrame(satirlar).sort_values("sMAPE_%").reset_index(drop=True)
    tablo.insert(0, "sira", range(1, len(tablo) + 1))

    print("=" * 70)
    print("ORTAK PENCEREDE MODEL KIYASI (sMAPE'e gore siralandi)")
    print("=" * 70)
    print(tablo.to_string(index=False))
    tablo.to_csv(CIKTI_CSV, index=False, encoding="utf-8-sig")
    print(f"\nKaydedildi: {CIKTI_CSV}")

    # --- Yorum ---
    en_iyi = tablo.iloc[0]
    naive_satirlari = tablo[tablo["model"].str.startswith("naive")]
    print("\n--- Yorum ---")
    print(f"En iyi model: {en_iyi['model']} (sMAPE {en_iyi['sMAPE_%']})")
    if not naive_satirlari.empty:
        en_iyi_naive = naive_satirlari["sMAPE_%"].min()
        gecenler = (tablo[~tablo["model"].str.startswith("naive")]["sMAPE_%"]
                    < en_iyi_naive).sum()
        toplam = len(tablo) - len(naive_satirlari)
        print(f"En iyi naive baseline sMAPE: {en_iyi_naive}")
        print(f"Naive'i gecen model sayisi : {gecenler}/{toplam}")
        if gecenler < toplam:
            print("Bazi modeller naive tekrardan daha kotu. Bu gizlenmemeli;")
            print("model karmasikligiyla performans arasindaki iliskiyi tartis.")


if __name__ == "__main__":
    main()