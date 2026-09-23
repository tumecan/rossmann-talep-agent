"""
model_validasyon.py — Tahmin modeli validasyon testleri

NEDEN BU SCRIPT VAR
    Egitim notebook'u magaza bazli sMAPE ve kapsama uretiyor. Bunlar dogru
    metrikler ama YETERSIZ:
      - sMAPE hatanin BUYUKLUGUNU olcer, YONUNU olcmez. Stok kararinda
        yon kritiktir: sistematik yuksek tahmin surekli fazla siparis ve
        fire demektir, dusuk tahmin surekli stoksuzluk.
      - Kapsama tek bir sayidir; bandin DAR mi yoksa KAYIK mi oldugunu
        soylemez. %60 kapsama, simetrik dar bir banttan da gelebilir,
        tamamen yukari kaymis bir banttan da. Ikisinin cozumu farklidir.
      - Kuantil tahmin icin uygun skorlama kurali PINBALL LOSS'tur;
        sMAPE yalnizca p50'yi degerlendirir, p10 ve p90 hic olculmez.

SIZINTI NOTU
    Servis tablosu (tahminler_tum_magazalar.parquet) gercek satis kolonu
    ICERMEZ; bu, egitim notebook'unda bilincli bir tasarim karari ve
    dogrudur. Bu yuzden gercek degerler HAM VERIDEN (train.csv) okunur ve
    yalnizca validasyon amaciyla kullanilir. Uretim hattina girmez.

TESTLER
    1. Nokta tahmin dogrulugu : sMAPE, wMAPE, MAE
    2. Sistematik sapma       : bias (MPE), yon dagilimi
    3. Kuantil kalitesi       : pinball loss (p10/p50/p90)
    4. Aralik backtest'i      : kapsama + AYRISTIRMA + Kupiec POF testi
    5. Segment kararliligi    : magaza tipi, promosyon, gun, talep dilimi

Calistirma:
    python model_validasyon.py
    python model_validasyon.py --ham "C:/yol/train.csv"
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools_toplu import ROSSMAN, _tahmin

# 80'lik aralik (p10-p90) icin beklenen ihlal orani
BEKLENEN_IHLAL = 0.20
KI_KARE_KRITIK_95 = 3.841  # 1 serbestlik derecesi, %95


# ===========================================================================
def ham_veri_bul(elle: str | None) -> Path:
    if elle:
        y = Path(elle)
        if y.exists():
            return y
        sys.exit(f"HATA: {y} bulunamadi.")
    for aday in (ROSSMAN / "data" / "train.csv", ROSSMAN / "train.csv",
                 ROSSMAN / "veri" / "train.csv", Path("data/train.csv"),
                 Path("train.csv"), Path("veri/train.csv")):
        if aday.exists():
            return aday
    sys.exit("HATA: train.csv bulunamadi. --ham ile yolunu verin.")


def veri_hazirla(ham_yol: Path) -> pd.DataFrame:
    """Tahmin ve gercek degeri test penceresinde birlestirir."""
    t = _tahmin()
    t = t[t["senaryo"] == "planli"].copy()

    ham = pd.read_csv(ham_yol, parse_dates=["Date"], low_memory=False)
    ham = ham.rename(columns={"Store": "magaza", "Date": "tarih",
                              "Sales": "gercek_eur", "Open": "ham_acik"})
    ham = ham[["magaza", "tarih", "gercek_eur", "ham_acik"]]

    d = t.merge(ham, on=["magaza", "tarih"], how="inner")
    if d.empty:
        sys.exit("HATA: tahmin ile ham veri kesismiyor (tarih araligi?).")

    # KAPALI GUNLER AYRI TUTULUR. Kapali gunde gercek satis sifirdir ve
    # model pozitif tahmin eder; bu gunleri metrige katmak modeli olculemez
    # bir seyden sorumlu tutmak olur. Ayrica raporlanir.
    d["kapali"] = (d["ham_acik"] == 0) | (d["acik"] == 0)
    return d


# ===========================================================================
# METRIKLER
# ===========================================================================
def smape(y, f) -> float:
    payda = (np.abs(y) + np.abs(f)) / 2
    gecerli = payda > 0
    if not gecerli.any():
        return float("nan")
    return float(np.mean(np.abs(f[gecerli] - y[gecerli]) / payda[gecerli]) * 100)


def wmape(y, f) -> float:
    """Hacim agirlikli MAPE: sum|e| / sum|y|.

    MAPE'nin sifira bolme sorunu yoktur ve buyuk magazalari hak ettigi
    agirlikta sayar. Perakende talep tahmininde tercih edilen metriktir:
    toplam is hacmi uzerinden hata, magaza ortalamasi degil.
    """
    tp = np.sum(np.abs(y))
    return float(np.sum(np.abs(f - y)) / tp * 100) if tp > 0 else float("nan")


def bias_yuzde(y, f) -> float:
    """Sistematik sapma: (tahmin - gercek) / gercek, toplam uzerinden.

    Pozitif = model SISTEMATIK YUKSEK tahmin ediyor -> fazla siparis, fire.
    Negatif = sistematik dusuk -> stoksuzluk, kacan satis.
    Sifira yakin olmasi sMAPE'in dusuk olmasindan daha kritiktir, cunku
    sapma zamanla BIRIKIR.
    """
    tp = np.sum(y)
    return float(np.sum(f - y) / tp * 100) if tp > 0 else float("nan")


def pinball(y, tahmin_q, q: float) -> float:
    """Kuantil (pinball) kaybi — kuantil tahmini icin uygun skorlama kurali.

    q=0.1 icin: gercek deger tahminin USTUNDEyse ceza 0.9 katsayiyla,
    ALTINDAysa 0.1 katsayiyla. Yani p10'un cok yuksek olmasi agir cezalanir.
    Dusuk = iyi. Yalnizca ayni veri setinde modeller arasi kiyaslanir.
    """
    e = y - tahmin_q
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def kupiec_pof(ihlal: int, n: int, beklenen: float = BEKLENEN_IHLAL) -> dict:
    """Kupiec kosulsuz kapsama testi (POF — Proportion of Failures).

    Basel'deki VaR backtest'inin ta kendisi: gozlenen ihlal orani, nominal
    orandan istatistiksel olarak farkli mi? p10-p90 araligi da bir aralik
    tahminidir, dolayisiyla ayni test uygulanir.

    H0: gercek ihlal orani = beklenen (0.20)
    LR ~ ki-kare(1). LR > 3.841 ise H0 %95 guvenle REDDEDILIR.
    """
    if n == 0:
        return {"lr": float("nan"), "red": False, "oran": float("nan")}
    x, oran = ihlal, ihlal / n
    if x == 0:
        lr = -2 * (n * math.log(1 - beklenen))
    elif x == n:
        lr = -2 * (n * math.log(beklenen))
    else:
        l0 = (n - x) * math.log(1 - beklenen) + x * math.log(beklenen)
        l1 = (n - x) * math.log(1 - oran) + x * math.log(oran)
        lr = -2 * (l0 - l1)
    return {"lr": round(lr, 2), "red": lr > KI_KARE_KRITIK_95,
            "oran": round(oran, 4)}


def olc(d: pd.DataFrame) -> dict:
    y = d["gercek_eur"].to_numpy(float)
    p10 = d["p10_eur"].to_numpy(float)
    p50 = d["p50_eur"].to_numpy(float)
    p90 = d["p90_eur"].to_numpy(float)

    alt_ihlal = int(np.sum(y < p10))
    ust_ihlal = int(np.sum(y > p90))
    n = len(d)

    return {
        "n": n,
        "sMAPE": round(smape(y, p50), 2),
        "wMAPE": round(wmape(y, p50), 2),
        "MAE_eur": round(float(np.mean(np.abs(p50 - y))), 1),
        "bias_%": round(bias_yuzde(y, p50), 2),
        "yuksek_tahmin_%": round(float(np.mean(p50 > y)) * 100, 1),
        "pinball_p10": round(pinball(y, p10, 0.10), 1),
        "pinball_p50": round(pinball(y, p50, 0.50), 1),
        "pinball_p90": round(pinball(y, p90, 0.90), 1),
        "kapsama_%": round((1 - (alt_ihlal + ust_ihlal) / n) * 100, 1) if n else float("nan"),
        "p10_alti_%": round(alt_ihlal / n * 100, 1) if n else float("nan"),
        "p90_ustu_%": round(ust_ihlal / n * 100, 1) if n else float("nan"),
    }


# ===========================================================================
def rapor(d: pd.DataFrame) -> None:
    acik = d[~d["kapali"]]
    kapali = d[d["kapali"]]

    print("=" * 74)
    print("MODEL VALIDASYON RAPORU")
    print("=" * 74)
    print(f"Gozlem            : {len(d)}")
    print(f"  Kapali gun      : {len(kapali)}  (metrik disi)")
    print(f"  Degerlendirilen : {len(acik)}")
    print(f"Tarih araligi     : {d.tarih.min().date()} - {d.tarih.max().date()}")

    m = olc(acik)
    print("\n--- 1. NOKTA TAHMIN DOGRULUGU ---")
    print(f"  sMAPE            : %{m['sMAPE']}")
    print(f"  wMAPE            : %{m['wMAPE']}   <- hacim agirlikli, is hacmi uzerinden")
    print(f"  MAE              : {m['MAE_eur']} EUR/gun")
    if m["wMAPE"] < m["sMAPE"]:
        print("  NOT: wMAPE < sMAPE -> hata kucuk magazalarda yogunlasiyor.")
    else:
        print("  NOT: wMAPE >= sMAPE -> hata buyuk magazalarda yogunlasiyor;")
        print("       is etkisi magaza ortalamasinin gosterdiginden buyuk.")

    print("\n--- 2. SISTEMATIK SAPMA (BIAS) ---")
    print(f"  bias             : %{m['bias_%']}")
    print(f"  yuksek tahmin    : gunlerin %{m['yuksek_tahmin_%']}'inde")
    if abs(m["bias_%"]) < 2:
        print("  Sapma ihmal edilebilir.")
    elif m["bias_%"] > 0:
        print("  Model SISTEMATIK YUKSEK tahmin ediyor -> fazla siparis ve")
        print("  fire egilimi. Stok kararinda bu sapma ZAMANLA BIRIKIR.")
    else:
        print("  Model SISTEMATIK DUSUK tahmin ediyor -> stoksuzluk ve")
        print("  kacan satis egilimi.")

    print("\n--- 3. KUANTIL KALITESI (pinball loss, dusuk=iyi) ---")
    print(f"  p10 : {m['pinball_p10']}    p50 : {m['pinball_p50']}    "
          f"p90 : {m['pinball_p90']}")
    print("  Yalnizca ayni veri setinde modeller arasi kiyaslanir;")
    print("  mutlak degeri tek basina yorumlanmaz.")

    print("\n--- 4. ARALIK BACKTEST'I (p10-p90, nominal %80) ---")
    print(f"  Gozlenen kapsama : %{m['kapsama_%']}   (hedef %80)")
    print(f"  p10 ALTINDA      : %{m['p10_alti_%']}   (beklenen %10)")
    print(f"  p90 USTUNDE      : %{m['p90_ustu_%']}   (beklenen %10)")

    fark = m["p10_alti_%"] - m["p90_ustu_%"]
    if abs(fark) < 3:
        print("  Ihlaller SIMETRIK -> bant dar (veya genis), kayik degil.")
        print("  Cozum: kuantil kalibrasyonu (bant genisligi).")
    elif fark > 0:
        print("  Ihlaller ASIMETRIK: alt tarafta yogun -> bant YUKARI kayik,")
        print("  yani model dusuk degerleri yeterince ongormuyor.")
    else:
        print("  Ihlaller ASIMETRIK: ust tarafta yogun -> bant ASAGI kayik,")
        print("  yani model yuksek talep gunlerini kaciriyor.")

    ihlal = int(round((100 - m["kapsama_%"]) / 100 * m["n"]))
    k = kupiec_pof(ihlal, m["n"])
    print(f"\n  Kupiec POF testi : LR = {k['lr']}  (kritik deger 3.841)")
    print(f"  Sonuc            : H0 {'REDDEDILDI' if k['red'] else 'reddedilemedi'}")
    if k["red"]:
        print("  Gozlenen ihlal orani nominal %20'den ISTATISTIKSEL OLARAK")
        print("  farkli. Aralik tahmini kalibre degil; kuantil kalibrasyonu")
        print("  gozden gecirilmeli.")
    else:
        print("  Aralik tahmini nominal seviyeyle tutarli.")

    # --- 5. SEGMENT KARARLILIGI ------------------------------------------
    print("\n--- 5. SEGMENT KARARLILIGI ---")
    satirlar = []

    def ekle(segment: str, deger, alt: pd.DataFrame):
        if len(alt) < 30:
            return
        r = olc(alt)
        r.update({"segment": segment, "deger": str(deger)})
        satirlar.append(r)

    for tip, alt in acik.groupby("magaza_tipi"):
        ekle("magaza_tipi", tip, alt)
    for pr, alt in acik.groupby("promo"):
        ekle("promosyon", "var" if pr else "yok", alt)
    for gun, alt in acik.groupby("gun_adi"):
        ekle("gun", gun, alt)

    # Talep buyuklugu dilimleri: model kucuk magazalarda mi zayif?
    a = acik.copy()
    a["dilim"] = pd.qcut(a["p50_eur"], 5, labels=["Q1 (en dusuk)", "Q2", "Q3",
                                                  "Q4", "Q5 (en yuksek)"],
                         duplicates="drop")
    for dl, alt in a.groupby("dilim", observed=True):
        ekle("talep_dilimi", dl, alt)

    t = pd.DataFrame(satirlar)[
        ["segment", "deger", "n", "sMAPE", "wMAPE", "bias_%",
         "kapsama_%", "p10_alti_%", "p90_ustu_%"]]
    print(t.to_string(index=False))

    yol = ROSSMAN / "model_validasyon.csv" if ROSSMAN.exists() \
        else Path("model_validasyon.csv")
    t.to_csv(yol, index=False)

    print("\nYORUM")
    print("Segmentler arasinda sMAPE veya bias belirgin sekilde farkliysa")
    print("model o segmentte kararsizdir; tek bir ortalama metrik bunu")
    print("gizler. Ozellikle bias'in segmentler arasi ISARET DEGISTIRMESI")
    print("onemlidir: bir segmentte fazla, digerinde eksik siparis demektir")
    print("ve toplamda birbirini goturerek 'sorun yok' izlenimi verir.")
    print(f"\nKaydedildi: {yol}")

    # Kapali gunlerin etkisi
    if len(kapali) > 20:
        mk = olc(kapali)
        print("\n--- EK: KAPALI GUNLER DAHIL EDILSEYDI ---")
        print(f"  sMAPE %{m['sMAPE']} -> %{olc(d)['sMAPE']}   "
              f"kapsama %{m['kapsama_%']} -> %{olc(d)['kapsama_%']}")
        print(f"  Kapali gunlerde model ortalama {mk['MAE_eur']} EUR tahmin")
        print("  ediyor, gercek satis sifir. Bu gunleri metrige katmak")
        print("  modeli olculemez bir seyden sorumlu tutmaktir.")


def main() -> None:
    a = sys.argv
    ham = a[a.index("--ham") + 1] if "--ham" in a else None
    d = veri_hazirla(ham_veri_bul(ham))
    rapor(d)


if __name__ == "__main__":
    main()