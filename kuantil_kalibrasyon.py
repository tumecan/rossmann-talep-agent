"""
kuantil_kalibrasyon.py — p10/p90 bandinin kalibre edilmesi

SORUN
    model_validasyon.py sunu olctu: gozlenen kapsama %77.4, nominal %80.
    Ihlaller SIMETRIK (p10 alti %11.6, p90 ustu %11.0), yani bant kayik
    degil DAR. Kupiec POF testi LR=110 ile H0'i reddediyor: sapma kucuk
    ama 26.845 gozlemde istatistiksel olarak anlamli.

    Bu sadece bir metrik sorunu degil. karar_tool.py'deki belirsizlik
    kapisi "kapsama < %70" diyor ve 217 magaza buna takiliyor. Bant
    sistematik olarak darsa, o magazalarin bir kismi HAKSIZ yere eleniyor
    ve onay yuku yapay olarak sisiyor.

YONTEM — Conformalized Quantile Regression (CQR), normalize edilmis
    Her gozlem icin uyumsuzluk skoru:
        E_i = max(p10_i - y_i, y_i - p90_i) / (p90_i - p10_i)
    Skor pozitifse gercek deger bandin DISINDA, negatifse icinde.
    Bant genisligine bolunmesi sebebi: buyuk ve kucuk magazalar ayni
    mutlak duzeltmeyi alirsa kucuk magazalarin bandi asiri genisler
    (veri heteroskedastik).

    Kalibrasyon setinde skorlarin (1-alpha) kuantili Q bulunur. Yeni bant:
        [p10 - Q*w, p90 + Q*w]
    Q pozitifse bant genisler, negatifse daralir.

    CQR'nin sonlu-orneklem kapsama garantisi vardir: kalibrasyon ve test
    degisilebilir (exchangeable) ise yeni bant nominal seviyeyi saglar.

VERI BOLUMU — ve durustluk notu
    Dogru yol validasyon doneminde kalibre edip testte olcmektir. Ancak
    tahmin tablosu YALNIZCA test penceresini (4-31 Temmuz) icerir;
    validasyon donemi tahminleri uretilmemistir.

    Bu yuzden test penceresi ikiye bolunur: ilk 14 gun KALIBRASYON,
    son 14 gun OLCUM. Olcum seti kalibrasyonda hic kullanilmaz, dolayisiyla
    sonuc gecerlidir — ama bu bir URETIM kalibrasyonu degil, YONTEM
    GOSTERIMIDIR. Uretimde ayri bir validasyon donemi kullanilmalidir.
    README'de bu sinir acikca belirtilir.

CIKTI
    Rossman/kalibrasyon_katsayilari.json  — Q degerleri
    Rossman/tahminler_kalibre.parquet     — duzeltilmis bantlar (opsiyonel)

Calistirma:
    python kuantil_kalibrasyon.py
    python kuantil_kalibrasyon.py --ham "Rossman/data/train.csv"
    python kuantil_kalibrasyon.py --yaz      # kalibre tahminleri kaydet
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools_toplu import ROSSMAN, _tahmin

ALPHA = 0.20                 # nominal %80 aralik
KI_KARE_KRITIK_95 = 3.841
KALIBRASYON_GUN = 14         # ilk 14 gun kalibrasyon, kalani olcum


def ham_bul(elle: str | None) -> Path:
    if elle:
        y = Path(elle)
        if y.exists():
            return y
        sys.exit(f"HATA: {y} bulunamadi.")
    for a in (ROSSMAN / "data" / "train.csv", ROSSMAN / "train.csv",
              Path("data/train.csv"), Path("train.csv")):
        if a.exists():
            return a
    sys.exit("HATA: train.csv bulunamadi. --ham ile yolunu verin.")


def veri(ham_yol: Path) -> pd.DataFrame:
    t = _tahmin()
    t = t[t["senaryo"] == "planli"].copy()

    ham = pd.read_csv(ham_yol, parse_dates=["Date"], low_memory=False)
    ham = ham.rename(columns={"Store": "magaza", "Date": "tarih",
                              "Sales": "gercek_eur", "Open": "ham_acik"})
    d = t.merge(ham[["magaza", "tarih", "gercek_eur", "ham_acik"]],
                on=["magaza", "tarih"], how="inner")

    # KAPALI GUNLER KALIBRASYONA GIRMEZ. O gunlerde model zaten 0 tahmin
    # ediyor ve gercek satis 0; bant genisligi de 0. Skor hesabi sifira
    # bolme uretir ve kalibrasyonu bozar.
    d = d[(d["ham_acik"] != 0) & (d["acik"] != 0)]
    d = d[(d["p90_eur"] - d["p10_eur"]) > 1e-6]
    return d


def kapsama_olc(y, alt, ust) -> dict:
    n = len(y)
    a_ihlal = int(np.sum(y < alt))
    u_ihlal = int(np.sum(y > ust))
    ihlal = a_ihlal + u_ihlal
    oran = ihlal / n if n else float("nan")

    if n == 0:
        lr = float("nan")
    elif ihlal == 0:
        lr = -2 * (n * math.log(1 - ALPHA))
    elif ihlal == n:
        lr = -2 * (n * math.log(ALPHA))
    else:
        l0 = (n - ihlal) * math.log(1 - ALPHA) + ihlal * math.log(ALPHA)
        l1 = (n - ihlal) * math.log(1 - oran) + ihlal * math.log(oran)
        lr = -2 * (l0 - l1)

    return {
        "n": n,
        "kapsama": round((1 - oran) * 100, 2),
        "p10_alti": round(a_ihlal / n * 100, 2) if n else float("nan"),
        "p90_ustu": round(u_ihlal / n * 100, 2) if n else float("nan"),
        "ort_bant_eur": round(float(np.mean(ust - alt)), 1),
        "kupiec_lr": round(lr, 2),
        "kupiec_red": bool(lr > KI_KARE_KRITIK_95),
    }


def yazdir(baslik: str, m: dict) -> None:
    print(f"\n{baslik}")
    print(f"  Kapsama      : %{m['kapsama']}   (hedef %80)")
    print(f"  p10 alti     : %{m['p10_alti']}   p90 ustu: %{m['p90_ustu']}")
    print(f"  Ort. bant    : {m['ort_bant_eur']} EUR")
    print(f"  Kupiec LR    : {m['kupiec_lr']}  -> H0 "
          f"{'REDDEDILDI' if m['kupiec_red'] else 'reddedilemedi'}")


def main() -> None:
    a = sys.argv
    d = veri(ham_bul(a[a.index("--ham") + 1] if "--ham" in a else None))

    gunler = sorted(d["tarih"].unique())
    if len(gunler) < KALIBRASYON_GUN + 7:
        sys.exit(f"HATA: {len(gunler)} gun var, bolme icin yetersiz.")
    kesme = pd.Timestamp(gunler[KALIBRASYON_GUN - 1])

    kal = d[d["tarih"] <= kesme]
    olc = d[d["tarih"] > kesme]

    print("=" * 74)
    print("KUANTIL KALIBRASYONU — Conformalized Quantile Regression")
    print("=" * 74)
    print(f"Kalibrasyon : {kal.tarih.min().date()} - {kal.tarih.max().date()}"
          f"  ({len(kal)} gozlem)")
    print(f"Olcum       : {olc.tarih.min().date()} - {olc.tarih.max().date()}"
          f"  ({len(olc)} gozlem)")
    print("\nOlcum seti kalibrasyonda HIC kullanilmaz.")

    # --- kalibrasyon skorlari --------------------------------------------
    def skor(x: pd.DataFrame) -> np.ndarray:
        y = x["gercek_eur"].to_numpy(float)
        lo = x["p10_eur"].to_numpy(float)
        hi = x["p90_eur"].to_numpy(float)
        w = hi - lo
        return np.maximum(lo - y, y - hi) / w

    e = skor(kal)
    n = len(e)
    # CQR'de kuantil seviyesi sonlu-orneklem duzeltmesi ile alinir
    seviye = min(1.0, math.ceil((n + 1) * (1 - ALPHA)) / n)
    Q = float(np.quantile(e, seviye))

    print("\nUyumsuzluk skorlari (normalize):")
    print(f"  ortalama {e.mean():+.3f}   medyan {np.median(e):+.3f}   "
          f"%{(1-ALPHA)*100:.0f}. kuantil {Q:+.3f}")
    if Q > 0:
        print(f"  Q > 0 -> bant DAR, %{Q*100:.1f} oraninda genisletilmeli.")
    else:
        print("  Q < 0 -> bant GENIS, daraltilabilir.")

    # --- olcum setinde once/sonra ----------------------------------------
    yo = olc["gercek_eur"].to_numpy(float)
    lo = olc["p10_eur"].to_numpy(float)
    hi = olc["p90_eur"].to_numpy(float)
    w = hi - lo

    once = kapsama_olc(yo, lo, hi)
    sonra = kapsama_olc(yo, lo - Q * w, hi + Q * w)

    print("\n" + "-" * 74)
    print("OLCUM SETINDE SONUC (kalibrasyonda kullanilmayan 14 gun)")
    print("-" * 74)
    yazdir("ONCE  (ham bant)", once)
    yazdir("SONRA (kalibre bant)", sonra)

    print("\n--- Degerlendirme ---")
    d_kapsama = sonra["kapsama"] - once["kapsama"]
    d_bant = (sonra["ort_bant_eur"] / once["ort_bant_eur"] - 1) * 100
    print(f"  Kapsama  : %{once['kapsama']} -> %{sonra['kapsama']}  "
          f"({d_kapsama:+.2f} puan)")
    print(f"  Bant     : {d_bant:+.1f}% genislik degisimi")
    if once["kupiec_red"] and not sonra["kupiec_red"]:
        print("  Kupiec   : REDDEDILIYORDU -> artik reddedilemiyor.")
        print("             Aralik tahmini nominal seviyeyle tutarli hale geldi.")
    elif not sonra["kupiec_red"]:
        print("  Kupiec   : her iki durumda da reddedilemiyor.")
    else:
        print("  Kupiec   : hala reddediliyor. Tek bir global katsayi yetmiyor;")
        print("             segment bazli kalibrasyon gerekebilir.")

    print("\n  TAKAS: bant genisledikce kapsama artar ama bant BILGI kaybeder.")
    print("  Cok genis bir bant her zaman kapsar ve hicbir sey soylemez.")
    print("  Karar zincirindeki etkisi: karar_tool.py belirsizlik kapisi")
    print("  'kapsama < %70' diyor; bant duzeldikce haksiz yere elenen")
    print("  magaza sayisi azalir ve onay yuku duser.")

    # --- segment bazli (yalnizca bilgi) -----------------------------------
    print("\n--- Segment bazli Q (bilgi amacli, uygulanmadi) ---")
    sat = []
    for tip, alt in kal.groupby("magaza_tipi"):
        if len(alt) < 200:
            continue
        ee = skor(alt)
        nn = len(ee)
        sv = min(1.0, math.ceil((nn + 1) * (1 - ALPHA)) / nn)
        sat.append({"segment": f"magaza_tipi={tip}", "n": nn,
                    "Q": round(float(np.quantile(ee, sv)), 3)})
    if sat:
        print(pd.DataFrame(sat).to_string(index=False))
        print("  Q'lar segmentler arasinda belirgin farkliysa tek global")
        print("  katsayi yetersizdir; segment bazli kalibrasyon dusunulmeli.")

    # --- kayit -------------------------------------------------------------
    cikti = {
        "yontem": "normalize edilmis CQR",
        "alpha": ALPHA,
        "kalibrasyon_gun": KALIBRASYON_GUN,
        "kalibrasyon_gozlem": int(len(kal)),
        "olcum_gozlem": int(len(olc)),
        "Q": round(Q, 4),
        "olcum_once": once,
        "olcum_sonra": sonra,
        "segment_Q": sat,
        "not": ("Test penceresi ikiye bolunerek kalibre edildi; uretimde "
                "ayri bir validasyon donemi kullanilmalidir."),
    }
    yol = (ROSSMAN if ROSSMAN.exists() else Path(".")) / "kalibrasyon_katsayilari.json"
    yol.write_text(json.dumps(cikti, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nKatsayilar kaydedildi: {yol}")

    # --- kalibre tahminleri yaz -------------------------------------------
    if "--yaz" in a:
        t = _tahmin().copy()
        g = (t["p90_eur"] - t["p10_eur"]).clip(lower=0)
        t["p10_eur_kalibre"] = t["p10_eur"] - Q * g
        t["p90_eur_kalibre"] = t["p90_eur"] + Q * g
        ga = (t["p90_adet"] - t["p10_adet"]).clip(lower=0)
        t["p10_adet_kalibre"] = t["p10_adet"] - Q * ga
        t["p90_adet_kalibre"] = t["p90_adet"] + Q * ga
        hedef = (ROSSMAN if ROSSMAN.exists() else Path(".")) / "tahminler_kalibre.parquet"
        t.to_parquet(hedef, index=False)
        print(f"Kalibre tahminler: {hedef}")
        print("NOT: karar_tool.py hala HAM bantlari kullanir. Kalibre bantlara")
        print("gecmek icin _guven_araligi icindeki kolon adlari degistirilmeli;")
        print("bu, karar kalitesi yeniden olculerek yapilmalidir.")


if __name__ == "__main__":
    main()