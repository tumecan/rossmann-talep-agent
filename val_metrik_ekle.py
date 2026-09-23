"""
val_metrik_ekle.py — Mevcut tahmin dosyasina VAL donemi kapsamasini ekler.

NEDEN
    Belirsizlik kapisi yalniz karar aninda bilinen (val donemi) metriklerle
    beslenir. Eski tahmin dosyasinda magaza_kapsama_val kolonu yok. Bu betik,
    Colab'i yeniden kosmadan kolonu global_model_sonuc.json icindeki val
    magaza kapsamasindan doldurur.

SINIR (README'de belirtilir)
    global_model_sonuc.json'daki val kapsamasi KALIBRASYON ONCESIdir
    (k = 1.05 uygulanmamis). Dolayisiyla biraz karamsardir; kapi gercekte
    olacagindan birkac magaza fazla kapanabilir. Kalici cozum guncel
    colab_03_tahmin_uret: kalibre val kapsamasi ve magaza_n_val kolonunu
    dogrudan uretir. Tahmin dosyasi o notebook'la uretildiyse bu betige
    gerek yoktur.

Calistirma:
    python val_metrik_ekle.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parent
ROSSMAN = KOK / "Rossman"
TAHMIN = ROSSMAN / "model" / "tahminler_tum_magazalar.parquet"
ADAYLAR = [ROSSMAN / "model" / "global_model_sonuc.json",
           ROSSMAN / "global_model_sonuc.json",
           KOK / "global_model_sonuc.json"]


def main() -> None:
    if not TAHMIN.exists():
        sys.exit(f"HATA: {TAHMIN} yok.")
    kaynak = next((p for p in ADAYLAR if p.exists()), None)
    if kaynak is None:
        sys.exit("HATA: global_model_sonuc.json bulunamadi: "
                 + ", ".join(str(p) for p in ADAYLAR))

    sonuc = json.loads(kaynak.read_text(encoding="utf-8"))
    kap_val = {int(k): float(v) for k, v in sonuc["val"]["magaza_kapsama"].items()}

    t = pd.read_parquet(TAHMIN)
    if "magaza_kapsama_val" in t.columns:
        print("magaza_kapsama_val zaten var (guncel colab_03 ciktisi) — dokunulmadi.")
        return

    t["magaza_kapsama_val"] = t["magaza"].map(kap_val)
    eksik = int(t.loc[t["senaryo"] == "planli", "magaza_kapsama_val"].isna().sum())
    t.to_parquet(TAHMIN, index=False)

    m = t[t["senaryo"] == "planli"].groupby("magaza")["magaza_kapsama_val"].first()
    print(f"kaynak           : {kaynak}")
    print(f"eklendi          : magaza_kapsama_val ({m.notna().sum()} magaza, "
          f"{eksik} eksik satir)")
    print(f"val kapsama      : ort %{m.mean():.1f} | medyan %{m.median():.1f} "
          f"| min %{m.min():.1f}")
    print("not              : kalibrasyon ONCESI deger; guncel colab_03 kalibre "
          "degeri ve magaza_n_val'i uretir.")


if __name__ == "__main__":
    main()