"""
agent_v3.py — Gerçek model çıktısına bağlı Karar Agent'ı (Rossmann, Faz 5 - adım 2)

v2'ye göre tek fark: girdi artık örnek veri degil, Prophet'in gercek tahmini.
Hesap katmani, sema dogrulama ve kural motoru agent_v2'den import edilir.

Onkosul:
    python Rossman\\prophet_model.py   -> Rossman/prophet_forecast.csv uretir

Calistirma:
    python agent_v3.py                      # son gun icin karar
    python agent_v3.py --tarih 2015-07-15   # belirli gun
    python agent_v3.py --toplu              # tum test gunleri (42 karar)
"""

import json
import sys
from datetime import datetime

import pandas as pd

# v2'deki katmanlari aynen kullaniyoruz (kod tekrari yok)
from agent_v2 import (
    CIKTI_DIR,
    MODEL_ADI,
    PROMPT_SURUM,
    hesapla,
    istemci_olustur,
    karar_uret,
    kurallari_dayat,
    logla,
)

FORECAST_CSV = CIKTI_DIR / "prophet_forecast.csv"
KARAR_JSON = CIKTI_DIR / "agent_karar.json"
TOPLU_CSV = CIKTI_DIR / "agent_kararlari.csv"

# --- VARSAYIM: Rossmann verisinde stok bilgisi YOK ---
# Bir onceki gunun gerceklesen satisinin bu oran kadarinin elde kaldigini
# varsayiyoruz. Sabit oran -> deterministik, tekrarlanabilir senaryo.
# (Sunumda "veri setinde olmayan degisken icin acik varsayim" diye anlatilacak.)
STOK_ORANI = 0.80


def forecast_yukle() -> pd.DataFrame:
    if not FORECAST_CSV.exists():
        sys.exit(
            f"HATA: {FORECAST_CSV} yok.\n"
            "Once 'python Rossman\\prophet_model.py' calistir."
        )
    df = pd.read_csv(FORECAST_CSV, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # stok simulasyonu: bir onceki gunun gercek satisinin %80'i
    df["mevcut_stok"] = (df["gercek"].shift(1) * STOK_ORANI).round(1)
    df.loc[0, "mevcut_stok"] = round(df.loc[0, "yhat"] * STOK_ORANI, 1)
    return df


def satir_to_girdi(satir: pd.Series) -> dict:
    return {
        "magaza": 1,
        "tarih": satir["ds"].strftime("%Y-%m-%d"),
        "tahmin_satis": round(float(satir["yhat"]), 1),
        "guven_alt": round(float(satir["yhat_lower"]), 1),
        "guven_ust": round(float(satir["yhat_upper"]), 1),
        "mevcut_stok": float(satir["mevcut_stok"]),
        "promo_var_mi": bool(satir["promo"]),
    }


def tek_karar(client, girdi: dict, yazdir: bool = True) -> dict:
    hesap = hesapla(girdi)
    ham_karar, deneme, ham_cikti = karar_uret(client, girdi, hesap)
    karar, mudahaleler = kurallari_dayat(ham_karar, hesap)

    kayit = {
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_ADI,
        "prompt_surum": PROMPT_SURUM,
        "kaynak": "prophet_forecast.csv",
        "girdi": girdi,
        "hesaplanan": hesap,
        "karar": karar.model_dump(),
        "mudahaleler": mudahaleler,
        "deneme_sayisi": deneme,
    }
    logla({**kayit, "llm_ham_cikti": ham_cikti})

    if yazdir:
        print(f"\n========== {girdi['tarih']} | Magaza {girdi['magaza']} ==========")
        print(f"Prophet tahmini : {girdi['tahmin_satis']} "
              f"[{girdi['guven_alt']} - {girdi['guven_ust']}]")
        print(f"Mevcut stok     : {girdi['mevcut_stok']} (varsayim)")
        print(f"Promosyon       : {'VAR' if girdi['promo_var_mi'] else 'yok'}")
        print("\n--- Hesaplanan ---")
        for k, v in hesap.items():
            print(f"  {k:20}: {v}")
        print("\n--- Agent karari ---")
        print(json.dumps(karar.model_dump(), ensure_ascii=False, indent=2))
        if mudahaleler:
            print("\n--- Kural motoru mudahaleleri ---")
            for m in mudahaleler:
                print("  *", m)

    return kayit


def toplu_calistir(client, df: pd.DataFrame) -> None:
    print(f"\n########## TOPLU KARAR ({len(df)} gun) ##########")
    satirlar = []
    for i, satir in df.iterrows():
        girdi = satir_to_girdi(satir)
        kayit = tek_karar(client, girdi, yazdir=False)
        k, h = kayit["karar"], kayit["hesaplanan"]
        satirlar.append({
            "tarih": girdi["tarih"],
            "tahmin": girdi["tahmin_satis"],
            "stok": girdi["mevcut_stok"],
            "promo": int(girdi["promo_var_mi"]),
            "belirsizlik": h["belirsizlik_orani"],
            "karar": k["karar"],
            "miktar": k["miktar"],
            "aciliyet": k["aciliyet"],
            "gercek": round(float(satir["gercek"]), 1),
            "gerekce": k["gerekce"],
        })
        print(f"{girdi['tarih']} -> {k['karar']:12} miktar={k['miktar']:6} "
              f"aciliyet={k['aciliyet']:7} belirsizlik={h['belirsizlik_orani']}")

    sonuc = pd.DataFrame(satirlar)
    sonuc.to_csv(TOPLU_CSV, index=False, encoding="utf-8-sig")

    print("\n--- Karar dagilimi ---")
    print(sonuc["karar"].value_counts().to_string())
    print("\n--- Aciliyet dagilimi ---")
    print(sonuc["aciliyet"].value_counts().to_string())
    print(f"\nToplam siparis miktari : {sonuc['miktar'].sum():,}")
    print(f"Insan onayina dusen gun: {(sonuc['karar'] == 'insana_sor').sum()}")
    print(f"\nKaydedildi: {TOPLU_CSV}")


if __name__ == "__main__":
    df = forecast_yukle()
    client = istemci_olustur()

    if "--toplu" in sys.argv:
        toplu_calistir(client, df)
    else:
        if "--tarih" in sys.argv:
            istenen = sys.argv[sys.argv.index("--tarih") + 1]
            eslesen = df[df["ds"] == pd.Timestamp(istenen)]
            if eslesen.empty:
                sys.exit(f"HATA: {istenen} tahmin dosyasinda yok. "
                         f"Aralik: {df['ds'].min().date()} - {df['ds'].max().date()}")
            satir = eslesen.iloc[0]
        else:
            satir = df.iloc[-1]

        kayit = tek_karar(client, satir_to_girdi(satir))
        KARAR_JSON.write_text(
            json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nKarar kaydedildi: {KARAR_JSON}")