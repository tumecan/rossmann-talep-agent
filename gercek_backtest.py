"""
gercek_backtest.py — Karar zincirinin GERCEKLESEN SATISLA degerlendirilmesi

NEDEN
    Eski karar_degerlendirme referansi simulasyondan (p90 senaryosu)
    geliyordu: model tahmini, yine model tahminiyle sinaniyordu. Ustelik
    siparis karari "bugun stok tukendi mi" ile olculuyordu; oysa bugun
    verilen siparis bugunu degil varis gununu kurtarir.

    Rossmann train.csv 2015-07-31'e kadar GERCEK satis icerir; tahmin
    penceresi (07-04..07-31) bunun icindedir. Burada:
      - siparisler MODEL TAHMINIYLE verilir (karar aninda bilinen),
      - tuketim GERCEK TALEPTEN olur (sonradan gerceklesen).
    Bu, karar zincirinin gercek bir geriye donuk testidir; referans
    agent'in kurallarindan da modelden de bagimsizdir.

UC BOLUM
    1) POLITIKA BACKTEST — siparis kararinin degeri, EUR cinsinden:
         siparis_yok      : hic siparis verilmezse (alt sinir)
         model            : siparisler model tahminiyle (sistemin kendisi)
         kusursuz_bilgi   : siparisler gercek talebi bilerek (ust sinir)
         plan             : sistemin KENDI beklentisi (tahmin = gercek varsayimi)
       Kayip = stoksuz x birim kar + fire x birim maliyet (newsvendor maliyeti).
       "Model hatasinin bedeli" = kayip(model) - kayip(kusursuz_bilgi).
    2) GUN BAZLI KARARLAR — karar_uret'in ex-ante sinyalleri gerceklesenle:
         indirim / fire      : agent fire dedi mi, gercekte fire oldu mu
         stok uyarisi        : stoksuz_p90_bugun > 0 dedi mi, gercekte tukendi mi
    3) FIRE ESIGI DUYARLILIGI — gerceklesen fireye gore

CALISTIRMA
    python gercek_backtest.py                  # train.csv otomatik aranir
    python gercek_backtest.py --train D:\\veri\\train.csv
    python gercek_backtest.py --yeniden        # karar onbellegini yeniden uret
    python gercek_backtest.py --sadece-politika # karar_uret cagrilmadan hizli
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import karar_tool as K
from projeksiyon_uret import (BIRIM_KAR, BIRIM_MALIYET, TEDARIKCILER, Z_BAND,
                              kos)
from tools_toplu import ROSSMAN, _tahmin

KOK = Path(__file__).resolve().parent
CIKTI = ROSSMAN if Path(ROSSMAN).exists() else KOK
KARAR_ONBELLEK = Path(CIKTI) / "karar_onbellek.csv"


# ===========================================================================
# VERI
# ===========================================================================
def train_bul() -> Path:
    a = sys.argv
    if "--train" in a:
        return Path(a[a.index("--train") + 1])
    adaylar = [Path(ROSSMAN) / "train.csv", Path(ROSSMAN) / "data" / "train.csv",
               Path(ROSSMAN) / "veri" / "train.csv", KOK / "train.csv",
               KOK / "data" / "train.csv"]
    for y in adaylar:
        if y.exists():
            return y
    for y in KOK.rglob("train.csv"):
        return y
    raise SystemExit("train.csv bulunamadi. --train <yol> ile ver.")


def veri_hazirla() -> dict:
    tah = _tahmin()
    tah = tah[tah["senaryo"] == "planli"].copy()
    tah["tarih"] = pd.to_datetime(tah["tarih"])
    stok = pd.read_csv(Path(ROSSMAN) / "stok_baslangic.csv") \
             .sort_values("magaza").reset_index(drop=True)
    magazalar = stok["magaza"].to_numpy()
    tarihler = pd.DatetimeIndex(np.sort(tah["tarih"].unique()))
    T = len(tarihler)

    def mat(kolon):
        return tah.pivot(index="magaza", columns="tarih", values=kolon) \
                  .reindex(index=magazalar, columns=tarihler).fillna(0).to_numpy()

    p10, p50, p90 = mat("p10_adet"), mat("p50_adet"), mat("p90_adet")
    sigma = (p90 - p10) / (2 * Z_BAND)

    # EUR -> adet: tahmin tablosunun KENDI donusum orani (TAZE_PAY / BIRIM_FIYAT).
    # Sabit burada yeniden yazilmaz; tahmin hangi oranla uretildiyse o kullanilir.
    pozitif = tah["p50_eur"] > 0
    oran = float((tah.loc[pozitif, "p50_adet"] / tah.loc[pozitif, "p50_eur"]).median())

    yol = train_bul()
    tr = pd.read_csv(yol, usecols=["Store", "Date", "Sales"],
                     parse_dates=["Date"], low_memory=False)
    tr = tr[tr["Date"].isin(tarihler)]
    gercek_eur = tr.pivot_table(index="Store", columns="Date", values="Sales",
                                aggfunc="sum") \
                   .reindex(index=magazalar, columns=tarihler)
    eksik = int(gercek_eur.isna().to_numpy().sum())
    gercek = gercek_eur.fillna(0).to_numpy() * oran

    print(f"train.csv          : {yol}")
    print(f"pencere            : {tarihler[0].date()} .. {tarihler[-1].date()} ({T} gun)")
    print(f"EUR->adet orani    : {oran:.5f}  (tahmin tablosundan)")
    print(f"eksik gercek hucre : {eksik} (0 kabul edildi)")
    print(f"toplam talep       : tahmin p50 {p50.sum():,.0f} | gercek {gercek.sum():,.0f} adet "
          f"(bias %{(p50.sum() / max(gercek.sum(), 1) - 1) * 100:+.2f})")

    dow = tarihler.dayofweek.to_numpy()
    takvim = {}
    for ad, t in TEDARIKCILER.items():
        tg = np.array([d in t["gunler"] for d in dow])
        son = np.full(T + 1, T, dtype=int)
        for i in range(T - 1, -1, -1):
            son[i] = i + 1 if (i + 1 < T and tg[i + 1]) else son[i + 1]
        takvim[ad] = (tg, son)

    return dict(magazalar=magazalar, tarihler=tarihler, stok=stok, p50=p50,
                sigma=sigma, gercek=gercek, takvim=takvim)


def calistir(v: dict, talep, talep_plan, sigma, siparis_var: bool) -> dict:
    """projeksiyon_uret.kos'u tedarikci gruplarina uygular (kod kopyalanmaz)."""
    n, T = talep.shape
    out = {k: np.zeros((n, T)) for k in ("satis", "stoksuz", "fire", "siparis", "acilis")}
    stok0 = v["stok"][["yas0_adet", "yas1_adet", "yas2_adet"]].to_numpy()
    ted = v["stok"]["tedarikci"].to_numpy()
    for ad, t in TEDARIKCILER.items():
        sec = ted == ad
        if not sec.any():
            continue
        tg, son = v["takvim"][ad]
        C = kos(talep[sec], stok0[sec], t, tg, son, talep_plan[sec], sigma[sec], siparis_var)
        for k in out:
            out[k][sec] = C[k]
    return out


def ozetle(ad: str, talep, C) -> dict:
    kacan = C["stoksuz"].sum() * BIRIM_KAR
    fire_m = C["fire"].sum() * BIRIM_MALIYET
    return {"senaryo": ad,
            "talep_adet": round(talep.sum()),
            "satis_adet": round(C["satis"].sum()),
            "doluluk_%": round(C["satis"].sum() / max(talep.sum(), 1) * 100, 2),
            "stoksuz_adet": round(C["stoksuz"].sum()),
            "fire_adet": round(C["fire"].sum()),
            "siparis_adet": round(C["siparis"].sum()),
            "kacan_kar_eur": round(kacan),
            "fire_maliyet_eur": round(fire_m),
            "toplam_kayip_eur": round(kacan + fire_m)}


# ===========================================================================
# 1) POLITIKA BACKTEST
# ===========================================================================
def politika_backtest(v: dict) -> tuple[pd.DataFrame, dict, dict]:
    print("\n" + "=" * 78)
    print("1) POLITIKA BACKTEST — siparisler tahminle, tuketim gercek talepten")
    print("=" * 78)
    g, p50, sig = v["gercek"], v["p50"], v["sigma"]
    kosular = {
        "siparis_yok": calistir(v, g, p50, sig, False),
        "model": calistir(v, g, p50, sig, True),
        "kusursuz_bilgi": calistir(v, g, g, np.zeros_like(g), True),
        "plan": calistir(v, p50, p50, sig, True),
    }
    satir = [ozetle(ad, p50 if ad == "plan" else g, C) for ad, C in kosular.items()]
    t = pd.DataFrame(satir)
    print(t.to_string(index=False))

    k = {r["senaryo"]: r["toplam_kayip_eur"] for r in satir}
    yakalanan = k["siparis_yok"] - k["model"]
    ulasilabilir = k["siparis_yok"] - k["kusursuz_bilgi"]
    ozet = {
        "model_hatasinin_bedeli_eur": k["model"] - k["kusursuz_bilgi"],
        "siparis_politikasinin_degeri_eur": yakalanan,
        "ulasilabilir_degerin_yakalanan_orani_%":
            round(yakalanan / ulasilabilir * 100, 1) if ulasilabilir else None,
        "plan_vs_gercek_kayip_farki_eur": k["model"] - k["plan"],
    }
    print(f"\nModel hatasinin bedeli (model - kusursuz)     : "
          f"{ozet['model_hatasinin_bedeli_eur']:,} EUR")
    print(f"Siparis politikasinin degeri (yok - model)    : {yakalanan:,} EUR")
    print(f"Ulasilabilir degerin yakalanan orani          : "
          f"%{ozet['ulasilabilir_degerin_yakalanan_orani_%']}")
    print(f"Plan iyimserligi (gercek kayip - planlanan)   : "
          f"{ozet['plan_vs_gercek_kayip_farki_eur']:,} EUR")

    # --- nerede? hafta ve belirsizlik kapisi kirilimi ---------------------
    M, KB = kosular["model"], kosular["kusursuz_bilgi"]
    kayip = lambda C: C["stoksuz"] * BIRIM_KAR + C["fire"] * BIRIM_MALIYET
    fark = kayip(M) - kayip(KB)                        # magaza x gun
    hafta = np.arange(len(v["tarihler"])) // 7 + 1
    hk = pd.DataFrame({"hafta": hafta, "model_hatasi_eur": fark.sum(axis=0),
                       "gercek_talep": g.sum(axis=0)}).groupby("hafta").sum()
    hk["eur_per_100_adet"] = (hk["model_hatasi_eur"] / hk["gercek_talep"] * 100).round(2)
    print("\nModel hatasinin bedeli — haftaya gore:")
    print(hk.round(0).to_string())

    perf = K._perf()
    zayif = {int(r.magaza): K.belirsizlik_kapisi(r.smape_val, r.kapsama_val, r.n_val)["zayif"]
             for r in perf.itertuples()}
    z = np.array([zayif.get(int(m), False) for m in v["magazalar"]])
    kk = pd.DataFrame({"kapi": np.where(z, "kapali (zayif)", "acik"),
                       "model_hatasi_eur": fark.sum(axis=1), "talep": g.sum(axis=1)}) \
           .groupby("kapi").sum()
    kk["eur_per_100_adet"] = (kk["model_hatasi_eur"] / kk["talep"] * 100).round(2)
    print("\nModel hatasinin bedeli — belirsizlik kapisina gore:")
    print(kk.round(0).to_string())
    print("Kapi dogru magazalari seciyorsa 'kapali' satirinda birim bedel yuksek olmali.")

    ozet["hafta"] = hk.reset_index().to_dict("records")
    ozet["kapi"] = kk.reset_index().to_dict("records")
    t.to_csv(Path(CIKTI) / "backtest_politika.csv", index=False)
    return t, ozet, kosular


# ===========================================================================
# 2) GUN BAZLI KARARLAR
# ===========================================================================
def kararlari_al(v: dict) -> pd.DataFrame:
    if KARAR_ONBELLEK.exists() and "--yeniden" not in sys.argv:
        print(f"\nKararlar onbellekten: {KARAR_ONBELLEK}")
        return pd.read_csv(KARAR_ONBELLEK, parse_dates=["tarih"])
    alanlar = ("karar", "onerilen_aksiyon", "fire_gun_karsiligi", "fire_riski_birim",
               "stoksuz_p90_bugun", "aciliyet", "onay_gerekli_mi", "magaza_kapali",
               "miktar", "indirim_orani")
    satir, bas = [], time.time()
    toplam = len(v["magazalar"]) * len(v["tarihler"])
    for i, m in enumerate(v["magazalar"]):
        for t in v["tarihler"]:
            k = json.loads(K.karar_uret(int(m), str(t.date())))
            if "karar" not in k:
                continue
            satir.append({"magaza": int(m), "tarih": t, **{a: k.get(a) for a in alanlar}})
        if (i + 1) % 100 == 0:
            gecen = time.time() - bas
            print(f"  {(i + 1) * len(v['tarihler']):>6}/{toplam} karar "
                  f"({gecen / 60:.1f} dk, kalan ~{gecen / (i + 1) * (len(v['magazalar']) - i - 1) / 60:.1f} dk)")
    d = pd.DataFrame(satir)
    d.to_csv(KARAR_ONBELLEK, index=False)
    return d


def gun_bazli(v: dict, kosular: dict) -> dict:
    print("\n" + "=" * 78)
    print("2) GUN BAZLI KARARLAR — ex-ante sinyal vs gerceklesen")
    print("=" * 78)
    d = kararlari_al(v)
    M = kosular["model"]
    idx_m = {int(m): i for i, m in enumerate(v["magazalar"])}
    idx_t = {t: j for j, t in enumerate(v["tarihler"])}
    i = d["magaza"].map(idx_m).to_numpy()
    j = d["tarih"].map(idx_t).to_numpy()
    d["fire_gercek"] = M["fire"][i, j]
    d["stoksuz_gercek"] = M["stoksuz"][i, j]
    d = d[~d["magaza_kapali"].astype(bool)].copy()

    ozet = {"acik_gun": len(d)}
    dag = (d["karar"].value_counts(normalize=True) * 100).round(1)
    print("Karar dagilimi (acik gunler):")
    print(dag.to_string())
    ozet["karar_dagilimi_%"] = dag.to_dict()
    ozet["onay_yuku_%"] = round(d["onay_gerekli_mi"].astype(bool).mean() * 100, 1)
    print(f"Onay yuku: %{ozet['onay_yuku_%']}")

    # --- fire / indirim ---------------------------------------------------
    indirim_dedi = (d["karar"] == "indirim_uygula") | \
                   d["onerilen_aksiyon"].fillna("").str.contains("indirim")
    fire_oldu = d["fire_gercek"] > 0.5
    eur = d["fire_gercek"] * BIRIM_MALIYET
    f = {
        "gercek_fire_gun": int(fire_oldu.sum()),
        "gercek_fire_eur": round(float(eur[fire_oldu].sum())),
        "yakalanan_gun": int((indirim_dedi & fire_oldu).sum()),
        "yakalanan_eur": round(float(eur[indirim_dedi & fire_oldu].sum())),
        "kacirilan_gun": int((~indirim_dedi & fire_oldu).sum()),
        "kacirilan_eur": round(float(eur[~indirim_dedi & fire_oldu].sum())),
        "gereksiz_indirim_gun": int((indirim_dedi & ~fire_oldu).sum()),
    }
    print("\nFire / indirim (gercek fire = model politikasiyla, gercek talepte):")
    for a, b in f.items():
        print(f"  {a:<22}: {b:,}")
    ozet["fire"] = f

    # --- stok uyarisi -----------------------------------------------------
    uyari = d["stoksuz_p90_bugun"].fillna(0) > 0.5
    tukendi = d["stoksuz_gercek"] > 0.5
    keur = d["stoksuz_gercek"] * BIRIM_KAR
    s = {
        "gercek_tukenme_gun": int(tukendi.sum()),
        "gercek_kacan_kar_eur": round(float(keur[tukendi].sum())),
        "uyari_gun": int(uyari.sum()),
        "isabet_precision_%": round(float((uyari & tukendi).sum() / max(uyari.sum(), 1) * 100), 1),
        "yakalama_recall_%": round(float((uyari & tukendi).sum() / max(tukendi.sum(), 1) * 100), 1),
        "uyarili_kacan_kar_eur": round(float(keur[uyari & tukendi].sum())),
    }
    print("\nStok uyarisi (p90'da bugun eksik kalir) vs gercek tukenme:")
    for a, b in s.items():
        print(f"  {a:<22}: {b:,}")
    ozet["stok_uyarisi"] = s

    # --- en pahali kacirmalar --------------------------------------------
    kac = d[~indirim_dedi & fire_oldu].assign(fire_eur=eur).nlargest(10, "fire_eur")
    if len(kac):
        print("\nEn pahali 10 kacirilan fire:")
        kac = kac.assign(tarih=kac["tarih"].dt.date)
        print(kac[["magaza", "tarih", "karar", "fire_gun_karsiligi", "fire_gercek",
                   "fire_eur"]].round(2).to_string(index=False))

    d.to_csv(Path(CIKTI) / "backtest_kararlar.csv", index=False)
    return ozet, d


# ===========================================================================
# 3) FIRE ESIGI DUYARLILIGI
# ===========================================================================
def duyarlilik(d: pd.DataFrame) -> list:
    print("\n" + "=" * 78)
    print("3) FIRE ESIGI DUYARLILIGI — gerceklesen fireye gore")
    print("=" * 78)
    print("Yaklasim: indirim sinyali = fire_gun_karsiligi > esik (kapi/celiski")
    print("etkisi haric tutulur; esigin kendi etkisini olcer).")
    fire_oldu = d["fire_gercek"] > 0.5
    eur = d["fire_gercek"] * BIRIM_MALIYET
    sonuc = []
    for e in (0.20, 0.30, 0.40, 0.60, 0.80):
        sin = d["fire_gun_karsiligi"].fillna(0) > e
        sonuc.append({"fire_esigi": e, "indirim_sinyali": int(sin.sum()),
                      "yakalanan_eur": round(float(eur[sin & fire_oldu].sum())),
                      "kacirilan_eur": round(float(eur[~sin & fire_oldu].sum())),
                      "gereksiz_gun": int((sin & ~fire_oldu).sum())})
    t = pd.DataFrame(sonuc)
    print(t.to_string(index=False))
    t.to_csv(Path(CIKTI) / "backtest_esik_duyarlilik.csv", index=False)
    return sonuc


# ===========================================================================
def main() -> None:
    v = veri_hazirla()
    _, ozet, kosular = politika_backtest(v)
    rapor = {"politika": ozet}
    if "--sadece-politika" not in sys.argv:
        g, d = gun_bazli(v, kosular)
        rapor["gun_bazli"] = g
        rapor["esik_duyarlilik"] = duyarlilik(d)
    yol = Path(CIKTI) / "gercek_backtest.json"
    yol.write_text(json.dumps(rapor, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nOzet: {yol}")


if __name__ == "__main__":
    main()
