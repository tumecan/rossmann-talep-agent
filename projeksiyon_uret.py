"""
projeksiyon_uret.py — 28 gunluk FIFO ileri projeksiyon (1115 magaza).

NE YAPAR: stok_baslangic.csv'deki parti yaslarini alir, tahmin tablosundaki
gunluk talebi tuketir, tedarikci sozlesmesine gore siparis verir ve her gun
icin acilis/satis/fire/stoksuz/kapanis uretir.

IKI MOD:
  siparis_yok : hic siparis verilmezse ne olur (taban senaryo)
  politika    : newsvendor hedef + tedarikci kisitlari (lead, teslimat gunleri)

UC TALEP SENARYOSU: p10 / p50 / p90 (kalibre edilmis banttan)
  -> tukenme riski p90 talebe, fire riski p10 talebe gore okunur

ONAY — satinalma_politikasi.md Madde 6 (Rev.2):
  6.1 MAGAZA DUZEYI (risk temelli, magaza muduru):
      kritik stok    : p90 talepte 1 gunde tukenir
      fire riski     : yas2 stok > gunluk talebin %40'i
      belirsiz durum : mevcut stok p10-p90 bandinin ICINDE
      (ardisik indirim kriteri agent katmaninda, burada yok)
  6.2 ZINCIR DUZEYI (tutar temelli, bolge muduru):
      tekil siparis > 4.000 EUR | bir tedarikciye ayni gun toplam > 150.000 EUR
  6.3 Minimum siparis tedarikciye verilen GUNLUK TOPLAMA uygulanir.

  Dort kriter AYRI kolonda tutulur: n8n/Telegram payload'inda "neden onaya
  gitti" bilgisi gorunsun diye. onay_gerekli = bunlarin max'i.

MINIMUM SIPARIS — ZINCIR DUZEYINDE (merkezi satinalma):
  Sozlesme minimumu tedarikcinin o gunku TOPLAM siparisine uygulanir, magaza
  basina degil. Magaza basina uygulamak, min 1500 birimlik Nordmann'i ortalama
  magazanin 9 gunluk talebine esitler; raf omru 3 gun olan bir uründe bu
  yapisal olarak imkansizdir. Tek magazalik simulasyonda kisit BAGLAYICIYDI,
  1115 magazada degil — ayni sozlesme maddesi olcege gore farkli kisit.

FIFO ve raf omru kurallari stok_simulasyon.py ile BIREBIR AYNI:
  gun ici sira: teslimat gelir -> satis (en eski partiden) -> gun sonu yas2 FIRE
  parti teslim gunu dahil 3 gun satilabilir.

Cikti: Rossman/projeksiyon.parquet   | Rossman/projeksiyon_ozet.csv
       Rossman/sevkiyat_plani.csv
"""
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# VARSAYIMLAR — stok_simulasyon.py ile ayni
# ---------------------------------------------------------------------------
BIRIM_FIYAT = 8.0
MARJ_ORANI = 0.25
RAF_OMRU_GUN = 3

BIRIM_KAR = BIRIM_FIYAT * MARJ_ORANI              # Cu = 2.0
BIRIM_MALIYET = BIRIM_FIYAT * (1 - MARJ_ORANI)    # Co = 6.0
KRITIK_ORAN = BIRIM_KAR / (BIRIM_KAR + BIRIM_MALIYET)      # 0.25
Z_KRITIK = NormalDist().inv_cdf(KRITIK_ORAN)               # -0.6745
Z_BAND = NormalDist().inv_cdf(0.90)

# --- Madde 6 Rev.2 onay esikleri ---
ONAY_TUTAR_ESIGI = 4000.0        # EUR, tekil siparis (6.2)
ONAY_GUNLUK_TOPLAM = 500000.0    # EUR, tedarikci x gun toplami (6.2)
FIRE_ESIK_ORAN = 0.40            # yas2 / gunluk talep (6.1)
KRITIK_TUKENME_GUN = 1           # p90 talepte 1 gunde tukenirse (6.1)

TEDARIKCILER = {
    "Schnellware": {"lead": 2, "min": 500, "gunler": {0, 2, 4}},
    "Nordmann":    {"lead": 5, "min": 1500, "gunler": {0, 1, 2, 3, 4}},
    "ExpressLog":  {"lead": 1, "min": 200, "gunler": {0, 1, 2, 3, 4, 5}},
    "Rheinland":   {"lead": 3, "min": 1000, "gunler": {1, 3}},
}

KOK = Path(__file__).resolve().parent
ROSSMAN = KOK / "Rossman"
FIGS = ROSSMAN / "figs"


# ---------------------------------------------------------------------------
def kos(talep, stok0, ted, teslim_gunu, sonraki_teslim, talep_p50, sigma,
        siparis_var):
    """Tek tedarikci grubu icin 28 gunluk FIFO projeksiyon.
    talep: (m, T) | stok0: (m, 3) yas0,yas1,yas2 | doner: dict of (m, T)"""
    m, T = talep.shape
    lead, min_sip = ted["lead"], ted["min"]
    gunluk_ort = np.maximum(talep_p50.mean(axis=1), 1e-6)
    tavan = RAF_OMRU_GUN * gunluk_ort            # raf omru kapasitesi

    parti = stok0.astype(float).copy()
    boru = np.zeros((m, T + 10))
    C = {k: np.zeros((m, T)) for k in
         ("acilis", "teslim", "satis", "stoksuz", "fire", "kapanis",
          "en_eski_yas", "kalan_raf", "siparis")}

    for t in range(T):
        parti[:, 0] += boru[:, t]
        C["teslim"][:, t] = boru[:, t]
        C["acilis"][:, t] = parti.sum(axis=1)

        satis = np.minimum(parti.sum(axis=1), talep[:, t])
        kalan = satis.copy()
        for y in (2, 1, 0):                      # FIFO: en eski once
            al = np.minimum(parti[:, y], kalan)
            parti[:, y] -= al
            kalan -= al
        C["satis"][:, t] = satis
        C["stoksuz"][:, t] = talep[:, t] - satis

        C["fire"][:, t] = parti[:, 2]            # gun sonu: yas2 kalani cope
        parti[:, 2], parti[:, 1], parti[:, 0] = parti[:, 1], parti[:, 0], 0.0

        C["kapanis"][:, t] = parti.sum(axis=1)
        var = parti > 1e-6
        yas = np.where(var[:, 2], 2, np.where(var[:, 1], 1, 0))
        dolu = parti.sum(axis=1) > 1e-6
        C["en_eski_yas"][:, t] = np.where(dolu, yas, 0)
        C["kalan_raf"][:, t] = np.where(dolu, RAF_OMRU_GUN - yas, 0)

        # --- siparis karari (politika modu) ---
        varis = t + lead
        if siparis_var and varis < T and teslim_gunu[varis]:
            son = sonraki_teslim[varis]
            donem = slice(varis, min(son, T))
            beklenen = talep_p50[:, donem].sum(axis=1)
            sig = np.sqrt((sigma[:, donem] ** 2).sum(axis=1))
            hedef = np.maximum(beklenen + Z_KRITIK * sig, 0.0)

            # varis gunune kadar: ara talep DUSULUR, YOLDAKI siparis EKLENIR
            ara = talep_p50[:, t + 1:varis].sum(axis=1) if varis > t + 1 else 0.0
            yolda = boru[:, t + 1:varis + 1].sum(axis=1)
            tahmini_stok = np.maximum(C["kapanis"][:, t] + yolda - ara, 0.0)

            miktar = np.maximum(hedef - tahmini_stok, 0.0)
            miktar = np.minimum(miktar, tavan)           # raf omru kapasitesi
            miktar = np.where(miktar < 1.0, 0.0, miktar)

            # minimum siparis ZINCIR duzeyinde (Madde 6.3)
            if miktar.sum() < min_sip:
                miktar = np.zeros(m)

            C["siparis"][:, t] = miktar
            boru[:, varis] += miktar

    return C


def main() -> None:
    tah_yol = ROSSMAN / "model" / "tahminler_tum_magazalar.parquet"
    stok_yol = ROSSMAN / "stok_baslangic.csv"
    for y in (tah_yol, stok_yol):
        if not y.exists():
            raise SystemExit(f"HATA: {y} yok.")

    tah = pd.read_parquet(tah_yol)
    tah = tah[tah.senaryo == "planli"].sort_values(["magaza", "h"])
    stok = pd.read_csv(stok_yol).sort_values("magaza").reset_index(drop=True)

    magazalar = stok["magaza"].to_numpy()
    T = int(tah.h.max())
    tarihler = np.sort(tah.tarih.unique())
    assert len(tarihler) == T
    assert (np.sort(tah.magaza.unique()) == magazalar).all(), "magaza listesi uyusmuyor!"
    print(f"magaza {len(magazalar)} | ufuk {T} gun | "
          f"{pd.Timestamp(tarihler[0]).date()}..{pd.Timestamp(tarihler[-1]).date()}")

    def mat(kolon):
        return tah.pivot(index="magaza", columns="h", values=kolon) \
                  .loc[magazalar].to_numpy()

    talep = {q: mat(f"{q}_adet") for q in ("p10", "p50", "p90")}
    acik = mat("acik")
    sigma = (talep["p90"] - talep["p10"]) / (2 * Z_BAND)
    stok0 = stok[["yas0_adet", "yas1_adet", "yas2_adet"]].to_numpy()
    dow = pd.DatetimeIndex(tarihler).dayofweek.to_numpy()

    print(f"\n28 gunluk toplam talep (p50): {talep['p50'].sum():,.0f} birim | "
          f"baslangic stogu: {stok0.sum():,.0f} birim")

    takvim = {}
    for ad, t in TEDARIKCILER.items():
        tg = np.array([d in t["gunler"] for d in dow])
        son = np.full(T + 1, T, dtype=int)
        for i in range(T - 1, -1, -1):
            son[i] = i + 1 if (i + 1 < T and tg[i + 1]) else son[i + 1]
        takvim[ad] = (tg, son)

    # -----------------------------------------------------------------
    satirlar = []
    for mod, siparis_var in [("siparis_yok", False), ("politika", True)]:
        for q in ("p10", "p50", "p90"):
            birik = {k: np.zeros((len(magazalar), T)) for k in
                     ("acilis", "teslim", "satis", "stoksuz", "fire", "kapanis",
                      "en_eski_yas", "kalan_raf", "siparis")}
            for ad, t in TEDARIKCILER.items():
                sec = (stok.tedarikci.to_numpy() == ad)
                if not sec.any():
                    continue
                tg, son = takvim[ad]
                C = kos(talep[q][sec], stok0[sec], t, tg, son,
                        talep["p50"][sec], sigma[sec], siparis_var)
                for k in birik:
                    birik[k][sec] = C[k]

            satirlar.append(pd.DataFrame({
                "magaza": np.repeat(magazalar, T),
                "tarih": np.tile(tarihler, len(magazalar)),
                "h": np.tile(np.arange(1, T + 1), len(magazalar)),
                "mod": mod, "talep_senaryo": q,
                "acik": acik.ravel().astype(int),
                "talep_adet": talep[q].ravel(),
                **{k: birik[k].ravel() for k in birik},
            }))
            print(f"  {mod:<12} {q}: fire {birik['fire'].sum():>9,.0f} | "
                  f"stoksuz {birik['stoksuz'].sum():>9,.0f} | "
                  f"siparis {birik['siparis'].sum():>10,.0f}")

    pro = pd.concat(satirlar, ignore_index=True)
    pro["kacan_ciro_eur"] = pro.stoksuz * BIRIM_FIYAT
    pro["kacan_kar_eur"] = pro.stoksuz * BIRIM_KAR
    pro["fire_maliyet_eur"] = pro.fire * BIRIM_MALIYET
    pro["tukendi"] = (pro.stoksuz > 0.5).astype(int)
    for k in ("acilis", "teslim", "satis", "stoksuz", "fire", "kapanis",
              "siparis", "talep_adet"):
        pro[k] = pro[k].round(1)
    pro["en_eski_yas"] = pro.en_eski_yas.astype(int)
    pro["kalan_raf"] = pro.kalan_raf.astype(int)

    # -----------------------------------------------------------------
    def ilk_tukenme(g):
        t = g.loc[g.tukendi == 1, "h"]
        return int(t.min()) if len(t) else 0        # 0 = tukenmedi

    ozet = stok[["magaza", "magaza_tipi", "tedarikci", "tipik_gunluk_talep_adet",
                 "stok_toplam_adet", "kalan_raf_omru_gun", "stok_gun_karsiligi"]].copy()

    for etiket, mod, q in [("taban_p50", "siparis_yok", "p50"),
                           ("taban_p90", "siparis_yok", "p90"),
                           ("politika_p50", "politika", "p50"),
                           ("politika_p10", "politika", "p10"),
                           ("politika_p90", "politika", "p90")]:
        alt = pro[(pro["mod"] == mod) & (pro.talep_senaryo == q)]
        g = alt.groupby("magaza")
        ozet[f"tukenme_gun_{etiket}"] = g.apply(ilk_tukenme, include_groups=False).to_numpy()
        ozet[f"fire_{etiket}"] = g.fire.sum().round(1).to_numpy()
        ozet[f"stoksuz_{etiket}"] = g.stoksuz.sum().round(1).to_numpy()
        ozet[f"kacan_kar_{etiket}"] = g.kacan_kar_eur.sum().round(1).to_numpy()

    pol = pro[(pro["mod"] == "politika") & (pro.talep_senaryo == "p50")]
    ozet["siparis_toplam"] = pol.groupby("magaza").siparis.sum().round(1).to_numpy()
    ozet["siparis_sayisi"] = pol[pol.siparis > 0].groupby("magaza").size() \
                                .reindex(magazalar, fill_value=0).to_numpy()
    ozet["net_kazanc_eur"] = (
        (ozet.kacan_kar_taban_p50 - ozet.kacan_kar_politika_p50)
        - (ozet.fire_politika_p50 - ozet.fire_taban_p50) * BIRIM_MALIYET).round(1)

    tg = ozet.tukenme_gun_taban_p90.to_numpy()
    ozet["aciliyet"] = np.select(
        [tg == 0, tg <= 1, tg <= 3, tg <= 7],
        ["yok", "kritik", "yuksek", "normal"], default="dusuk")

    # -----------------------------------------------------------------
    # MADDE 6 — ONAY KRITERLERI (dort kolon ayri tutulur, izlenebilirlik icin)
    # -----------------------------------------------------------------
    # 6.1-a kritik stok: p90 talepte 1 gunde tukenir
    ozet["onay_kritik_stok"] = ((tg > 0) & (tg <= KRITIK_TUKENME_GUN)).astype(int)

    # 6.1-b yuksek fire riski: yas2 stok > gunluk talebin %40'i
    gunluk = np.maximum(stok["tipik_gunluk_talep_adet"].to_numpy(), 1e-6)
    ozet["onay_fire_riski"] = (
        stok["yas2_adet"].to_numpy() > FIRE_ESIK_ORAN * gunluk).astype(int)

    # 6.1-c belirsiz durum: mevcut stok, ilk gunun p10-p90 bandinin ICINDE
    stok_top = stok["stok_toplam_adet"].to_numpy()
    ozet["onay_belirsiz"] = ((stok_top >= talep["p10"][:, 0]) &
                             (stok_top <= talep["p90"][:, 0])).astype(int)

    # 6.2 yuksek tutar: magazanin en buyuk tekil siparisi > 4.000 EUR
    max_sip = pol.groupby("magaza").siparis.max().reindex(magazalar, fill_value=0).to_numpy()
    ozet["max_siparis_tutar_eur"] = (max_sip * BIRIM_MALIYET).round(1)
    ozet["onay_yuksek_tutar"] = (ozet.max_siparis_tutar_eur > ONAY_TUTAR_ESIGI).astype(int)

    ONAY_KOL = ["onay_kritik_stok", "onay_fire_riski", "onay_belirsiz",
                "onay_yuksek_tutar"]
    ozet["onay_gerekli"] = ozet[ONAY_KOL].max(axis=1)
    ozet["onay_sebep"] = ozet[ONAY_KOL].apply(
        lambda r: "|".join(k.replace("onay_", "") for k in ONAY_KOL if r[k] == 1) or "yok",
        axis=1)

    # -----------------------------------------------------------------
    sev = pol[pol.siparis > 0][["magaza", "tarih", "h", "siparis"]].copy()
    sev = sev.merge(stok[["magaza", "tedarikci"]], on="magaza")
    sev["lead_gun"] = sev.tedarikci.map(lambda a: TEDARIKCILER[a]["lead"])
    sev["varis_tarihi"] = sev.tarih + pd.to_timedelta(sev.lead_gun, unit="D")
    sev["tutar_eur"] = (sev.siparis * BIRIM_MALIYET).round(1)

    # Madde 6.2 — tekil ve zincir duzeyi
    sev["onay_tekil"] = (sev.tutar_eur > ONAY_TUTAR_ESIGI).astype(int)
    gunluk_toplam = sev.groupby(["tedarikci", "tarih"]).tutar_eur.transform("sum")
    sev["tedarikci_gun_tutar_eur"] = gunluk_toplam.round(1)
    sev["onay_zincir"] = (gunluk_toplam > ONAY_GUNLUK_TOPLAM).astype(int)
    sev["onay_gerekli"] = sev[["onay_tekil", "onay_zincir"]].max(axis=1)
    sev = sev.sort_values(["tarih", "siparis"], ascending=[True, False])

    # -----------------------------------------------------------------
    print("\n" + "=" * 62); print("TABAN SENARYO — hic siparis verilmezse"); print("=" * 62)
    for q in ("p50", "p90"):
        t = ozet[f"tukenme_gun_taban_{q}"]
        print(f"{q}: {int((t > 0).sum())} magaza tukeniyor | medyan {int(t[t > 0].median())}. gun "
              f"| kacan kar {ozet[f'kacan_kar_taban_{q}'].sum():,.0f} EUR")
    print(f"fire (p50): {ozet.fire_taban_p50.sum():,.0f} birim")

    print("\n" + "=" * 62); print("POLITIKA — newsvendor + sozlesme kisitlari"); print("=" * 62)
    print(f"siparis    : {ozet.siparis_toplam.sum():,.0f} birim  "
          f"(28 gunluk talep {talep['p50'].sum():,.0f}) | {len(sev)} sevkiyat")
    print(f"stoksuz    : {ozet.stoksuz_politika_p50.sum():,.0f} birim "
          f"(taban {ozet.stoksuz_taban_p50.sum():,.0f})")
    print(f"fire       : {ozet.fire_politika_p50.sum():,.0f} birim "
          f"= siparisin %{100*ozet.fire_politika_p50.sum()/max(ozet.siparis_toplam.sum(),1):.1f}'i")
    print(f"  p10 talepte {ozet.fire_politika_p10.sum():,.0f} | "
          f"p90 talepte {ozet.fire_politika_p90.sum():,.0f}  <- belirsizligin karar maliyeti")
    print(f"kacan kar  : {ozet.kacan_kar_politika_p50.sum():,.0f} EUR "
          f"(taban {ozet.kacan_kar_taban_p50.sum():,.0f})")
    print(f"NET KAZANC : {ozet.net_kazanc_eur.sum():,.0f} EUR  (kurtarilan kar - artan fire)")

    print("\n" + "=" * 62)
    print("MADDE 6 — ONAY DAGILIMI (magaza duzeyi, 6.1 + 6.2)")
    print("=" * 62)
    n = len(ozet)
    for k in ONAY_KOL:
        a = int(ozet[k].sum())
        print(f"  {k:<22}: {a:>5} magaza  (%{100*a/n:.1f})")
    ong = int(ozet.onay_gerekli.sum())
    print(f"  {'ONAY GEREKLI (max)':<22}: {ong:>5} magaza  (%{100*ong/n:.1f})")
    print("\n  sebep kirilimi (ilk 10):")
    print(ozet.onay_sebep.value_counts().head(10).to_string())
    print(f"\nzincir duzeyi (6.2): sevkiyat {len(sev)} | "
          f"onay_tekil {int(sev.onay_tekil.sum())} | "
          f"onay_zincir {int(sev.onay_zincir.sum())} | "
          f"toplam onaya giden {int(sev.onay_gerekli.sum())}")
    tgt = sev.groupby(["tedarikci", "tarih"]).tutar_eur.sum()
    print(f"tedarikci x gun tutar: medyan {tgt.median():,.0f} EUR | "
          f"max {tgt.max():,.0f} EUR | esigi asan {int((tgt > ONAY_GUNLUK_TOPLAM).sum())} gun")

    print("\naciliyet dagilimi:"); print(ozet.aciliyet.value_counts().to_string())

    print("\ntedarikci performansi (politika, p50):")
    print(ozet.groupby("tedarikci").agg(
        magaza=("magaza", "size"),
        tukenen=("tukenme_gun_politika_p50", lambda s: int((s > 0).sum())),
        fire=("fire_politika_p50", "sum"),
        stoksuz=("stoksuz_politika_p50", "sum"),
        siparis=("siparis_toplam", "sum"),
        onay=("onay_gerekli", "sum"),
        net_kazanc=("net_kazanc_eur", "sum")).round(0).to_string())

    print("\nEN ACIL 10 MAGAZA (p90 talepte en erken tukenen):")
    acil = ozet[ozet.tukenme_gun_taban_p90 > 0].nsmallest(
        10, ["tukenme_gun_taban_p90", "stok_gun_karsiligi"])
    print(acil[["magaza", "tedarikci", "stok_toplam_adet", "tipik_gunluk_talep_adet",
                "tukenme_gun_taban_p90", "aciliyet", "siparis_toplam",
                "onay_sebep"]].to_string(index=False))

    print("\nSEVKIYAT PLANI (ilk 10 satir):")
    print(sev.head(10)[["tarih", "magaza", "tedarikci", "siparis", "varis_tarihi",
                        "tutar_eur", "onay_tekil", "onay_zincir"]].to_string(index=False))

    # -----------------------------------------------------------------
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    gun = np.arange(1, T + 1)
    for mod, stil in [("siparis_yok", "--"), ("politika", "-")]:
        s = pro[(pro["mod"] == mod) & (pro.talep_senaryo == "p50")].groupby("h")
        ax[0].plot(gun, s.stoksuz.sum(), stil, label=mod)
        ax[1].plot(gun, s.fire.sum(), stil, label=mod)
    ax[0].set_xlabel("gun"); ax[0].set_ylabel("adet"); ax[0].set_title("Stoksuz kalan")
    ax[1].set_xlabel("gun"); ax[1].set_ylabel("adet"); ax[1].set_title("Fire")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    t90 = ozet.tukenme_gun_taban_p90
    ax[2].hist(t90[t90 > 0], bins=np.arange(0.5, T + 1.5))
    ax[2].set_xlabel("ilk tukenme gunu (p90, siparissiz)"); ax[2].set_ylabel("magaza")
    ax[2].set_title("Siparis verilmezse tukenme")
    say = [int(ozet[k].sum()) for k in ONAY_KOL] + [int(ozet.onay_gerekli.sum())]
    etk = [k.replace("onay_", "") for k in ONAY_KOL] + ["TOPLAM"]
    ax[3].barh(etk, say, color=["#4c72b0"] * 4 + ["#c44e52"])
    ax[3].set_xlabel("magaza"); ax[3].set_title("Madde 6 — onay kriterleri")
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIGS / "31_projeksiyon.png", dpi=130)

    pro.to_parquet(ROSSMAN / "projeksiyon.parquet", index=False)
    ozet.to_csv(ROSSMAN / "projeksiyon_ozet.csv", index=False, encoding="utf-8")
    sev.to_csv(ROSSMAN / "sevkiyat_plani.csv", index=False, encoding="utf-8")
    print(f"\nkaydedildi -> projeksiyon.parquet ({len(pro):,}) | "
          f"projeksiyon_ozet.csv ({len(ozet)}) | sevkiyat_plani.csv ({len(sev)})")
    print("grafik     -> figs/31_projeksiyon.png")


if __name__ == "__main__":
    main()