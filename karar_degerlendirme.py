"""
karar_degerlendirme.py — Agent karar kalitesinin olculmesi

NEDEN BU SCRIPT VAR
    Routing eval "dogru araci secti mi" sorusunu olcuyor. Bu script farkli
    bir soruyu soruyor: "verdigi KARAR dogru muydu?" Ikisi ayri seydir ve
    ikincisi olculmeden "agent iyi calisiyor" denemez.

REFERANS KARARIN TURETILMESI — dairesellikten kacinma
    Referans, agent'in KENDI esiklerinden turetilmez; oyle yapilsaydi ayni
    kural iki kez uygulanip yapay bir %100 isabet cikardi. Bunun yerine
    referans, simulasyonun GERCEKLESEN SONUCUNDAN alinir:

      - siparis_yok modunda (p50 talep) o gun stok tukendiyse
        -> siparis verilmesi gerekiyordu            -> siparis_ver
      - politika modunda (p50 talep) o gun mal fire verdiyse
        -> elde eriyemeyen fazla stok vardi         -> indirim_uygula
      - ikisi de varsa: EUR kaybi buyuk olan kazanir
      - hicbiri yoksa                               -> bekle

    Bu hala ayni simulasyondan geliyor, dolayisiyla mutlak bir "yer
    gercegi" degil. Ama KURAL degil SONUC temelli oldugu icin agent'in
    esiklerinden bagimsizdir ve gercek bir sinama saglar. README'de bu
    sinir acikca belirtilmelidir.

KAPALI GUNLER
    Magaza kapaliysa (acik == 0) ne siparis ne indirim anlamlidir; bu
    gunler degerlendirme disi birakilir ve ayrica raporlanir.

CALISTIRMA
    python karar_degerlendirme.py                    # 200 magazalik ornek
    python karar_degerlendirme.py --magaza 500       # 500 magaza
    python karar_degerlendirme.py --tum              # 1115 magazanin tamami
    python karar_degerlendirme.py --duyarlilik       # esik duyarlilik analizi
    python karar_degerlendirme.py --smape            # kapali gun duzeltmeli sMAPE
    python karar_degerlendirme.py --kapi             # belirsizlik kapisi dogrulamasi
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

import karar_tool as K
from tools_toplu import (
    ROSSMAN,
    _ozet,
    _projeksiyon,
    _tahmin,
)

CIKTI_DIR = ROSSMAN if ROSSMAN.exists() else Path(".")
VARSAYILAN_MAGAZA_SAYISI = 200


# ===========================================================================
# REFERANS KARAR — simulasyon sonucundan
# ===========================================================================
def referans_tablosu(eski_yontem: bool = False) -> pd.DataFrame:
    """Her magaza-gun icin SONUC temelli referans karari uretir.

    YONTEM DEGISIKLIGI — neden siparis_yok baseline'i birakildi
        Ilk tasarimda referans, "hic siparis verilmeseydi ne olurdu"
        senaryosundan (mod=siparis_yok) turetiliyordu. Sorun: taze
        kategorinin raf omru 3 gun ve bu baseline 28 gun boyunca HIC
        siparis vermiyor. Stok 3-4 gunde tukeniyor, sonraki tum gunler
        otomatik olarak "siparis gerekiyordu" diyor.

        Olculdu: 31.220 magaza-gunun 25.148'i (%80.5) bu sekilde
        DEJENEREydi. Referans bilgi tasimiyordu ve agent'in cogunlukla
        siparis_ver demesi yapay bir isabet uretiyordu.

    YENI REFERANS — politika modunun KENDI sonucundan
        Politika modu her gun siparis verir, dolayisiyla dejenere olmaz.
        O gunun gerceklesen sonucuna bakilir:
          - acilis stok, p90 (kotumser) talebi karsilamiyorsa
            -> o gun siparis gerekiyordu          -> siparis_ver
          - o gun mal fire verdiyse
            -> elde eriyemeyen fazla stok vardi   -> indirim_uygula
          - ikisi de -> EUR kaybi buyuk olan
          - hicbiri  -> bekle

        Bu referans da agent'in ESIKLERINDEN bagimsizdir (kural degil
        SONUC temelli), ama artik TUM gunler icin tanimlidir.

    eski_yontem=True ile onceki referans kiyaslama amaciyla uretilebilir.
    """
    p = _projeksiyon()

    pol = p[(p["mod"] == "politika") & (p["talep_senaryo"] == "p50")][
        ["magaza", "tarih", "acik", "acilis", "fire", "fire_maliyet_eur",
         "stoksuz", "kacan_kar_eur"]
    ].rename(columns={"fire": "pol_fire", "fire_maliyet_eur": "pol_fire_eur",
                      "acilis": "pol_acilis", "stoksuz": "pol_stoksuz",
                      "kacan_kar_eur": "pol_kacan_kar"})

    # p90 (kotumser) talep: o gun stok yetmeme riski
    p90 = p[(p["mod"] == "politika") & (p["talep_senaryo"] == "p90")][
        ["magaza", "tarih", "talep_adet", "stoksuz", "kacan_kar_eur"]
    ].rename(columns={"talep_adet": "talep_p90", "stoksuz": "stoksuz_p90",
                      "kacan_kar_eur": "kacan_kar_p90"})

    r = pol.merge(p90, on=["magaza", "tarih"], how="inner")

    if eski_yontem:
        yok = p[(p["mod"] == "siparis_yok") & (p["talep_senaryo"] == "p50")][
            ["magaza", "tarih", "stoksuz", "kacan_kar_eur", "acilis"]
        ].rename(columns={"stoksuz": "yok_stoksuz",
                          "kacan_kar_eur": "yok_kacan_kar",
                          "acilis": "yok_acilis"})
        r = r.merge(yok, on=["magaza", "tarih"], how="inner")
        r["dejenere"] = r["yok_acilis"] <= 0.5
        r["_tukendi"] = r["yok_stoksuz"] > 0.5
        r["_kacan"] = r["yok_kacan_kar"]
    else:
        # Yeni yontemde dejenere gun YOKTUR; kolon uyumluluk icin tutulur.
        r["dejenere"] = False
        r["_tukendi"] = r["stoksuz_p90"] > 0.5
        r["_kacan"] = r["kacan_kar_p90"]

    def karar(s) -> str:
        tukendi = bool(s._tukendi)
        fire_var = s.pol_fire > 0.5
        if tukendi and fire_var:
            return "siparis_ver" if s._kacan >= s.pol_fire_eur else "indirim_uygula"
        if tukendi:
            return "siparis_ver"
        if fire_var:
            return "indirim_uygula"
        return "bekle"

    r["referans"] = r.apply(karar, axis=1)
    # Riske maruz tutar: o gun yanlis karar verilirse kaybedilen EUR
    r["risk_eur"] = r[["_kacan", "pol_fire_eur"]].max(axis=1)
    return r


# ===========================================================================
# AGENT KARARLARI
# ===========================================================================
def agent_kararlari(magazalar: list[int], tarihler: list[pd.Timestamp],
                    sessiz: bool = False) -> pd.DataFrame:
    """karar_uret'i her magaza-gun icin cagirir. Gercek tool cagrilir;
    kural mantigi burada YENIDEN YAZILMAZ, aksi halde tool degil kopyasi
    olculmus olurdu."""
    satirlar = []
    toplam = len(magazalar) * len(tarihler)
    bas = time.time()
    for i, m in enumerate(magazalar, 1):
        for t in tarihler:
            try:
                k = json.loads(K.karar_uret(int(m), str(t.date())))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(k, dict) or "karar" not in k:
                continue
            satirlar.append({
                "magaza": int(m), "tarih": t,
                "agent": k["karar"],
                "aciliyet": k.get("aciliyet"),
                "miktar": k.get("miktar", 0),
                "indirim_orani": k.get("indirim_orani", 0.0),
                "net_etki_eur": k.get("net_etki_eur", 0.0),
                "onay_gerekli": bool(k.get("onay_gerekli_mi")),
                "magaza_kapali": bool(k.get("magaza_kapali")),
                # kapinin gordugu (val) metrik; test yalniz geriye donuk
                "smape": k.get("model_smape_val", 0.0),
                "kapsama": k.get("model_kapsama_val", 0.0),
                "smape_test": k.get("model_smape_test", 0.0),
                "onerilen_aksiyon": k.get("onerilen_aksiyon", ""),
            })
        if not sessiz and (i % 25 == 0 or i == len(magazalar)):
            gecen = time.time() - bas
            print(f"  {i}/{len(magazalar)} magaza  "
                  f"({len(satirlar)}/{toplam} karar, {gecen:.0f} sn)")
    return pd.DataFrame(satirlar)


# ===========================================================================
# RAPORLAMA
# ===========================================================================
def rapor(d: pd.DataFrame, dejenere_dahil: bool = False) -> None:
    kapali = d[d["magaza_kapali"]]
    acik = d[~d["magaza_kapali"]].copy()
    dej = acik[acik["dejenere"]]
    if not dejenere_dahil:
        acik = acik[~acik["dejenere"]].copy()

    print("\n" + "=" * 72)
    print("KARAR KALITESI RAPORU")
    print("=" * 72)
    print(f"Toplam magaza-gun        : {len(d)}")
    print(f"  Kapali gun (disarida)  : {len(kapali)}")
    if len(dej):
        print(f"  Dejenere gun (disarida): {len(dej)}")
        print("    (eski referans yontemi: baseline'da stok zaten tukenmis,")
        print("     'siparis gerekiyordu' cevabi trivial oldugu icin elendi)")
    print(f"  Degerlendirilen        : {len(acik)}")
    if not len(acik):
        print("\nUYARI: degerlendirilecek gun kalmadi.")
        return

    # --- karar dagilimi ---------------------------------------------------
    print("\n--- Karar dagilimi ---")
    dag = pd.crosstab(acik["agent"], "adet")
    dag["yuzde"] = (dag["adet"] / len(acik) * 100).round(1)
    print(dag.sort_values("adet", ascending=False).to_string())

    # --- confusion matrix -------------------------------------------------
    print("\n--- Confusion matrix (satir: referans, sutun: agent) ---")
    cm = pd.crosstab(acik["referans"], acik["agent"])
    print(cm.to_string())

    isabet = (acik["agent"] == acik["referans"]).mean()
    print(f"\nHam isabet: {isabet:.1%}")
    print("NOT: insana_sor bir KARAR degil, karari devretmedir. Asagidaki")
    print("     'devirli isabet' onu ayri degerlendirir.")

    devredilen = acik["agent"] == "insana_sor"
    karar_verilen = acik[~devredilen]
    if len(karar_verilen):
        print(f"Devredilen                 : {devredilen.mean():.1%}")
        print(f"Karar verilenlerde isabet  : "
              f"{(karar_verilen['agent'] == karar_verilen['referans']).mean():.1%}")

    # --- ekonomik kapsama -------------------------------------------------
    print("\n--- Ekonomik kapsama ---")
    aksiyon_gerek = acik[acik["referans"] != "bekle"]
    toplam_risk = aksiyon_gerek["risk_eur"].sum()

    kacirilan = aksiyon_gerek[aksiyon_gerek["agent"] == "bekle"]
    devredilen_risk = aksiyon_gerek[aksiyon_gerek["agent"] == "insana_sor"]
    yakalanan = aksiyon_gerek[~aksiyon_gerek["agent"].isin(["bekle", "insana_sor"])]

    gereksiz = acik[(acik["referans"] == "bekle")
                    & (~acik["agent"].isin(["bekle", "insana_sor"]))]

    print(f"Aksiyon gereken gun sayisi : {len(aksiyon_gerek)}")
    print(f"Riske maruz tutar          : {toplam_risk:,.0f} EUR")
    print(f"  Agent aksiyon aldi       : {len(yakalanan)} gun / "
          f"{yakalanan['risk_eur'].sum():,.0f} EUR "
          f"({yakalanan['risk_eur'].sum() / max(toplam_risk, 1):.1%})")
    print(f"  Insana devredildi        : {len(devredilen_risk)} gun / "
          f"{devredilen_risk['risk_eur'].sum():,.0f} EUR")
    print(f"  KACIRILDI (bekle dedi)   : {len(kacirilan)} gun / "
          f"{kacirilan['risk_eur'].sum():,.0f} EUR "
          f"({kacirilan['risk_eur'].sum() / max(toplam_risk, 1):.1%})")
    print(f"Gereksiz aksiyon           : {len(gereksiz)} gun")

    # --- onay yuku --------------------------------------------------------
    print("\n--- Insan onayi yuku ---")
    onay = acik["onay_gerekli"].sum()
    print(f"Onaya giden karar          : {onay} / {len(acik)} ({onay/max(len(acik),1):.1%})")
    print("NOT: bu oran cok yuksekse otomasyonun anlami azalir; cok dusukse")
    print("     riskli kararlar denetimsiz gecer. Esik secimi bu dengedir.")

    # --- en buyuk hatalar -------------------------------------------------
    if len(kacirilan):
        print("\n--- En pahali 10 kacirilmis gun ---")
        print(kacirilan.nlargest(10, "risk_eur")[
            ["magaza", "tarih", "referans", "agent", "risk_eur", "smape"]
        ].to_string(index=False))

    yol = CIKTI_DIR / "karar_degerlendirme.csv"
    d.to_csv(yol, index=False)
    print(f"\nAyrinti: {yol}")


# ===========================================================================
# ESIK DUYARLILIGI
# ===========================================================================
def duyarlilik(magazalar, tarihler, ref) -> None:
    print("\n" + "=" * 72)
    print("ESIK DUYARLILIK ANALIZI")
    print("=" * 72)
    print("Her esik degeri icin tum kararlar yeniden uretilir.\n")

    asil = (K.FIRE_ORAN_ESIGI, K.STOK_GUN_ALT_ESIK, K.SMAPE_BELIRSIZ_ESIK)
    sonuc = []

    for fire_esik in (0.20, 0.30, 0.40, 0.60, 0.80):
        K.FIRE_ORAN_ESIGI = fire_esik
        d = agent_kararlari(magazalar, tarihler, sessiz=True).merge(
            ref, on=["magaza", "tarih"], how="inner")
        a = d[(~d["magaza_kapali"]) & (~d["dejenere"])]
        ag = a[a["referans"] != "bekle"]
        kac = ag[ag["agent"] == "bekle"]["risk_eur"].sum()
        dev = ag[ag["agent"] == "insana_sor"]["risk_eur"].sum()
        ger = len(a[(a["referans"] == "bekle")
                    & (~a["agent"].isin(["bekle", "insana_sor"]))])
        sonuc.append({
            "fire_esigi": fire_esik,
            "indirim_karari": int((a["agent"] == "indirim_uygula").sum()),
            "devredilen": int((a["agent"] == "insana_sor").sum()),
            "devredilen_eur": round(dev),
            "kacirilan_eur": round(kac),
            "gereksiz_aksiyon": ger,
            "onay_orani": round(a["onay_gerekli"].mean(), 3),
        })
        print(f"  fire esigi {fire_esik:.2f} tamamlandi")

    K.FIRE_ORAN_ESIGI, K.STOK_GUN_ALT_ESIK, K.SMAPE_BELIRSIZ_ESIK = asil

    t = pd.DataFrame(sonuc)
    print("\n" + t.to_string(index=False))
    yol = CIKTI_DIR / "esik_duyarlilik.csv"
    t.to_csv(yol, index=False)
    print(f"\nKaydedildi: {yol}")
    print("\nYORUM")
    print("Esik dustukce daha cok gun aksiyon kapsamina girer ve kacirilan")
    print("risk azalir. ANCAK kacirilan riskin sifira inmesi riskin ortadan")
    print("kalktigi anlamina GELMEZ: belirsizlik kapisi devredeyse risk")
    print("insan onayina kayar. Bu yuzden devredilen_eur sutunu kacirilan_eur")
    print("ile BIRLIKTE okunmalidir. Tek dogru esik yoktur; secim, otomatik")
    print("aksiyon ile insan denetimi arasindaki takastir.")


# ===========================================================================
# KAPALI GUN DUZELTMELI sMAPE
# ===========================================================================
def smape_duzeltme() -> None:
    print("\n" + "=" * 72)
    print("KAPALI GUN DUZELTMELI MODEL PERFORMANSI")
    print("=" * 72)
    t = _tahmin()
    t = t[t["senaryo"] == "planli"]

    g = t.groupby("magaza").agg(
        gun=("tarih", "count"),
        kapali_gun=("acik", lambda s: int((s == 0).sum())),
        smape_test=("magaza_smape_test", "first"),
        kapsama=("magaza_kapsama_test", "first"),
    ).reset_index()
    g["kapali_oran"] = (g["kapali_gun"] / g["gun"]).round(3)

    print(f"Magaza sayisi                     : {len(g)}")
    print(f"Hic kapali gunu olmayan           : {(g.kapali_gun == 0).sum()}")
    print(f"En az 1 gun kapali                : {(g.kapali_gun > 0).sum()}")
    print(f"Yarisindan fazlasi kapali         : {(g.kapali_oran > 0.5).sum()}")

    zayif = g[g["smape_test"] > K.SMAPE_BELIRSIZ_ESIK]
    print(f"\n'Zayif' sayilan magaza (sMAPE>%{K.SMAPE_BELIRSIZ_ESIK:.0f}) : {len(zayif)}")
    if len(zayif):
        print(f"  Bunlarin kapali gun ortalamasi   : {zayif.kapali_oran.mean():.1%}")
        print(f"  Tum magazalarda kapali ortalama  : {g.kapali_oran.mean():.1%}")
        print("\nEger zayif magazalarda kapali oran belirgin sekilde yuksekse,")
        print("sMAPE modeli degil KAPALI GUNLERI olcuyor demektir. Bu durumda")
        print("belirsizlik kapisi yanlis magazalari eliyor.")
        print("\n--- En yuksek sMAPE'li 10 magaza ---")
        print(g.nlargest(10, "smape_test")[
            ["magaza", "smape_test", "kapsama", "kapali_gun", "kapali_oran"]
        ].to_string(index=False))

    korelasyon = g[["smape_test", "kapali_oran"]].corr().iloc[0, 1]
    print(f"\nsMAPE ile kapali gun orani korelasyonu: {korelasyon:.3f}")

    yol = CIKTI_DIR / "magaza_performans_kapali_duzeltme.csv"
    g.to_csv(yol, index=False)
    print(f"Kaydedildi: {yol}")


# ===========================================================================
# BELIRSIZLIK KAPISI DOGRULAMASI — kapi val ile kurulur, test ile sinanir
# ===========================================================================
def kapi_dogrulama() -> None:
    """Kapi karar aninda bilinen (val) metrikle calisir. Burada, kapinin
    kapattigi magazalarin TEST doneminde gercekten daha kotu olup olmadigi
    olculur. Test bu adimda yalniz SINAMA icin kullanilir; kapiya geri
    beslenmez (esik secimi testle yapilmaz)."""
    print("\n" + "=" * 72)
    print("BELIRSIZLIK KAPISI DOGRULAMASI (val ile kur, test ile sina)")
    print("=" * 72)
    p = K._perf().copy()
    hukum = [K.belirsizlik_kapisi(r.smape_val, r.kapsama_val, r.n_val)
             for r in p.itertuples()]
    p["kapali"] = [h["zayif"] for h in hukum]
    p["eski_kapi_test"] = ((p["smape_test"] > 20) | (p["kapsama_test"] < 70))

    kap, acik = p[p["kapali"]], p[~p["kapali"]]
    print(f"Kapinin kapattigi magaza          : {len(kap)} / {len(p)}")
    print(f"  test sMAPE   kapali / acik       : %{kap.smape_test.mean():.2f} / "
          f"%{acik.smape_test.mean():.2f}")
    print(f"  test kapsama kapali / acik       : %{kap.kapsama_test.mean():.1f} / "
          f"%{acik.kapsama_test.mean():.1f}")
    kotu = p["smape_test"] > p["smape_test"].quantile(0.90)
    yakalanan = int((p["kapali"] & kotu).sum())
    print(f"  testte en kotu %10 ({int(kotu.sum())} magaza)'dan yakalanan: {yakalanan} "
          f"(sansla beklenen ~{kotu.sum() * len(kap) / len(p):.0f})")
    print(f"Eski kapi (test metrigiyle, sizintili) kapattigi: {int(p.eski_kapi_test.sum())}")
    print("Not: eski kapi test donemini GORDUGU icin kiyas adil degildir; "
          "yalniz olcek fikri verir.")

    yol = CIKTI_DIR / "kapi_dogrulama.csv"
    p.to_csv(yol, index=False)
    print(f"Kaydedildi: {yol}")


# ===========================================================================
def main() -> None:
    a = sys.argv

    if "--smape" in a:
        smape_duzeltme()
        return

    if "--kapi" in a:
        kapi_dogrulama()
        return

    ozet = _ozet()
    tum = sorted(ozet["magaza"].unique())
    if "--tum" in a:
        magazalar = tum
    else:
        n = int(a[a.index("--magaza") + 1]) if "--magaza" in a else VARSAYILAN_MAGAZA_SAYISI
        magazalar = tum[:n]

    tarihler = sorted(_tahmin()["tarih"].unique())
    tarihler = [pd.Timestamp(x) for x in tarihler]

    print(f"Magaza: {len(magazalar)}  |  Gun: {len(tarihler)}  |  "
          f"Toplam karar: {len(magazalar) * len(tarihler)}")

    ref = referans_tablosu(eski_yontem="--eski-referans" in a)

    if "--duyarlilik" in a:
        duyarlilik(magazalar, tarihler, ref)
        return

    print("\nKararlar uretiliyor...")
    d = agent_kararlari(magazalar, tarihler).merge(
        ref, on=["magaza", "tarih"], how="inner")
    rapor(d, dejenere_dahil="--dejenere-dahil" in a)


if __name__ == "__main__":
    main()