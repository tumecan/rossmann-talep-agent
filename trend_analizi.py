"""
trend_analizi.py — Gecmis kayitlardan trend ve drift analizi

NEDEN BU SCRIPT VAR
    model_validasyon.py ve gercek_backtest.py tum pencereyi TEK SAYIYA
    indirger. Tek sayi "model iyi mi" sorusunu cevaplar ama "model ZAMANLA
    kotulesiyor mu" sorusunu cevaplayamaz. Uretimde asil risk ikincisidir:
    bir model ilk gun iyi olup dort hafta sonra sessizce bozulabilir
    (drift). Kalibrasyon calismasinda bunun ilk isareti goruldu: p90 ustu
    ihlal ilk iki haftada ~%11 iken son iki haftada %15.9.

IKI ZAMAN EKSENI
    1. IS ZAMANI  (2015-07-04 - 2015-07-31): model ve karar trendi.
       Gercek satisla olculur, 4 hafta x 1115 magaza. Asil analiz budur.
    2. GERCEK ZAMAN (onay_kayitlari.jsonl): insan onay/red trendi.
       Sistem calistikca birikir; az gun varsa "yetersiz" diye isaretlenir.

UC TREND + ALARM
    1. Model drift   : hafta bazli sMAPE, kapsama, p10/p90 ihlali, bias, Kupiec
    2. Karar trendi  : hafta bazli karar dagilimi, onay yuku, gerceklesen
                       fire yakalama, kacirilan EUR, model hatasinin bedeli
    3. Insan trendi  : gun bazli onay orani ve red sebepleri
    4. Alarm         : SON hafta, ONCEKI haftalarin ortalamasindan esik
                       kadar saparsa alarm uretilir.

    Alarm esikleri DEGISTIRMEZ, karar URETMEZ. Yalnizca insanin dikkatini
    cekmek icindir — ogrenme_dongusu.py ile ayni ilke.

KARAR TRENDI KAYNAGI
    gercek_backtest.py ciktisi (backtest_kararlar.csv + gercek_backtest.json).
    Kararlar karar_tool ile uretildi; sonuc, siparisler tahminle verilip
    tuketim GERCEK satistan yapilarak simule edildi. Eski simulasyon
    referansi (p90 senaryosu) birakildi: modeli yine modelle sinuyordu.

Calistirma:
    python trend_analizi.py
    python trend_analizi.py --grafik           # trend_grafik.png uretir
    python trend_analizi.py --ham "C:/yol/train.csv"
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tools_toplu import ROSSMAN, _pencere

CIKTI_DIR = ROSSMAN if ROSSMAN.exists() else Path(".")
RAPOR = CIKTI_DIR / "trend_raporu.json"
KARAR_CSV = CIKTI_DIR / "backtest_kararlar.csv"
BACKTEST_JSON = CIKTI_DIR / "gercek_backtest.json"
GRAFIK = CIKTI_DIR / "trend_grafik.png"

# Alarm esikleri. Son hafta, ONCEKI haftalarin ortalamasiyla kiyaslanir.
ESIK = {
    "smape_goreli_artis": 0.20,     # sMAPE %20 goreli kotulesirse
    "kapsama_dusus_puan": 3.0,      # kapsama 3 puan duserse
    "kapsama_mutlak_alt": 75.0,     # veya %75'in altina inerse
    "p90_ustu_artis_puan": 3.0,     # ust kuyruk ihlali 3 puan artarsa
    "bias_mutlak": 5.0,             # |bias| %5'i gecerse
    "onay_yuku_artis_puan": 10.0,   # onaya giden oran 10 puan artarsa
    "kacirilan_kat": 2.0,           # kacirilan EUR 2 katina cikarsa
    "model_hatasi_kat": 1.5,        # 100 adet basina model hatasi 1.5 katina cikarsa
}
KARARLAR = ["siparis_ver", "indirim_uygula", "bekle", "insana_sor"]
MIN_INSAN_GUN = 3   # insan trendi icin gereken minimum farkli gun


# ===========================================================================
# YARDIMCILAR
# ===========================================================================
def _hafta_ekle(d: pd.DataFrame, bas: pd.Timestamp, bit: pd.Timestamp,
                kolon: str = "tarih") -> pd.DataFrame:
    d = d.copy()
    d[kolon] = pd.to_datetime(d[kolon])
    d["hafta"] = (d[kolon] - bas).dt.days // 7 + 1
    return d[(d[kolon] >= bas) & (d[kolon] <= bit)]


def _hafta_etiketi(h: int, bas: pd.Timestamp, bit: pd.Timestamp) -> str:
    h1 = bas + pd.Timedelta(days=7 * (h - 1))
    h2 = min(bas + pd.Timedelta(days=7 * h - 1), bit)
    return f"H{h} ({h1:%m-%d} - {h2:%m-%d})"


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _egim(y: np.ndarray) -> dict:
    """Gunluk serinin dogrusal egimi. Isaret yonu, r gucu soyler."""
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    if ok.sum() < 5:
        return {"egim_gun": None, "r": None}
    x = np.arange(len(y))[ok]
    e, _ = np.polyfit(x, y[ok], 1)
    r = np.corrcoef(x, y[ok])[0, 1]
    return {"egim_gun": _r(e, 3), "r": _r(r, 3)}


def _son_vs_once(seri: list[dict], alan: str) -> tuple[float | None, float | None]:
    deg = [s[alan] for s in seri if s.get(alan) is not None]
    if len(deg) < 2:
        return None, None
    return deg[-1], float(np.mean(deg[:-1]))


# ===========================================================================
# 1) MODEL DRIFT — gercek satisla, hafta bazli
# ===========================================================================
def model_drift(ham_yol: str | None = None) -> dict:
    from model_validasyon import (ham_veri_bul, kupiec_pof, olc, smape,
                                  bias_yuzde, veri_hazirla)
    bas, bit = _pencere()
    d = veri_hazirla(ham_veri_bul(ham_yol))
    d = _hafta_ekle(d[~d["kapali"]], bas, bit)

    haftalik = []
    for h, g in d.groupby("hafta"):
        m = olc(g)
        ihlal = int(((g["gercek_eur"] < g["p10_eur"]) |
                     (g["gercek_eur"] > g["p90_eur"])).sum())
        k = kupiec_pof(ihlal, len(g))
        haftalik.append({
            "hafta": int(h), "etiket": _hafta_etiketi(int(h), bas, bit),
            "n": m["n"], "sMAPE": m["sMAPE"], "wMAPE": m["wMAPE"],
            "bias_%": m["bias_%"], "kapsama_%": m["kapsama_%"],
            "p10_alti_%": m["p10_alti_%"], "p90_ustu_%": m["p90_ustu_%"],
            "kupiec_lr": k["lr"], "kupiec_red": bool(k["red"]),
        })

    gunluk = []
    for t, g in d.groupby("tarih"):
        y, f = g["gercek_eur"].to_numpy(float), g["p50_eur"].to_numpy(float)
        ic = ((g["gercek_eur"] >= g["p10_eur"]) &
              (g["gercek_eur"] <= g["p90_eur"])).mean() * 100
        gunluk.append({"tarih": str(pd.Timestamp(t).date()),
                       "sMAPE": _r(smape(y, f)), "kapsama_%": _r(ic, 1),
                       "bias_%": _r(bias_yuzde(y, f))})

    return {
        "haftalik": haftalik,
        "gunluk": gunluk,
        "egim_smape": _egim([g["sMAPE"] for g in gunluk]),
        "egim_kapsama": _egim([g["kapsama_%"] for g in gunluk]),
    }


# ===========================================================================
# 2) KARAR TRENDI — karar_tool kararlari + GERCEKLESEN sonuc
# ===========================================================================
def karar_trendi() -> dict:
    """Hafta bazli karar dagilimi ve gercek satisla olculen kalite.

    fire_yakalama_% : gercekte fire olan gunlerin kacinda agent indirim
                      sinyali verdi (karar veya onerilen aksiyon)
    kacirilan_eur   : sinyal verilmeyen gunlerin gerceklesen fire maliyeti
    devredilen_eur  : insana devredilen gunlerin gerceklesen kaybi
    model_hatasi_eur_100_adet : politika backtest'inden, (model - kusursuz
                      bilgi) kaybi / 100 adet gercek talep
    """
    if not KARAR_CSV.exists():
        return {"durum": "yok",
                "mesaj": "backtest_kararlar.csv yok. Once "
                         "'python gercek_backtest.py' calistirin."}
    from projeksiyon_uret import BIRIM_KAR, BIRIM_MALIYET
    bas, bit = _pencere()
    d = _hafta_ekle(pd.read_csv(KARAR_CSV), bas, bit)

    indirim = (d["karar"] == "indirim_uygula") | \
        d["onerilen_aksiyon"].fillna("").astype(str).str.contains("indirim")
    fire_eur = d["fire_gercek"] * BIRIM_MALIYET
    kayip = fire_eur + d["stoksuz_gercek"] * BIRIM_KAR

    bt = {}
    if BACKTEST_JSON.exists():
        for h in json.loads(BACKTEST_JSON.read_text(encoding="utf-8"))["politika"].get("hafta", []):
            bt[int(h["hafta"])] = h

    haftalik = []
    for h, g in d.groupby("hafta"):
        i = g.index
        fire_oldu = g["fire_gercek"] > 0.5
        ind = indirim[i]
        dag = g["karar"].value_counts(normalize=True) * 100
        haftalik.append({
            "hafta": int(h), "etiket": _hafta_etiketi(int(h), bas, bit),
            "n": len(g),
            **{f"{kk}_%": _r(dag.get(kk, 0.0), 1) for kk in KARARLAR},
            "onay_yuku_%": _r(g["onay_gerekli_mi"].astype(bool).mean() * 100, 1),
            "fire_yakalama_%": _r((ind & fire_oldu).sum() / fire_oldu.sum() * 100, 1)
                               if fire_oldu.sum() else None,
            "kacirilan_gun": int((~ind & fire_oldu).sum()),
            "kacirilan_eur": _r(fire_eur[i][~ind & fire_oldu].sum(), 0),
            "devredilen_eur": _r(kayip[i][g["karar"] == "insana_sor"].sum(), 0),
            "model_hatasi_eur_100_adet": _r(bt.get(int(h), {}).get("eur_per_100_adet"), 2),
        })
    return {"durum": "tamam", "kaynak": "gercek satis (gercek_backtest.py)",
            "haftalik": haftalik}


# ===========================================================================
# 3) INSAN TRENDI — onay kayitlari, gercek zaman
# ===========================================================================
def insan_trendi() -> dict:
    import ogrenme_dongusu as O
    if not O.LOG.exists():   # kayitlari_oku() dosya yoksa sys.exit yapar
        return {"durum": "yok", "mesaj": "Henuz onay kaydi yok."}
    son = O.son_durumlar(O.kayitlari_oku())
    if not son:
        return {"durum": "yok", "mesaj": "Henuz onay kaydi yok."}

    df = pd.DataFrame(son)
    df["gun"] = pd.to_datetime(df["zaman"]).dt.date.astype(str)
    gunluk = []
    for gun, g in df.groupby("gun"):
        onay = int((g["durum"] == "onaylandi").sum())
        gunluk.append({
            "gun": gun, "karar": len(g), "onaylandi": onay,
            "reddedildi": len(g) - onay,
            "onay_orani_%": _r(onay / len(g) * 100, 1),
            "red_sebepleri": dict(Counter(
                g.loc[g["durum"] == "reddedildi", "sebep_kodu"].fillna("belirtilmemis"))),
        })
    return {
        "durum": "tamam" if len(gunluk) >= MIN_INSAN_GUN else "yetersiz",
        "mesaj": None if len(gunluk) >= MIN_INSAN_GUN else
                 f"{len(gunluk)} gun kayit var; trend icin en az {MIN_INSAN_GUN} gun gerekir.",
        "gunluk": gunluk,
    }


# ===========================================================================
# 4) ALARMLAR — son hafta vs onceki haftalarin ortalamasi
# ===========================================================================
def alarmlar(model: dict, karar: dict) -> list[dict]:
    a = []
    hw = model.get("haftalik", [])

    son, once = _son_vs_once(hw, "sMAPE")
    if son is not None and once and son > once * (1 + ESIK["smape_goreli_artis"]):
        a.append({"alan": "model", "seviye": "yuksek",
                  "mesaj": f"sMAPE son hafta {son:.2f}, onceki ort. {once:.2f} "
                           f"(%{(son / once - 1) * 100:.0f} kotulesme)"})

    son, once = _son_vs_once(hw, "kapsama_%")
    if son is not None and (son < once - ESIK["kapsama_dusus_puan"]
                            or son < ESIK["kapsama_mutlak_alt"]):
        a.append({"alan": "model", "seviye": "orta",
                  "mesaj": f"Kapsama son hafta %{son:.1f}, onceki ort. %{once:.1f} "
                           f"(hedef %80)"})

    son, once = _son_vs_once(hw, "p90_ustu_%")
    if son is not None and son > once + ESIK["p90_ustu_artis_puan"]:
        a.append({"alan": "model", "seviye": "orta",
                  "mesaj": f"p90 ustu ihlal %{once:.1f} -> %{son:.1f}: gercek talep "
                           f"ust bandi daha sik asiyor, stoksuzluk riski artar"})

    if hw and abs(hw[-1]["bias_%"]) > ESIK["bias_mutlak"]:
        yon = "dusuk" if hw[-1]["bias_%"] < 0 else "yuksek"
        a.append({"alan": "model", "seviye": "orta",
                  "mesaj": f"Son hafta sistematik {yon} tahmin (bias %{hw[-1]['bias_%']:.1f})"})

    kh = karar.get("haftalik", [])
    son, once = _son_vs_once(kh, "onay_yuku_%")
    if son is not None and son > once + ESIK["onay_yuku_artis_puan"]:
        a.append({"alan": "karar", "seviye": "orta",
                  "mesaj": f"Onay yuku %{once:.1f} -> %{son:.1f}"})

    son, once = _son_vs_once(kh, "kacirilan_eur")
    if son is not None and once and son > once * ESIK["kacirilan_kat"]:
        a.append({"alan": "karar", "seviye": "yuksek",
                  "mesaj": f"Kacirilan fire {once:,.0f} -> {son:,.0f} EUR"})

    son, once = _son_vs_once(kh, "model_hatasi_eur_100_adet")
    if son is not None and once and son > once * ESIK["model_hatasi_kat"]:
        a.append({"alan": "karar", "seviye": "yuksek",
                  "mesaj": f"Model hatasinin bedeli {once:.1f} -> {son:.1f} EUR / 100 adet "
                           f"(tahmin hatasi karar maliyetine yansiyor)"})
    return a


# ===========================================================================
# ANA HESAP + ONBELLEK
# ===========================================================================
def trend_hesapla(ham_yol: str | None = None) -> dict:
    model = model_drift(ham_yol)
    karar = karar_trendi()
    insan = insan_trendi()
    rapor = {
        "uretim_zamani": datetime.now().isoformat(timespec="seconds"),
        "esikler": ESIK,
        "model": model, "karar": karar, "insan": insan,
        "alarmlar": alarmlar(model, karar),
    }
    RAPOR.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    return rapor


def trend_yukle(yenile: bool = False, max_yas_saat: float = 24) -> dict:
    """Is zamani verisi degismedigi icin sonuc onbellege alinir. Onay
    kayitlari degisir; insan trendi her cagrida tazelenir (ucuz)."""
    if not yenile and RAPOR.exists():
        r = json.loads(RAPOR.read_text(encoding="utf-8"))
        yas = (datetime.now() - datetime.fromisoformat(r["uretim_zamani"])).total_seconds()
        if yas < max_yas_saat * 3600:
            r["insan"] = insan_trendi()
            return r
    return trend_hesapla()


# ===========================================================================
# AGENT ICIN KISA OZET — trend_ozeti araci bunu dondurur
# ===========================================================================
def trend_ozeti() -> str:
    """LLM icin kisa, hesaplanmis ozet. LLM hicbir farki kendi hesaplamaz;
    degisim yuzdeleri burada hazir verilir."""
    try:
        r = trend_yukle()
    except SystemExit as e:           # train.csv bulunamadi vb.
        return json.dumps({"hata": str(e)}, ensure_ascii=False)
    hw = r["model"]["haftalik"]
    ozet = {
        "kapsam": "Zincir geneli, acik gunler, haftalik. Son hafta onceki "
                  "haftalarin ortalamasiyla kiyaslanir.",
        "model_haftalik": [{k: h[k] for k in ("etiket", "sMAPE", "kapsama_%",
                            "p90_ustu_%", "bias_%")} for h in hw],
        "smape_gunluk_egim": r["model"]["egim_smape"],
        "kapsama_gunluk_egim": r["model"]["egim_kapsama"],
    }
    s, o = _son_vs_once(hw, "sMAPE")
    if s is not None and o:
        ozet["smape_son_hafta_degisim_goreli_%"] = _r((s / o - 1) * 100, 1)
    s, o = _son_vs_once(hw, "kapsama_%")
    if s is not None:
        ozet["kapsama_son_hafta_degisim_puan"] = _r(s - o, 1)
    if r["karar"].get("durum") == "tamam":
        ozet["karar_haftalik"] = [{k: h.get(k) for k in (
            "etiket", "onay_yuku_%", "insana_sor_%", "fire_yakalama_%",
            "kacirilan_eur", "model_hatasi_eur_100_adet")}
            for h in r["karar"]["haftalik"]]
        ozet["karar_notu"] = ("model_hatasi_eur_100_adet: tahmin hatasinin karar "
                              "zincirine maliyeti, gercek satisla olculdu.")
    ozet["insan_onay"] = ({"durum": r["insan"]["durum"], "mesaj": r["insan"].get("mesaj"),
                           "gun_sayisi": len(r["insan"].get("gunluk", []))})
    ozet["alarmlar"] = [x["mesaj"] for x in r["alarmlar"]] or ["Alarm yok"]
    return json.dumps(ozet, ensure_ascii=False)


# ===========================================================================
# GUNLUK OTOMATIK RAPOR — n8n Schedule bunu cagirir
# ===========================================================================
def gunluk_rapor() -> dict:
    """Trend + ogrenme dongusu tek pakette. Doner:
    alarm (bool), mesaj (Telegram metni), gecmis (Sheets'e yazilacak satir)."""
    tr = trend_yukle()

    import ogrenme_dongusu as O
    og = None
    if O.LOG.exists():
        with contextlib.redirect_stdout(io.StringIO()):
            og = O.rapor(O.kayitlari_oku(), O.VARSAYILAN_MIN)

    hw = tr["model"]["haftalik"]
    kh = tr["karar"].get("haftalik", [])
    son_m = hw[-1] if hw else {}
    son_k = kh[-1] if kh else {}
    al = tr["alarmlar"]

    satirlar = [("🔴 DRIFT ALARMI" if al else "📊 GUNLUK SISTEM RAPORU")
                + f" — {datetime.now():%Y-%m-%d}", ""]
    if hw:
        satirlar.append("Model (haftalik sMAPE / kapsama):")
        satirlar += [f"  {h['etiket']}: {h['sMAPE']:.2f} / %{h['kapsama_%']:.1f}" for h in hw]
    if al:
        satirlar += ["", "Alarmlar:"] + [f"  • {x['mesaj']}" for x in al]
    if son_k:
        satirlar += ["", f"Son hafta karar: onay yuku %{son_k['onay_yuku_%']}, "
                         f"kacirilan fire {son_k['kacirilan_eur'] or 0:,.0f} EUR"]
        if son_k.get("model_hatasi_eur_100_adet") is not None:
            satirlar.append(f"Model hatasinin bedeli: "
                            f"{son_k['model_hatasi_eur_100_adet']:.1f} EUR / 100 adet")
    if og:
        oran = "-" if og["onay_orani"] is None else f"%{og['onay_orani'] * 100:.0f}"
        satirlar += ["", f"Insan onayi: {og['benzersiz_karar']} karar, "
                         f"{og['onaylandi']} onay / {og['reddedildi']} red ({oran})"]
        yeterli = [o for o in og["oneriler"] if o["yeterli_veri"]]
        if yeterli:
            satirlar += ["Esik onerileri (elle uygulanir):"] + \
                        [f"  • {o['baslik']}" for o in yeterli]
        elif og["oneriler"]:
            satirlar.append(f"{len(og['oneriler'])} oneri var, veri yetersiz "
                            f"(min {og['min_gozlem']}); uygulanmaz.")
    satirlar += ["", "Not: sistem esikleri kendisi DEGISTIRMEZ."]

    return {
        "alarm": bool(al),
        "alarm_sayisi": len(al),
        "mesaj": "\n".join(satirlar),
        "gecmis": {
            "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "son_hafta_smape": son_m.get("sMAPE"),
            "son_hafta_kapsama": son_m.get("kapsama_%"),
            "son_hafta_p90_ustu": son_m.get("p90_ustu_%"),
            "onay_yuku": son_k.get("onay_yuku_%"),
            "kacirilan_eur": son_k.get("kacirilan_eur"),
            "insan_karar": og["benzersiz_karar"] if og else 0,
            "insan_onay_orani": og["onay_orani"] if og else None,
            "oneri_sayisi": len(og["oneriler"]) if og else 0,
            "alarm_sayisi": len(al),
            "alarmlar": " | ".join(x["mesaj"] for x in al),
        },
    }


# ===========================================================================
# KONSOL RAPORU + GRAFIK
# ===========================================================================
def yazdir(r: dict) -> None:
    print("=" * 74)
    print("TREND VE DRIFT ANALIZI")
    print("=" * 74)
    hw = pd.DataFrame(r["model"]["haftalik"])
    print("\n--- 1) Model — haftalik (kapali gunler haric) ---")
    print(hw[["etiket", "n", "sMAPE", "bias_%", "kapsama_%", "p10_alti_%",
              "p90_ustu_%", "kupiec_lr"]].to_string(index=False))
    es, ek = r["model"]["egim_smape"], r["model"]["egim_kapsama"]
    print(f"\nGunluk egim: sMAPE {es['egim_gun']} puan/gun (r={es['r']}), "
          f"kapsama {ek['egim_gun']} puan/gun (r={ek['r']})")

    print("\n--- 2) Karar — haftalik (gercek satisla) ---")
    if r["karar"].get("durum") == "tamam":
        kh = pd.DataFrame(r["karar"]["haftalik"])
        print(kh[["etiket", "n", "siparis_ver_%", "bekle_%", "insana_sor_%",
                  "indirim_uygula_%", "onay_yuku_%", "fire_yakalama_%",
                  "kacirilan_eur", "model_hatasi_eur_100_adet"]].to_string(index=False))
    else:
        print("  " + r["karar"]["mesaj"])

    print("\n--- 3) Insan onayi — gunluk (gercek zaman) ---")
    ins = r["insan"]
    if ins.get("gunluk"):
        for g in ins["gunluk"]:
            print(f"  {g['gun']}: {g['karar']} karar, onay %{g['onay_orani_%']}, "
                  f"red sebepleri {g['red_sebepleri'] or '-'}")
    if ins.get("mesaj"):
        print("  NOT: " + ins["mesaj"])

    print("\n--- 4) ALARMLAR (son hafta vs onceki haftalar) ---")
    if r["alarmlar"]:
        for a in r["alarmlar"]:
            print(f"  [{a['seviye'].upper()}] {a['alan']}: {a['mesaj']}")
    else:
        print("  Alarm yok.")
    print("\nNOT: Alarmlar esik DEGISTIRMEZ; insanin dikkatini ceker.")
    print(f"Rapor: {RAPOR}")


def grafik(r: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = pd.DataFrame(r["model"]["gunluk"])
    g["tarih"] = pd.to_datetime(g["tarih"])
    hw = pd.DataFrame(r["model"]["haftalik"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    for a, kol, baslik, ref in ((ax[0, 0], "sMAPE", "Gunluk sMAPE (dusuk = iyi)", None),
                                (ax[0, 1], "kapsama_%", "Gunluk kapsama p10-p90", 80)):
        a.plot(g["tarih"], g[kol], marker="o", ms=3)
        ok = g[kol].notna()
        x = np.arange(len(g))[ok]
        e, b = np.polyfit(x, g.loc[ok, kol], 1)
        a.plot(g["tarih"][ok], e * x + b, "--", label=f"egim {e:+.3f}/gun")
        if ref:
            a.axhline(ref, color="gray", lw=1, ls=":", label=f"hedef %{ref}")
        a.set_title(baslik); a.legend(); a.tick_params(axis="x", rotation=30)

    x = np.arange(len(hw)); w = 0.38
    ax[1, 0].bar(x - w / 2, hw["p10_alti_%"], w, label="p10 alti")
    ax[1, 0].bar(x + w / 2, hw["p90_ustu_%"], w, label="p90 ustu")
    ax[1, 0].axhline(10, color="gray", lw=1, ls=":", label="nominal %10")
    ax[1, 0].set_xticks(x, [f"H{h}" for h in hw["hafta"]])
    ax[1, 0].set_title("Haftalik bant ihlali (simetri)"); ax[1, 0].legend()

    if r["karar"].get("durum") == "tamam":
        kh = pd.DataFrame(r["karar"]["haftalik"])
        a = ax[1, 1]
        etiket = [f"H{h}" for h in kh["hafta"]]
        a.bar(etiket, kh["onay_yuku_%"], color="tab:orange", alpha=0.7,
              label="onay yuku %")
        a.set_ylabel("onay yuku %")
        a2 = a.twinx()
        a2.plot(etiket, kh["model_hatasi_eur_100_adet"], "k-o",
                label="model hatasi EUR / 100 adet")
        a2.set_ylabel("EUR / 100 adet")
        a.set_title("Onay yuku ve model hatasinin bedeli (gercek satisla)")
        a.legend(loc="upper left"); a2.legend(loc="upper right")
    else:
        ax[1, 1].axis("off")

    fig.suptitle("Rossmann — trend ve drift (2015-07-04 / 2015-07-31)")
    fig.tight_layout()
    fig.savefig(GRAFIK, dpi=130)
    print(f"Grafik: {GRAFIK}")


def main() -> None:
    a = sys.argv
    ham = a[a.index("--ham") + 1] if "--ham" in a else None
    r = trend_hesapla(ham)
    yazdir(r)
    if "--grafik" in a:
        grafik(r)


if __name__ == "__main__":
    main()
