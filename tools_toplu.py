"""
tools_toplu.py — agent v7 icin toplu (cok magazali) sorgu tool'lari.

NE YAPAR: 1115 magazalik hazir tablolari okuyan INCE bir katman. Hicbir tool
model calistirmaz, tahmin uretmez, hesap "icat etmez"; yalnizca uretilmis
tablolardan filtreleyip ozetler.

MIMARI ILKE (v6'dan devralindi): LLM sayi URETMEZ. Burada LLM'in tek isi
hangi soruya hangi tool'un cevap verecegini secmektir — bir YONLENDIRICI'dir.
Sayilarin tamami bu modulden, yani deterministik kaynaktan gelir.

IKINCI ILKE (v6 hata (i)/(j)'nin bu katmandaki karsiligi):
    Bir bilgi kritikse, onu ayri bir tool'a birakma — ayni ciktiya koy.
    Magaza listeleyen her tool, o magazanin tahmin hatasini (smape_val) de
    dondurur. Boylece ajan model_performansi'yi cagirmayi unutsa bile
    guvenilirlik bilgisi elindedir. Savunma, ajanin dogru tool'u secmesine
    bagimli olmamalidir.

UCUNCU ILKE (sizinti): karar ve uyari ureten her alan KARAR ANINDA BILINEN
    metrige dayanir. "Bugun" 2015-07-03; test donemi (07-04..07-31) hatasi o
    gun bilinemez. Bu yuzden tahmin_zayif_mi, zincir ortalamasi ve liste
    tool'larindaki hata alani VALIDASYON donemi metrigidir. Test metrikleri
    yalniz model_performansi'nda "geriye_donuk" basligi altinda raporlanir.

VERI KAYNAKLARI (hepsi onceden uretilmis):
    Rossman/model/tahminler_tum_magazalar.parquet   93.660 satir (1115x28x3)
    Rossman/model/tahmin_meta.json                  demo_bugun, donusum sabitleri
    Rossman/projeksiyon.parquet                     187.320 satir (2 mod x 3 senaryo)
    Rossman/projeksiyon_ozet.csv                    1115 satir
    Rossman/sevkiyat_plani.csv                      14.648 satir
    Rossman/stok_baslangic.csv                      1115 satir

TAHMIN PENCERESI: 2015-07-04 .. 2015-07-31 (28 gun). "Bugun" = 2015-07-03.
Pencere disi her istek NET HATA doner — canli demoda juri bunu deneyecek.

Calistirma (kendi basina duman testi):
    python tools_toplu.py
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parent
ROSSMAN = KOK / "Rossman"

BIRIM_FIYAT = 8.0
MARJ_ORANI = 0.25
BIRIM_KAR = BIRIM_FIYAT * MARJ_ORANI              # 2.0
BIRIM_MALIYET = BIRIM_FIYAT * (1 - MARJ_ORANI)    # 6.0

MAKS_SATIR = 25          # LLM'e donen liste uzunlugu tavani (token kontrolu)


# ===========================================================================
# VERI YUKLEME — tembel, tek sefer
# ===========================================================================
@lru_cache(maxsize=1)
def _tahmin() -> pd.DataFrame:
    y = ROSSMAN / "model" / "tahminler_tum_magazalar.parquet"
    if not y.exists():
        raise SystemExit(f"HATA: {y} yok. Once colab_03_tahmin_uret.py calistir.")
    d = pd.read_parquet(y)
    d["tarih"] = pd.to_datetime(d["tarih"])
    return d


@lru_cache(maxsize=1)
def _meta() -> dict:
    y = ROSSMAN / "model" / "tahmin_meta.json"
    return json.loads(y.read_text(encoding="utf-8")) if y.exists() else {}


@lru_cache(maxsize=1)
def _projeksiyon() -> pd.DataFrame:
    y = ROSSMAN / "projeksiyon.parquet"
    if not y.exists():
        raise SystemExit(f"HATA: {y} yok. Once projeksiyon_uret.py calistir.")
    d = pd.read_parquet(y)
    d["tarih"] = pd.to_datetime(d["tarih"])
    return d


@lru_cache(maxsize=1)
def _ozet() -> pd.DataFrame:
    return pd.read_csv(ROSSMAN / "projeksiyon_ozet.csv")


@lru_cache(maxsize=1)
def _sevkiyat() -> pd.DataFrame:
    d = pd.read_csv(ROSSMAN / "sevkiyat_plani.csv")
    d["tarih"] = pd.to_datetime(d["tarih"])
    d["varis_tarihi"] = pd.to_datetime(d["varis_tarihi"])
    return d


@lru_cache(maxsize=1)
def _stok() -> pd.DataFrame:
    return pd.read_csv(ROSSMAN / "stok_baslangic.csv")


@lru_cache(maxsize=1)
def _pencere() -> tuple[pd.Timestamp, pd.Timestamp]:
    t = _tahmin()["tarih"]
    return t.min(), t.max()


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _perf() -> pd.DataFrame:
    """Magaza bazli model performansi — tek yerde, yuzdeye cevrilmis.

    smape_val / kapsama_val / n_val : karar aninda bilinen (val donemi)
    smape_test / kapsama_test       : geriye donuk, yalniz raporlama
    Eski tahmin dosyalarinda kapsama_val / n_val yoksa NaN kalir; kapi
    bu durumda yalniz sMAPE ile calisir (bkz. karar_tool.belirsizlik_kapisi).
    """
    t = _tahmin()
    t = t[t["senaryo"] == "planli"]
    kolonlar = {"smape_val": "magaza_smape_val",
                "kapsama_val": "magaza_kapsama_val",
                "n_val": "magaza_n_val",
                "smape_test": "magaza_smape_test",
                "kapsama_test": "magaza_kapsama_test"}
    g = t.groupby("magaza").first().reset_index()
    out = pd.DataFrame({"magaza": g["magaza"]})
    for yeni, eski in kolonlar.items():
        out[yeni] = g[eski] if eski in g.columns else float("nan")
    # kaynak oran mi yuzde mi olabilir; tek yerde normalize edilir
    for k in ("smape_val", "kapsama_val", "smape_test", "kapsama_test"):
        if out[k].notna().any() and out[k].max() <= 1.0:
            out[k] = out[k] * 100
        out[k] = out[k].round(2)
    return out


def _perf_ekle(d: pd.DataFrame) -> pd.DataFrame:
    return d.merge(_perf()[["magaza", "smape_val"]], on="magaza", how="left")


@lru_cache(maxsize=1)
def _zincir_ort_smape() -> float:
    return round(float(_perf()["smape_val"].mean()), 2)


# ===========================================================================
# SAVUNMACI GIRDI AYRISTIRMA
# Canli demoda juri once buralari zorlar: olmayan magaza, pencere disi tarih,
# ters sirali aralik. Hepsi COKME degil, NET HATA METNI dondurur.
# ===========================================================================
def _hata(mesaj: str) -> str:
    return json.dumps({"hata": mesaj}, ensure_ascii=False)


def _tarih_ayristir(deger, alan: str):
    """Doner: (Timestamp, None) veya (None, hata_metni)."""
    bas, bit = _pencere()
    try:
        t = pd.Timestamp(str(deger))
    except Exception:
        return None, _hata(
            f"'{deger}' gecerli bir tarih degil ({alan}). Bicim: YYYY-MM-DD. "
            f"Tahmin penceresi {bas.date()} - {bit.date()}.")
    if pd.isna(t) or not (bas <= t <= bit):
        return None, _hata(
            f"{t.date() if not pd.isna(t) else deger} tahmin penceresinin "
            f"disinda ({alan}). Gecerli aralik: {bas.date()} - {bit.date()}. "
            f"Bu tarih icin uretilmis tahmin yok.")
    return t, None


def _aralik_ayristir(baslangic, bitis):
    bas, bit = _pencere()
    b1, h = (_tarih_ayristir(baslangic, "baslangic") if baslangic
             else (bas, None))
    if h:
        return None, None, h
    b2, h = _tarih_ayristir(bitis, "bitis") if bitis else (bit, None)
    if h:
        return None, None, h
    if b1 > b2:
        b1, b2 = b2, b1
    return b1, b2, None


def _magaza_ayristir(deger):
    try:
        m = int(deger)
    except (TypeError, ValueError):
        return None, _hata(
            f"'{deger}' gecerli bir magaza numarasi degil. 1-1115 arasi bir "
            f"tam sayi ver.")
    if m not in set(_ozet()["magaza"].tolist()):
        return None, _hata(
            f"Magaza {m} yok. Gecerli magaza numaralari 1 ile 1115 arasindadir.")
    return m, None


def _n_ayristir(n, varsayilan=10):
    try:
        k = int(n)
    except (TypeError, ValueError):
        k = varsayilan
    return max(1, min(k, MAKS_SATIR))


def _sayi_ayristir(deger, alan: str, taban=0.0):
    try:
        x = float(deger)
    except (TypeError, ValueError):
        return None, _hata(f"'{deger}' gecerli bir sayi degil ({alan}).")
    if x <= taban:
        return None, _hata(f"{alan} {taban}'dan buyuk olmali (verilen: {deger}).")
    return x, None


# ===========================================================================
# TOOL 1 — tukenme riski
# ===========================================================================
def tukenme_riski(baslangic: str = "", bitis: str = "", n: int = 10) -> str:
    """Belirtilen tarih araliginda stogu tukenecek magazalari, en erken
    tukenenden baslayarak listeler. YUKSEK TALEP (p90) senaryosuna ve
    SIPARIS VERILMEZSE varsayimina dayanir; yani en kotu durum analizidir.
    Tarihler YYYY-MM-DD bicimindedir ve bos birakilirsa tum tahmin penceresi
    kullanilir. n, listelenecek magaza sayisidir (en fazla 25)."""
    b1, b2, h = _aralik_ayristir(baslangic, bitis)
    if h:
        return h
    k = _n_ayristir(n)

    p = _projeksiyon()
    alt = p[(p["mod"] == "siparis_yok") & (p["talep_senaryo"] == "p90")
            & (p["tarih"] >= b1) & (p["tarih"] <= b2) & (p["stoksuz"] > 0.5)]
    if alt.empty:
        return json.dumps({
            "aralik": f"{b1.date()} - {b2.date()}",
            "tukenecek_magaza_sayisi_TOPLAM": 0,
            "not": "Bu aralikta stogu tukenen magaza yok.",
        }, ensure_ascii=False)

    ilk = alt.groupby("magaza").agg(
        ilk_tukenme_tarihi=("tarih", "min"),
        toplam_stoksuz_adet=("stoksuz", "sum"),
        kacan_kar_eur=("kacan_kar_eur", "sum")).reset_index()
    ilk = ilk.merge(_ozet()[["magaza", "tedarikci", "aciliyet",
                             "stok_toplam_adet", "tipik_gunluk_talep_adet",
                             "onay_sebep"]], on="magaza")
    ilk = _perf_ekle(ilk)
    ilk = ilk.sort_values(["ilk_tukenme_tarihi", "kacan_kar_eur"],
                          ascending=[True, False]).head(k)

    ort = _zincir_ort_smape()
    return json.dumps({
        "aralik": f"{b1.date()} - {b2.date()}",
        "senaryo": "p90 talep, siparis verilmezse (en kotu durum)",
        "tukenecek_magaza_sayisi_TOPLAM": int(alt["magaza"].nunique()),
        "bu_ciktida_listelenen": len(ilk),
        "toplam_kacan_kar_eur_TUM_MAGAZALAR": round(float(alt["kacan_kar_eur"].sum()), 1),
        "zincir_ort_smape_yuzde": ort,
        "magazalar": [{
            "magaza": int(r.magaza),
            "ilk_tukenme_tarihi": r.ilk_tukenme_tarihi.strftime("%Y-%m-%d"),
            "tedarikci": r.tedarikci,
            "aciliyet": r.aciliyet,
            "mevcut_stok_adet": round(float(r.stok_toplam_adet), 1),
            "gunluk_talep_adet": round(float(r.tipik_gunluk_talep_adet), 1),
            "stoksuz_adet": round(float(r.toplam_stoksuz_adet), 1),
            "kacan_kar_eur": round(float(r.kacan_kar_eur), 1),
            "onay_sebebi": r.onay_sebep,
            "tahmin_hatasi_smape_yuzde": float(r.smape_val),
            "tahmin_zayif_mi": bool(r.smape_val > ort),
        } for r in ilk.itertuples()],
        "not": ("tahmin_zayif_mi=true olan magazalarda model zincir "
                "ortalamasindan kotu calisiyor; sayilari temkinli yorumla."),
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 2 — sevkiyat plani
# ===========================================================================
def sevkiyat_plani_getir(tarih: str, n: int = 15) -> str:
    """Belirtilen SIPARIS TARIHINDE hangi magazaya ne kadar gonderilecegini,
    hangi tedarikciden ve ne zaman varacagini listeler. Tutar ve onay durumu
    da doner. Tarih YYYY-MM-DD bicimindedir. n en fazla 25'tir."""
    t, h = _tarih_ayristir(tarih, "tarih")
    if h:
        return h
    k = _n_ayristir(n, 15)

    s = _sevkiyat()
    gun = s[s["tarih"] == t]
    if gun.empty:
        return json.dumps({
            "tarih": str(t.date()),
            "sevkiyat_sayisi_TOPLAM": 0,
            "not": "Bu tarihte planlanmis sevkiyat yok (tedarikci teslimat "
                   "gunu olmayabilir).",
        }, ensure_ascii=False)

    ted = gun.groupby("tedarikci").agg(
        magaza=("magaza", "size"), adet=("siparis", "sum"),
        tutar_eur=("tutar_eur", "sum"),
        onay_zincir=("onay_zincir", "max")).round(1).reset_index()

    ilk = gun.sort_values("siparis", ascending=False).head(k)
    return json.dumps({
        "tarih": str(t.date()),
        "sevkiyat_sayisi_TOPLAM": len(gun),
        "toplam_adet": round(float(gun["siparis"].sum()), 1),
        "toplam_tutar_eur": round(float(gun["tutar_eur"].sum()), 1),
        "onay_gereken_sevkiyat_TOPLAM": int(gun["onay_gerekli"].sum()),
        "bu_ciktida_listelenen": len(ilk),
        "tedarikci_ozeti": [{
            "tedarikci": r.tedarikci, "magaza_sayisi": int(r.magaza),
            "adet": float(r.adet), "tutar_eur": float(r.tutar_eur),
            "zincir_onayi_gerekli": bool(r.onay_zincir),
        } for r in ted.itertuples()],
        "en_buyuk_sevkiyatlar": [{
            "magaza": int(r.magaza), "tedarikci": r.tedarikci,
            "adet": round(float(r.siparis), 1),
            "tutar_eur": round(float(r.tutar_eur), 1),
            "varis_tarihi": r.varis_tarihi.strftime("%Y-%m-%d"),
            "onay_tekil": bool(r.onay_tekil), "onay_zincir": bool(r.onay_zincir),
        } for r in ilk.itertuples()],
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 3 — en cok ihtiyaci olan magazalar
# ===========================================================================
def en_cok_ihtiyac_duyan(baslangic: str = "", bitis: str = "", n: int = 10) -> str:
    """Belirtilen aralikta ihtiyaci en buyuk olan magazalari siralar.
    Siralama olcutu, kacirilan satisin magazanin KENDI gunluk talebine
    oranidir; boylece buyuk magazalar listeyi otomatik doldurmaz, gercekten
    kitlik yasayan kucuk magazalar da gorunur. n en fazla 25'tir."""
    b1, b2, h = _aralik_ayristir(baslangic, bitis)
    if h:
        return h
    k = _n_ayristir(n)

    p = _projeksiyon()
    alt = p[(p["mod"] == "siparis_yok") & (p["talep_senaryo"] == "p50")
            & (p["tarih"] >= b1) & (p["tarih"] <= b2)]
    g = alt.groupby("magaza").agg(
        stoksuz_adet=("stoksuz", "sum"),
        kacan_kar_eur=("kacan_kar_eur", "sum")).reset_index()
    g = g.merge(_ozet()[["magaza", "tedarikci", "aciliyet",
                         "tipik_gunluk_talep_adet", "stok_toplam_adet",
                         "siparis_toplam"]], on="magaza")
    g = _perf_ekle(g)
    g["eksik_stok_gun_karsiligi"] = (
        g["stoksuz_adet"] / g["tipik_gunluk_talep_adet"].clip(lower=1e-6)).round(2)
    toplam = int((g["stoksuz_adet"] > 0.5).sum())
    g = g[g["stoksuz_adet"] > 0.5].sort_values(
        ["eksik_stok_gun_karsiligi", "kacan_kar_eur"], ascending=False).head(k)

    if g.empty:
        return json.dumps({"aralik": f"{b1.date()} - {b2.date()}",
                           "not": "Bu aralikta ihtiyac tespit edilmedi."},
                          ensure_ascii=False)

    ort = _zincir_ort_smape()
    return json.dumps({
        "aralik": f"{b1.date()} - {b2.date()}",
        "olcut": "kacirilan satis / magazanin gunluk talebi",
        "olcut_aciklama": ("eksik_stok_gun_karsiligi = kacirilan toplam satis, "
                           "magazanin KAC GUNLUK talebine denk geliyor. "
                           "Bu bir sure degil, bir buyukluk olcusudur."),
        "senaryo": "p50 talep, siparis verilmezse",
        "ihtiyac_duyan_magaza_sayisi_TOPLAM": toplam,
        "bu_ciktida_listelenen": len(g),
        "zincir_ort_smape_yuzde": ort,
        "magazalar": [{
            "magaza": int(r.magaza), "tedarikci": r.tedarikci,
            "aciliyet": r.aciliyet,
            "eksik_stok_gun_karsiligi": float(r.eksik_stok_gun_karsiligi),
            "eksik_adet": round(float(r.stoksuz_adet), 1),
            "gunluk_talep_adet": round(float(r.tipik_gunluk_talep_adet), 1),
            "mevcut_stok_adet": round(float(r.stok_toplam_adet), 1),
            "planlanan_siparis_adet": round(float(r.siparis_toplam), 1),
            "kacan_kar_eur": round(float(r.kacan_kar_eur), 1),
            "tahmin_hatasi_smape_yuzde": float(r.smape_val),
            "tahmin_zayif_mi": bool(r.smape_val > ort),
        } for r in g.itertuples()],
        "not": ("tahmin_zayif_mi=true olan magazalarda model zincir "
                "ortalamasindan kotu calisiyor; onceliklendirirken bunu belirt."),
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 4 — segment ozeti
# ===========================================================================
def segment_ozeti(magaza_tipi: str = "", urun_yelpazesi: str = "") -> str:
    """Bir magaza segmentinin talep, stok, tahmin dogrulugu ve onay
    profilini ozetler. magaza_tipi 'a','b','c','d'; urun_yelpazesi
    'a','b','c' olabilir. Ikisi de bos birakilirsa tum segmentler
    karsilastirmali listelenir."""
    d = _ozet().merge(_stok()[["magaza", "urun_yelpazesi"]], on="magaza")
    d = d.merge(_perf(), on="magaza", how="left")

    tah = _tahmin()
    tah = tah[(tah["senaryo"] == "planli") & (tah["acik"] == 1)]
    ciro = tah.groupby("magaza")["p50_eur"].mean().rename(
        "gunluk_ciro_eur").reset_index()
    d = d.merge(ciro, on="magaza", how="left")

    if magaza_tipi:
        mt = str(magaza_tipi).strip().lower()
        if mt not in set(d["magaza_tipi"].astype(str).str.lower()):
            return _hata(f"'{magaza_tipi}' gecerli bir magaza tipi degil. "
                         f"Gecerli tipler: a, b, c, d.")
        d = d[d["magaza_tipi"].astype(str).str.lower() == mt]
    if urun_yelpazesi:
        uy = str(urun_yelpazesi).strip().lower()
        if uy not in set(d["urun_yelpazesi"].astype(str).str.lower()):
            return _hata(f"'{urun_yelpazesi}' gecerli bir urun yelpazesi degil. "
                         f"Gecerli degerler: a, b, c.")
        d = d[d["urun_yelpazesi"].astype(str).str.lower() == uy]

    kirilim = "magaza_tipi" if not magaza_tipi else (
        "urun_yelpazesi" if not urun_yelpazesi else None)

    def blok(alt: pd.DataFrame, etiket: str) -> dict:
        return {
            "segment": etiket,
            "magaza_sayisi": len(alt),
            "ort_gunluk_ciro_eur": round(float(alt["gunluk_ciro_eur"].mean()), 1),
            "ort_gunluk_talep_adet": round(float(alt["tipik_gunluk_talep_adet"].mean()), 1),
            "ort_smape_val_yuzde": round(float(alt["smape_val"].mean()), 2),
            "ort_kapsama_val_yuzde": (round(float(alt["kapsama_val"].mean()), 1)
                                      if alt["kapsama_val"].notna().any() else None),
            "ort_stok_gun_karsiligi": round(float(alt["stok_gun_karsiligi"].mean()), 2),
            "onay_gereken_magaza": int(alt["onay_gerekli"].sum()),
            "kritik_aciliyet": int((alt["aciliyet"] == "kritik").sum()),
            "toplam_siparis_adet": round(float(alt["siparis_toplam"].sum()), 1),
            "toplam_net_kazanc_eur": round(float(alt["net_kazanc_eur"].sum()), 1),
        }

    if kirilim:
        gruplar = [blok(g, f"{kirilim}={ad}")
                   for ad, g in d.groupby(d[kirilim].astype(str))]
    else:
        gruplar = [blok(d, f"magaza_tipi={magaza_tipi}, "
                           f"urun_yelpazesi={urun_yelpazesi}")]

    return json.dumps({
        "filtre": {"magaza_tipi": magaza_tipi or "hepsi",
                   "urun_yelpazesi": urun_yelpazesi or "hepsi"},
        "toplam_magaza": len(d),
        "zincir_ort_smape_yuzde": _zincir_ort_smape(),
        "segmentler": gruplar,
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 5 — model performansi (agent kendi zayifligini raporlar)
# ===========================================================================
def model_performansi(magaza: int) -> str:
    """Belirli bir magaza icin tahmin modelinin ne kadar guvenilir oldugunu
    raporlar: sMAPE hata orani ve tahmin bandinin gercek degeri yakalama
    orani (kapsama). Zincir ortalamasiyla karsilastirir."""
    m, h = _magaza_ayristir(magaza)
    if h:
        return h

    p = _perf()
    kendi = p[p["magaza"] == m].iloc[0]
    s_val = float(kendi["smape_val"])
    ort = _zincir_ort_smape()
    sira = int((p["smape_val"] < s_val).sum()) + 1
    k_val = None if pd.isna(kendi["kapsama_val"]) else float(kendi["kapsama_val"])

    # Karar kapisiyla AYNI hukum (kural tek yerde: karar_tool)
    from karar_tool import belirsizlik_kapisi
    kapi = belirsizlik_kapisi(s_val, k_val, kendi["n_val"])

    o = _ozet()[_ozet()["magaza"] == m].iloc[0]
    return json.dumps({
        "magaza": m,
        "magaza_tipi": str(o["magaza_tipi"]),
        "tedarikci": str(o["tedarikci"]),
        "donem": "validasyon (2015-05-25..07-03) — karar aninda bilinen hata",
        "smape_val_yuzde": s_val,
        "zincir_ort_smape_yuzde": ort,
        "hata_siralamasi": f"{sira}/1115 (1 = en dusuk hata)",
        "kapsama_val_yuzde": k_val,
        "zincir_ort_kapsama_yuzde": (round(float(p["kapsama_val"].mean()), 1)
                                     if p["kapsama_val"].notna().any() else None),
        "hedef_kapsama_yuzde": 80.0,
        "tahmin_zayif_mi": bool(s_val > ort),
        "belirsizlik_kapisi_kapali_mi": kapi["zayif"],
        "kapi_sebepleri": kapi["sebepler"],
        "geriye_donuk_test": {
            "aciklama": ("Test donemi (07-04..07-31) gerceklesen hatasi. "
                         "Karar kuralinda KULLANILMAZ; yalniz model "
                         "degerlendirmesi icindir."),
            "smape_test_yuzde": float(kendi["smape_test"]),
            "kapsama_test_yuzde": float(kendi["kapsama_test"]),
        },
        "yorum": (
            "Bu magazada model zincir ortalamasindan daha ZAYIF; "
            "tahmin araligini genis yorumla."
            if s_val > ort else
            "Bu magazada model zincir ortalamasindan daha IYI calisiyor."),
        "not": ("Kapsama, gercek degerin p10-p90 bandi icinde kalma oranidir. "
                "Hedef %80."),
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 6 — raf omru durumu
# ===========================================================================
def raf_omru_durumu(tarih: str, n: int = 10) -> str:
    """Belirtilen tarihte hangi magazalarda malin raf omru dolmak uzere
    oldugunu ve o gun ne kadar fire verilecegini listeler. Fire, raf omrunun
    son gununu dolduran ve satilamayan stoktur. Tarih YYYY-MM-DD."""
    t, h = _tarih_ayristir(tarih, "tarih")
    if h:
        return h
    k = _n_ayristir(n)

    p = _projeksiyon()
    gun = p[(p["mod"] == "politika") & (p["talep_senaryo"] == "p50")
            & (p["tarih"] == t)]
    if gun.empty:
        return _hata(f"{t.date()} icin projeksiyon satiri bulunamadi.")

    riskli = gun[gun["fire"] > 0.5].copy()
    riskli = riskli.merge(_ozet()[["magaza", "tedarikci", "tipik_gunluk_talep_adet"]],
                          on="magaza")
    riskli["fire_gun_karsiligi"] = (
        riskli["fire"] / riskli["tipik_gunluk_talep_adet"].clip(lower=1e-6)).round(2)
    ilk = riskli.sort_values("fire", ascending=False).head(k)

    raf_dagilim = gun.groupby("kalan_raf")["magaza"].size().to_dict()
    return json.dumps({
        "tarih": str(t.date()),
        "senaryo": "politika modu, p50 talep",
        "fire_veren_magaza_sayisi_TOPLAM": len(riskli),
        "toplam_fire_adet": round(float(gun["fire"].sum()), 1),
        "toplam_fire_maliyet_eur": round(float(gun["fire"].sum() * BIRIM_MALIYET), 1),
        "kalan_raf_omru_dagilimi_magaza_sayisi": {
            f"{int(k2)}_gun": int(v) for k2, v in sorted(raf_dagilim.items())},
        "bu_ciktida_listelenen": len(ilk),
        "en_cok_fire_verenler": [{
            "magaza": int(r.magaza), "tedarikci": r.tedarikci,
            "fire_adet": round(float(r.fire), 1),
            "fire_maliyet_eur": round(float(r.fire) * BIRIM_MALIYET, 1),
            "fire_gun_karsiligi": float(r.fire_gun_karsiligi),
            "kalan_raf_omru_gun": int(r.kalan_raf),
            "acilis_stok_adet": round(float(r.acilis), 1),
        } for r in ilk.itertuples()],
        "not": ("fire_gun_karsiligi, fire miktarinin magazanin kac gunluk "
                "talebine denk geldigini gosterir. 0.40'i asan magazalar "
                "satinalma politikasi Madde 6.1'e gore onaya gider."),
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 7 — dagitim plani (kisitli stok onceliklendirmesi)
# ===========================================================================
def dagitim_plani(toplam_stok: float, tarih: str, n: int = 10) -> str:
    """Elde belirli bir miktar stok varsa bunun magazalara nasil
    dagitilacagini hesaplar. Dagitim, magazalarin o gunku karsilanmamis
    ihtiyaci oraninda yapilir; yani kitlik herkese esit pay edilir.
    toplam_stok adet cinsinden, tarih YYYY-MM-DD bicimindedir."""
    miktar, h = _sayi_ayristir(toplam_stok, "toplam_stok")
    if h:
        return h
    t, h = _tarih_ayristir(tarih, "tarih")
    if h:
        return h
    k = _n_ayristir(n)

    p = _projeksiyon()
    gun = p[(p["mod"] == "siparis_yok") & (p["talep_senaryo"] == "p50")
            & (p["tarih"] == t) & (p["stoksuz"] > 0.5)].copy()
    if gun.empty:
        return json.dumps({
            "tarih": str(t.date()), "eldeki_stok_adet": miktar,
            "not": "Bu tarihte karsilanmamis ihtiyaci olan magaza yok; "
                   "dagitima gerek yok.",
        }, ensure_ascii=False)

    ihtiyac_toplam = float(gun["stoksuz"].sum())
    oran = min(1.0, miktar / ihtiyac_toplam)
    gun["pay_adet"] = (gun["stoksuz"] * oran).round(1)
    gun["karsilanma_yuzde"] = round(oran * 100, 1)
    gun = gun.merge(_ozet()[["magaza", "tedarikci", "aciliyet"]], on="magaza")
    gun = _perf_ekle(gun)
    ilk = gun.sort_values("pay_adet", ascending=False).head(k)

    return json.dumps({
        "tarih": str(t.date()),
        "eldeki_stok_adet": miktar,
        "toplam_ihtiyac_adet": round(ihtiyac_toplam, 1),
        "ihtiyac_duyan_magaza_sayisi_TOPLAM": len(gun),
        "karsilanma_orani_yuzde": round(oran * 100, 1),
        "yetersizlik_adet": round(max(0.0, ihtiyac_toplam - miktar), 1),
        "dagitim_kurali": "her magazaya ihtiyaci oraninda esit yuzde",
        "bu_ciktida_listelenen": len(ilk),
        "dagitim": [{
            "magaza": int(r.magaza), "tedarikci": r.tedarikci,
            "aciliyet": r.aciliyet,
            "ihtiyac_adet": round(float(r.stoksuz), 1),
            "gonderilecek_adet": float(r.pay_adet),
            "tahmin_hatasi_smape_yuzde": float(r.smape_val),
        } for r in ilk.itertuples()],
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 8 — magaza karsilastir
# ===========================================================================
def magaza_karsilastir(magaza_a: int, magaza_b: int,
                       baslangic: str = "", bitis: str = "") -> str:
    """Iki magazayi talep, stok, risk, siparis ve tahmin dogrulugu
    bakimindan yan yana karsilastirir. Tarihler bos birakilirsa tum tahmin
    penceresi kullanilir."""
    a, h = _magaza_ayristir(magaza_a)
    if h:
        return h
    b, h = _magaza_ayristir(magaza_b)
    if h:
        return h
    if a == b:
        return _hata("Ayni magaza iki kez verildi; farkli iki magaza sec.")
    b1, b2, h = _aralik_ayristir(baslangic, bitis)
    if h:
        return h

    tah = _tahmin()
    tah = tah[(tah["senaryo"] == "planli") & (tah["tarih"] >= b1)
              & (tah["tarih"] <= b2)]
    pro = _projeksiyon()
    pro = pro[(pro["tarih"] >= b1) & (pro["tarih"] <= b2)]
    ozet, perf, stok = _ozet(), _perf(), _stok()

    def blok(m: int) -> dict:
        o = ozet[ozet["magaza"] == m].iloc[0]
        pf = perf[perf["magaza"] == m].iloc[0]
        s = stok[stok["magaza"] == m].iloc[0]
        tm = tah[tah["magaza"] == m]
        taban = pro[(pro["magaza"] == m) & (pro["mod"] == "siparis_yok")
                    & (pro["talep_senaryo"] == "p50")]
        pol = pro[(pro["magaza"] == m) & (pro["mod"] == "politika")
                  & (pro["talep_senaryo"] == "p50")]
        return {
            "magaza": m,
            "magaza_tipi": str(o["magaza_tipi"]),
            "urun_yelpazesi": str(s["urun_yelpazesi"]),
            "tedarikci": str(o["tedarikci"]),
            "ort_gunluk_ciro_eur": round(float(tm["p50_eur"].mean()), 1),
            "ort_gunluk_talep_adet": round(float(tm["p50_adet"].mean()), 1),
            "mevcut_stok_adet": round(float(o["stok_toplam_adet"]), 1),
            "stok_gun_karsiligi": round(float(o["stok_gun_karsiligi"]), 2),
            "aciliyet": str(o["aciliyet"]),
            "siparissiz_stoksuz_adet": round(float(taban["stoksuz"].sum()), 1),
            "politika_siparis_adet": round(float(pol["siparis"].sum()), 1),
            "politika_fire_adet": round(float(pol["fire"].sum()), 1),
            "net_kazanc_eur": round(float(o["net_kazanc_eur"]), 1),
            "smape_val_yuzde": float(pf["smape_val"]),
            "kapsama_val_yuzde": (None if pd.isna(pf["kapsama_val"])
                                  else float(pf["kapsama_val"])),
            "onay_sebebi": str(o["onay_sebep"]),
        }

    ka, kb = blok(a), blok(b)
    farklar = {}
    for alan in ("ort_gunluk_talep_adet", "stok_gun_karsiligi",
                 "politika_siparis_adet", "smape_val_yuzde", "net_kazanc_eur"):
        farklar[alan] = round(ka[alan] - kb[alan], 2)

    return json.dumps({
        "aralik": f"{b1.date()} - {b2.date()}",
        "magaza_a": ka,
        "magaza_b": kb,
        "fark_a_eksi_b": farklar,
        "not": "fark_a_eksi_b pozitifse A daha buyuk demektir.",
    }, ensure_ascii=False)


# ===========================================================================
# TOOL 9 — promosyon senaryosu
# ===========================================================================
def promo_senaryosu(magaza: int, tarih: str = "") -> str:
    """Bir magazada promosyon yapilmasi durumunda talebin ne kadar
    artacagini gosterir. Tahmin tablosunda ayni gun icin uc senaryo vardir:
    planli (mevcut takvim), promo_yok, promo_var. Tarih bos birakilirsa tum
    pencere ortalamasi doner."""
    m, h = _magaza_ayristir(magaza)
    if h:
        return h

    tah = _tahmin()
    tah = tah[(tah["magaza"] == m) & (tah["acik"] == 1)]
    if tarih:
        t, h = _tarih_ayristir(tarih, "tarih")
        if h:
            return h
        tah = tah[tah["tarih"] == t]
        if tah.empty:
            return json.dumps({
                "magaza": m, "tarih": str(t.date()),
                "not": "Magaza bu tarihte KAPALI; promosyon senaryosu yok.",
            }, ensure_ascii=False)
        etiket = str(t.date())
    else:
        etiket = "tum pencere ortalamasi"

    def q(senaryo, kolon="p50_adet"):
        alt = tah[tah["senaryo"] == senaryo]
        return round(float(alt[kolon].mean()), 1) if len(alt) else 0.0

    yok, var, planli = q("promo_yok"), q("promo_var"), q("planli")
    artis = round((var / yok - 1) * 100, 1) if yok > 0 else 0.0
    ek_adet = round(var - yok, 1)

    o = _ozet()[_ozet()["magaza"] == m].iloc[0]
    return json.dumps({
        "magaza": m,
        "donem": etiket,
        "gunluk_talep_promosyonsuz_adet": yok,
        "gunluk_talep_promosyonlu_adet": var,
        "planli_takvime_gore_adet": planli,
        "promo_artisi_yuzde": artis,
        "ek_talep_adet": ek_adet,
        "ek_ciro_eur": round(ek_adet * BIRIM_FIYAT, 1),
        "mevcut_stok_adet": round(float(o["stok_toplam_adet"]), 1),
        "stok_gun_karsiligi": round(float(o["stok_gun_karsiligi"]), 2),
        "tedarikci": str(o["tedarikci"]),
        "smape_val_yuzde": float(_perf()[_perf()["magaza"] == m].iloc[0]["smape_val"]),
        "uyari": ("Promosyon karari verilirken tedarikci lead-time'i ve "
                  "minimum siparis miktari kontrol edilmelidir; raf omru "
                  "3 gundur. Promosyon doneminde emniyet stogu orani "
                  "politikaya gore %15'ten %25'e cikar."),
    }, ensure_ascii=False)


# ===========================================================================
TOOL_FONKSIYONLARI = [tukenme_riski, sevkiyat_plani_getir, en_cok_ihtiyac_duyan,
                      segment_ozeti, model_performansi, raf_omru_durumu,
                      dagitim_plani, magaza_karsilastir, promo_senaryosu]


if __name__ == "__main__":
    bas, bit = _pencere()
    print(f"pencere: {bas.date()} - {bit.date()} | "
          f"demo_bugun: {_meta().get('demo_bugun', '?')} | "
          f"zincir ort sMAPE: {_zincir_ort_smape()}%")
    testler = [
        ("tukenme_riski (ilk 14 gun, 3)",
         lambda: tukenme_riski("2015-07-04", "2015-07-17", 3)),
        ("sevkiyat_plani_getir (07-06)", lambda: sevkiyat_plani_getir("2015-07-06", 3)),
        ("en_cok_ihtiyac_duyan (3)", lambda: en_cok_ihtiyac_duyan(n=3)),
        ("segment_ozeti (tum tipler)", lambda: segment_ozeti()),
        ("model_performansi (530)", lambda: model_performansi(530)),
        ("raf_omru_durumu (07-10)", lambda: raf_omru_durumu("2015-07-10", 3)),
        ("dagitim_plani (50.000 adet, 07-10)",
         lambda: dagitim_plani(50000, "2015-07-10", 3)),
        ("magaza_karsilastir (530 vs 1)", lambda: magaza_karsilastir(530, 1)),
        ("promo_senaryosu (530)", lambda: promo_senaryosu(530)),
        ("promo_senaryosu (530, 07-12 pazar)", lambda: promo_senaryosu(530, "2015-07-12")),
        ("DAYANIKLILIK: olmayan magaza", lambda: model_performansi(1116)),
        ("DAYANIKLILIK: pencere disi", lambda: raf_omru_durumu("2015-06-26")),
        ("DAYANIKLILIK: negatif stok", lambda: dagitim_plani(-5, "2015-07-10")),
        ("DAYANIKLILIK: ayni magaza", lambda: magaza_karsilastir(5, 5)),
        ("DAYANIKLILIK: sacma tarih", lambda: tukenme_riski("Migros", "")),
    ]
    for ad, f in testler:
        print(f"\n{'='*70}\n{ad}\n{'='*70}")
        try:
            c = f()
            print(c[:800] + (" ...[kisaltildi]" if len(c) > 800 else ""))
        except Exception as e:
            print(f"COKTU: {type(e).__name__}: {e}")