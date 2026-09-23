"""
stok_simulasyon.py — FIFO stok dongusu, fire hesabi, politika ve TEDARIKCI kiyasi.

BULGU: kisitli senaryoda "ortalama" ve "newsvendor" politikalari neredeyse ayni
sonucu veriyordu. Sebep: Schnellware'in 500 birimlik minimum siparisi ~4 gunluk
talebe denk geliyor, yani siparis miktarini politika degil SOZLESME belirliyor.
Baglayici kisit tahmin tarafinda degil, tedarik tarafinda. Bu yuzden simulasyon
tedarikci bazinda kosulur: her tedarikcinin kendi lead-time'i, minimum miktari,
teslimat gunleri, fiyat farki ve iskonto kademesi vardir. Parametreler RAG
belgelerindeki sozlesmelerle birebir aynidir.

MUHASEBE
    net_kar = ciro + kalan_stok_degeri - satinalma_maliyeti - baslangic_stok_degeri
Fire ayrica cezalandirilmaz; bedeli zaten satin alinip satilamayan malin
maliyetinde gizlidir (cift sayim olmaz). Stoksuz kalma, kaybedilen marj olarak
FIRSAT maliyeti sutununda ayrica raporlanir.

PARTI YASI: gunluk cikti "en_eski_yas" ve "kalan_raf_omru" kolonlarini da
icerir. Agent'in israf riski hesabi stok HACMINE degil, elde duran partinin
YASINA bakmak zorundadir; aksi halde az ama bayat stok "risksiz" gorunur.

Calistirma:
    python stok_simulasyon.py              # tedarikci x politika tam kiyas
    python stok_simulasyon.py --tarama     # raf omru duyarliligi
    python stok_simulasyon.py --detay Schnellware
"""

import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

BURASI = Path(__file__).resolve().parent
KOK = BURASI / "Rossman" if (BURASI / "Rossman").is_dir() else BURASI

# ---------------------------------------------------------------------------
# VARSAYIMLAR
# ---------------------------------------------------------------------------
TAZE_ORAN = 0.20        # cironun taze kategori payi
BIRIM_FIYAT = 8.0       # satis fiyati (EUR)
MARJ_ORANI = 0.25       # brut marj
RAF_OMRU_GUN = 3        # taze kategori raf omru
GUVEN_ARALIGI = 0.90

BIRIM_KAR = BIRIM_FIYAT * MARJ_ORANI            # Cu
BIRIM_MALIYET = BIRIM_FIYAT * (1 - MARJ_ORANI)  # Co (baz alis fiyati)
KRITIK_ORAN = BIRIM_KAR / (BIRIM_KAR + BIRIM_MALIYET)

_nd = NormalDist()
Z_ARALIK = _nd.inv_cdf(0.5 + GUVEN_ARALIGI / 2)
Z_KRITIK = _nd.inv_cdf(KRITIK_ORAN)

# ---------------------------------------------------------------------------
# TEDARIKCILER — RAG belgeleriyle birebir ayni kosullar
# ---------------------------------------------------------------------------
TEDARIKCILER = {
    "Schnellware": {
        "lead": 2, "min": 500, "gunler": {0, 2, 4}, "carpan": 1.00, "tavan": 8000,
        "iskonto": [(5000, 0.08), (2000, 0.05)],
        "not": "varsayilan tedarikci, Pzt/Car/Cum",
    },
    "Nordmann": {
        "lead": 5, "min": 1500, "gunler": {0, 1, 2, 3, 4}, "carpan": 0.93, "tavan": None,
        "iskonto": [(3000, 0.10)],
        "not": "%7 ucuz ama 5 gun lead-time",
    },
    "ExpressLog": {
        "lead": 1, "min": 200, "gunler": {0, 1, 2, 3, 4, 5}, "carpan": 1.12, "tavan": 3000,
        "iskonto": [],
        "not": "%12 pahali, 1 gun, dusuk minimum",
    },
    "Rheinland": {
        "lead": 3, "min": 1000, "gunler": {1, 3}, "carpan": 1.00, "tavan": None,
        "iskonto": [], "promo_lead": 2, "promo_min": 500, "promo_iskonto": 0.06,
        "not": "promosyonda lead 2 gun, min 500, %6 iskonto",
    },
}


def iskonto_uygula(ted: dict, miktar: float, promo: bool) -> float:
    """Birim alis fiyatini dondurur."""
    carpan = ted["carpan"]
    if promo and "promo_iskonto" in ted:
        return BIRIM_MALIYET * carpan * (1 - ted["promo_iskonto"])
    for esik, oran in ted["iskonto"]:
        if miktar >= esik:
            return BIRIM_MALIYET * carpan * (1 - oran)
    return BIRIM_MALIYET * carpan


# ---------------------------------------------------------------------------
def veri_hazirla(kaynak: Path) -> pd.DataFrame:
    if not kaynak.exists():
        sys.exit(f"HATA: {kaynak} yok. Once prophet_model.py calistir.")
    df = pd.read_csv(kaynak, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    df["talep_tahmin"] = df["yhat"] * TAZE_ORAN / BIRIM_FIYAT
    df["talep_gercek"] = df["gercek"] * TAZE_ORAN / BIRIM_FIYAT
    df["sigma"] = ((df["yhat_upper"] - df["yhat_lower"]) * TAZE_ORAN
                   / BIRIM_FIYAT / (2 * Z_ARALIK))
    df["haftagunu"] = df["ds"].dt.dayofweek
    return df


def pencere_hedefi(df: pd.DataFrame, bas: int, son: int, politika: str) -> float:
    pencere = df.iloc[bas:son]
    if pencere.empty:
        return 0.0
    ort = pencere["talep_tahmin"].sum()
    if politika == "ortalama":
        return ort
    sigma = (pencere["sigma"] ** 2).sum() ** 0.5
    return max(0.0, ort + Z_KRITIK * sigma)


def simule_et(df: pd.DataFrame, tedarikci_adi: str, politika: str,
              raf_omru: int = RAF_OMRU_GUN) -> tuple[pd.DataFrame, dict]:
    ted = TEDARIKCILER[tedarikci_adi]
    teslimat_idx = [i for i in range(len(df)) if df.iloc[i]["haftagunu"] in ted["gunler"]]

    partiler = [[0, float(df.iloc[0]["talep_tahmin"]) * 2]]
    baslangic_deger = partiler[0][1] * BIRIM_MALIYET
    yoldakiler: dict[int, float] = {}
    satirlar = []
    satinalma_toplam = 0.0

    for gun in range(len(df)):
        s = df.iloc[gun]
        promo = bool(s["promo"])
        lead = ted.get("promo_lead", ted["lead"]) if promo else ted["lead"]
        minimum = ted.get("promo_min", ted["min"]) if promo else ted["min"]

        gelen = yoldakiler.pop(gun, 0.0)
        if gelen > 0:
            partiler.append([gun, gelen])
        acilis = sum(p[1] for p in partiler)

        # satis (FIFO — en eski partiden)
        kalan = float(s["talep_gercek"])
        for p in partiler:
            if kalan <= 0:
                break
            dus = min(p[1], kalan)
            p[1] -= dus
            kalan -= dus
        satilan = float(s["talep_gercek"]) - kalan
        stoksuz = kalan

        # raf omru — suresi dolan parti fire
        fire = sum(p[1] for p in partiler if gun - p[0] >= raf_omru)
        partiler = [p for p in partiler if p[1] > 1e-9 and gun - p[0] < raf_omru]
        eldeki = sum(p[1] for p in partiler)

        # elde duran en eski partinin yasi ve kalan raf omru
        en_eski_yas = (gun - min(p[0] for p in partiler)) if partiler else 0
        kalan_raf_omru = max(0, raf_omru - en_eski_yas)

        # siparis
        siparis, hedef, birim_alis = 0.0, 0.0, 0.0
        varis = gun + lead
        if varis in teslimat_idx:
            sonrakiler = [i for i in teslimat_idx if i > varis]
            hedef = pencere_hedefi(df, varis, sonrakiler[0] if sonrakiler else len(df),
                                   politika)
            siparis = max(0.0, hedef - eldeki - sum(yoldakiler.values()))
            if 0 < siparis < minimum:
                siparis = minimum
            if ted["tavan"]:
                siparis = min(siparis, ted["tavan"])
            if siparis > 0 and varis < len(df):
                birim_alis = iskonto_uygula(ted, siparis, promo)
                satinalma_toplam += siparis * birim_alis
                yoldakiler[varis] = yoldakiler.get(varis, 0.0) + siparis

        satirlar.append({
            "tarih": s["ds"].strftime("%Y-%m-%d"),
            "gun": ["Pzt", "Sal", "Car", "Per", "Cum", "Cmt", "Paz"][int(s["haftagunu"])],
            "promo": int(promo),
            "talep_gercek": round(float(s["talep_gercek"]), 1),
            "acilis_stok": round(acilis, 1),
            "satilan": round(satilan, 1),
            "stoksuz": round(stoksuz, 1),
            "fire": round(fire, 1),
            "kapanis_stok": round(eldeki, 1),
            "en_eski_yas": en_eski_yas,
            "kalan_raf_omru": kalan_raf_omru,
            "siparis": round(siparis, 1),
            "birim_alis": round(birim_alis, 2),
        })

    sonuc = pd.DataFrame(satirlar)
    kalan_deger = sum(p[1] for p in partiler) * BIRIM_MALIYET
    ciro = sonuc["satilan"].sum() * BIRIM_FIYAT
    ozet = {
        "tedarikci": tedarikci_adi,
        "politika": politika,
        "fire_birim": round(sonuc["fire"].sum(), 1),
        "stoksuz_birim": round(sonuc["stoksuz"].sum(), 1),
        "firsat_maliyeti": round(sonuc["stoksuz"].sum() * BIRIM_KAR, 1),
        "satinalma": round(satinalma_toplam, 1),
        "ciro": round(ciro, 1),
        "net_kar": round(ciro + kalan_deger - satinalma_toplam - baslangic_deger, 1),
        "ort_stok": round(sonuc["kapanis_stok"].mean(), 1),
        "siparis_sayisi": int((sonuc["siparis"] > 0).sum()),
    }
    return sonuc, ozet


# ---------------------------------------------------------------------------
def parametreler() -> None:
    print("=" * 92)
    print("PARAMETRELER (varsayim)")
    print("=" * 92)
    print(f"  Taze pay: %{TAZE_ORAN*100:.0f} | Birim fiyat: {BIRIM_FIYAT:.2f} EUR "
          f"| Marj: %{MARJ_ORANI*100:.0f} | Raf omru: {RAF_OMRU_GUN} gun")
    print(f"  Cu: {BIRIM_KAR:.2f} EUR | Co: {BIRIM_MALIYET:.2f} EUR "
          f"| Kritik oran: {KRITIK_ORAN:.2f} | newsvendor z: {Z_KRITIK:+.4f}")


def tam_kiyas(df: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("TEDARIKCI x POLITIKA KIYASI")
    print("=" * 92)
    print(f"{'Tedarikci':<13}{'Politika':<12}{'Fire':>8}{'Stoksuz':>9}"
          f"{'Satinalma':>12}{'Ciro':>10}{'NET KAR':>11}{'Ort.stok':>10}")
    print("-" * 92)

    ozetler = []
    for ad in TEDARIKCILER:
        for politika in ("ortalama", "newsvendor"):
            detay, o = simule_et(df, ad, politika)
            ozetler.append(o)
            detay.to_csv(KOK / f"stok_sim_{ad}_{politika}.csv",
                         index=False, encoding="utf-8-sig")
            print(f"{ad:<13}{politika:<12}{o['fire_birim']:>8,.0f}"
                  f"{o['stoksuz_birim']:>9,.0f}{o['satinalma']:>12,.0f}"
                  f"{o['ciro']:>10,.0f}{o['net_kar']:>11,.0f}{o['ort_stok']:>10,.0f}")
        print("-" * 92)

    en_iyi = max(ozetler, key=lambda x: x["net_kar"])
    en_kotu = min(ozetler, key=lambda x: x["net_kar"])
    print(f"\nEN IYI : {en_iyi['tedarikci']} / {en_iyi['politika']} "
          f"-> net kar {en_iyi['net_kar']:,.0f} EUR  "
          f"({TEDARIKCILER[en_iyi['tedarikci']]['not']})")
    print(f"EN KOTU: {en_kotu['tedarikci']} / {en_kotu['politika']} "
          f"-> net kar {en_kotu['net_kar']:,.0f} EUR")
    print(f"Aradaki fark: {en_iyi['net_kar'] - en_kotu['net_kar']:,.0f} EUR")

    pd.DataFrame(ozetler).to_csv(KOK / "tedarikci_kiyas.csv",
                                 index=False, encoding="utf-8-sig")
    print(f"\nOzet tablo: {KOK}/tedarikci_kiyas.csv")


def tarama(df: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("RAF OMRU DUYARLILIGI — her tedarikcinin en iyi politikasiyla net kar (EUR)")
    print("=" * 92)
    print(f"{'Raf omru':<10}" + "".join(f"{ad:>18}" for ad in TEDARIKCILER))
    print("-" * 92)
    for raf in (2, 3, 4, 5, 7):
        satir = f"{raf:<10}"
        for ad in TEDARIKCILER:
            en_iyi = max(
                simule_et(df, ad, p, raf_omru=raf)[1]["net_kar"]
                for p in ("ortalama", "newsvendor")
            )
            satir += f"{en_iyi:>18,.0f}"
        print(satir)


if __name__ == "__main__":
    df = veri_hazirla(KOK / "prophet_forecast.csv")
    parametreler()

    if "--tarama" in sys.argv:
        tarama(df)
    elif "--detay" in sys.argv:
        ad = sys.argv[sys.argv.index("--detay") + 1]
        detay, o = simule_et(df, ad, "newsvendor")
        print(f"\n{ad} / newsvendor — gunluk detay\n")
        print(detay.to_string(index=False))
        print("\nOzet:", o)
    else:
        tam_kiyas(df)