"""
karar_tool.py — Magaza bazli YAPISAL KARAR uretimi (agent_v7 icin 11. tool)

NEDEN AYRI DOSYA
    tools_toplu.py routing eval'de 48/50 veren dosya; ona dokunmuyoruz.
    Bu modul onun veri yukleyicilerini (_tahmin/_projeksiyon/_ozet/_perf)
    yeniden kullanir, kendi veri kopyasini tutmaz.

MIMARI ILKE (agent_v6'dan devralindi)
    LLM sayi URETMEZ. Bu tool'un dondurdugu her sayi deterministik olarak
    hesaplanir; agent yalnizca tool'u DOGRU PARAMETRELERLE cagirmaktan ve
    sonucu yorumlamaktan sorumludur. Karar setini de kural motoru belirler.

KARAR SETI
    siparis_ver | bekle | indirim_uygula | insana_sor

SIPARIS KURALI
    Siparis karari politika motorunun (projeksiyon_uret.kos) O GUN verdigi
    siparise baglanir. O motor tedarikci lead-time'ini,
    teslimat takvimini, yoldaki siparisi ve newsvendor hedefini zaten
    hesaba katar. Bugun verilen siparis bugunu degil, varis gununu kurtarir;
    bu yuzden "bugun stok yetmiyor mu" sorusu karar degil ACILIYET bilgisidir.

    Neden: siparisi mod='siparis_yok' + p90 stoksuzluguna baglayan onceki
    kural, bu mod hic siparis vermedigi icin 4. gunden itibaren her acik
    magazada kendiliginden tetikleniyordu (teshis_siparis.py).

FIRE SINYALI — BEKLENEN FIRE
    Yalniz plana (p50) bakan bir sinyal, tahmin HATASINDAN dogan fireyi
    (fazla tahmin -> fazla siparis -> bozulan mal) goremez; plan "talep
    tahmin kadar gelecek" varsayar. Sinyal bu yuzden modelin belirsizlik
    bandini kullanir:
        beklenen_fire = 0.3 x fire(p10) + 0.4 x fire(p50) + 0.3 x fire(p90)
    Bu, uc kuantilden beklenen deger icin bilinen Swanson kuralidir; agirlik
    veriye bakilarak SECILMEDI. Esik (0.40, politika Madde 6.1) degismedi.

CIKTI SEMASI
    n8n_gonder.py'nin webhook payload'i ile AYNI alan isimleri kullanilir.
    Boylece Google Sheets'teki mevcut kolon eslemesi degistirilmeden calisir.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from projeksiyon_uret import TEDARIKCILER
from tools_toplu import (
    BIRIM_FIYAT,
    BIRIM_KAR,
    ROSSMAN,
    _hata,
    _magaza_ayristir,
    _ozet,
    _perf,
    _projeksiyon,
    _tahmin,
    _tarih_ayristir,
)

# ===========================================================================
# SABITLER — hepsi belgelenmis varsayim, README'de gerekcesiyle yer alir
# ===========================================================================
ONAY_ESIGI_EUR = 3000.0      # bu tutari asan net etki insan onayina gider
FIRE_ORAN_ESIGI = 0.40       # fire / gunluk talep; satinalma politikasi Madde 6.1
# Karar kurali DEGIL; yalniz aciliyet ve raporlama icin tasinir.
STOK_GUN_ALT_ESIK = 1.5
# BELIRSIZLIK KAPISI — yalniz KARAR ANINDA BILINEN metriklerle beslenir.
# Karar gunu 2015-07-03; test donemi (07-04..07-31) metrikleri o gun
# bilinemez. Kapi validasyon donemi (05-25..07-03) metriklerini kullanir;
# test metrikleri yalniz geriye donuk degerlendirme icin ciktida kalir.
SMAPE_BELIRSIZ_ESIK = 20.0   # val sMAPE (%) bunun ustundeyse karar insana sorulur
KAPSAMA_HEDEF = 80.0         # p10-p90 bandinin nominal kapsamasi (%)
# Kapsama kapisi sabit %70 esigi yerine TEK YONLU BINOM TESTI kullanir:
# 28 gozlemde kusursuz kalibre bir magaza bile %9 olasilikla %70'in altina
# duser; sabit esik gurultuyu model zayifligi sanar. Kapi ancak kapsama
# nominalin ALTINDA ve bu fark sansla aciklanamiyorsa (p < 0.05) kapanir.
KAPSAMA_P_ESIK = 0.05
N_VAL_VARSAYILAN = 28        # tahmin tablosunda magaza_n_val yoksa (temkinli)

# Beklenen deger icin Swanson kurali (p10/p50/p90 agirliklari). Veriye
# bakilarak ayarlanmaz; uc kuantilden ortalama tahmini icin standart yaklasim.
SWANSON = {"p10": 0.3, "p50": 0.4, "p90": 0.3}

# Indirim elastikiyeti: %X indirim, talebi (1 + ELASTIKIYET * X) katina cikarir.
# Varsayim — gercek fiyat testi verisi yok, README'de acikca belirtilir.
# NOT: dogrusal erime modelinde getiri s*e*o*(1-o)*fiyat'tir ve maksimumu
# e<=2 icin her zaman o=0.5'tedir; %30'u yalniz ardisik-gun kisiti sectirir.
ELASTIKIYET = 1.5
INDIRIM_SECENEKLERI = (0.30, 0.50)

# "Pes pese iki gun %50 indirim uygulanamaz" kisiti icin magaza bazli gecmis
GECMIS_DOSYA = ROSSMAN / "indirim_gecmisi_magaza.json"


# ===========================================================================
# INDIRIM GECMISI — politika kisiti gercek veriye dayansin
# ===========================================================================
def _gecmis_oku() -> dict:
    if GECMIS_DOSYA.exists():
        try:
            return json.loads(GECMIS_DOSYA.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _gecmise_yaz(magaza: int, tarih: str, oran: float) -> None:
    g = _gecmis_oku()
    g.setdefault(str(magaza), {})[tarih] = oran
    GECMIS_DOSYA.parent.mkdir(parents=True, exist_ok=True)
    GECMIS_DOSYA.write_text(json.dumps(g, ensure_ascii=False, indent=2),
                            encoding="utf-8")


def _onceki_gun_indirimi(magaza: int, t: pd.Timestamp) -> float:
    """Onceki TAKVIM gununde bu magazada uygulanan indirim orani."""
    onceki = str((t - pd.Timedelta(days=1)).date())
    return float(_gecmis_oku().get(str(magaza), {}).get(onceki, 0.0))


# ===========================================================================
# BELIRSIZLIK KAPISI — tek yerde; test_karar_tool ve karar_degerlendirme de
# ayni fonksiyonu kullanir, kural iki yerde ayri ayri yazilmaz.
# ===========================================================================
def _binom_alt_kuyruk(x: int, n: int, p: float) -> float:
    """P(X <= x), X ~ Binom(n, p). scipy bagimliligi olmadan."""
    if n <= 0:
        return 1.0
    x = max(0, min(int(x), n))
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(x + 1)))


def belirsizlik_kapisi(smape_val: float | None, kapsama_val: float | None,
                       n_val: int | None = None) -> dict:
    """Model bu magazada karar icin yeterince guvenilir mi?

    Doner: {"zayif": bool, "sebepler": [...], "kapsama_p": float | None}
    Eksik metrik kapiyi TETIKLEMEZ ama sebep listesinde belirtilir.
    """
    def _sayi(x):
        # None, NaN ve pandas NA (Int64 kolonlari) ayni sekilde "yok" sayilir
        try:
            return None if x is None or pd.isna(x) else float(x)
        except (TypeError, ValueError):
            return None

    smape_val, kapsama_val, n_val = _sayi(smape_val), _sayi(kapsama_val), _sayi(n_val)
    sebepler: list[str] = []
    kapsama_p = None
    zayif = False

    if smape_val is not None:
        if smape_val > SMAPE_BELIRSIZ_ESIK:
            zayif = True
            sebepler.append(f"val sMAPE %{smape_val:.1f} > %{SMAPE_BELIRSIZ_ESIK:.0f}")

    if kapsama_val is not None:
        n = int(n_val) if n_val else N_VAL_VARSAYILAN
        icinde = int(round(kapsama_val / 100.0 * n))
        kapsama_p = _binom_alt_kuyruk(icinde, n, KAPSAMA_HEDEF / 100.0)
        if kapsama_val < KAPSAMA_HEDEF and kapsama_p < KAPSAMA_P_ESIK:
            zayif = True
            sebepler.append(f"val kapsama %{kapsama_val:.1f}, nominal "
                            f"%{KAPSAMA_HEDEF:.0f}'in anlamli altinda "
                            f"(binom p={kapsama_p:.3f}, n={n})")

    return {"zayif": zayif, "sebepler": sebepler,
            "kapsama_p": None if kapsama_p is None else round(kapsama_p, 4)}


# ===========================================================================
# GUVEN ARALIGI — kolon adlari surumden surume degisebiliyor, toleransli oku
# ===========================================================================
def _guven_araligi(satir: pd.Series, p50: float) -> tuple[float, float]:
    """Tahmin tablosundan alt/ust siniri cikarir. Kolon yoksa sMAPE'den turetir."""
    for alt_ad, ust_ad in (("p10_adet", "p90_adet"),
                           ("alt_adet", "ust_adet"),
                           ("yhat_lower_adet", "yhat_upper_adet")):
        if alt_ad in satir.index and ust_ad in satir.index:
            alt, ust = float(satir[alt_ad]), float(satir[ust_ad])
            if ust > alt:
                return round(alt, 1), round(ust, 1)

    # Yedek yol: magazanin VAL sMAPE'i kadar simetrik bant varsay
    # (test sMAPE karar aninda bilinmez — sizinti olurdu)
    smape = float(satir.get("magaza_smape_val", 15.0) or 15.0) / 100.0
    return round(p50 * (1 - smape), 1), round(p50 * (1 + smape), 1)


# ===========================================================================
# INDIRIM SENARYOSU — deterministik net etki hesabi
# ===========================================================================
def _indirim_etkisi(fazla_stok: float, oran: float) -> dict:
    """Bir indirim oraninin EUR cinsinden net etkisini hesaplar.

    Referans senaryo: hic aksiyon alinmazsa fazla stok tamamen fire olur ve
    getirisi SIFIR'dir (mal zaten satin alinmistir — batik maliyet).
    Indirim senaryosunda bu stokun bir kismi indirimli fiyattan satilir.
    Dolayisiyla net etki = indirimle kurtarilan ciro.

    Erime orani varsayimi: min(1, ELASTIKIYET * oran)
        %30 indirim -> fazla stokun %45'i erir
        %50 indirim -> fazla stokun %75'i erir
    Gercek fiyat esneklik verisi olmadigi icin bu bir VARSAYIMDIR;
    README'de acikca belirtilir ve duyarlilik analizine konu edilir.
    """
    if fazla_stok <= 0:
        return {"oran": oran, "satilan": 0.0, "net_etki_eur": 0.0}

    erime_orani = min(1.0, ELASTIKIYET * oran)
    satilan = fazla_stok * erime_orani
    indirimli_fiyat = BIRIM_FIYAT * (1 - oran)
    net_etki = satilan * indirimli_fiyat

    return {
        "oran": oran,
        "satilan": round(satilan, 1),
        "indirimli_fiyat_eur": round(indirimli_fiyat, 2),
        "net_etki_eur": round(net_etki, 1),
    }


def _en_iyi_indirim(fazla_stok: float, yuzde50_serbest: bool) -> dict:
    adaylar = [_indirim_etkisi(fazla_stok, o) for o in INDIRIM_SECENEKLERI
               if yuzde50_serbest or o < 0.50]
    if not adaylar:
        return {"oran": 0.0, "satilan": 0.0, "net_etki_eur": 0.0}
    return max(adaylar, key=lambda d: d["net_etki_eur"])


# ===========================================================================
# ANA TOOL
# ===========================================================================
def karar_uret(magaza: int, tarih: str, uygula: bool = False) -> str:
    """Belirli bir magaza ve tarih icin YAPISAL STOK KARARI uretir.

    Kullanicinin bir magaza icin "ne yapmaliyim", "siparis vereyim mi",
    "karar ver", "aksiyon onerisi" gibi talepleri bu tool ile karsilanir.
    Karar seti: siparis_ver, bekle, indirim_uygula, insana_sor.
    Tum sayilar deterministik hesaplanir; tahmin, stok, fire, raf omru ve
    tedarikci bilgisi birlestirilerek gerekceli tek bir karar dondurulur.

    magaza: magaza numarasi (orn. 264)
    tarih : YYYY-MM-DD
    uygula: True ise uygulanan indirim gecmise yazilir (varsayilan False)
    """
    m, h = _magaza_ayristir(magaza)
    if h:
        return h
    t, h = _tarih_ayristir(tarih, "tarih")
    if h:
        return h

    # --- veri cekme -------------------------------------------------------
    ozet = _ozet()
    o_alt = ozet[ozet["magaza"] == m]
    if o_alt.empty:
        return _hata(f"{m} numarali magaza ozet tablosunda yok.")
    o = o_alt.iloc[0]

    pro = _projeksiyon()
    pol = pro[(pro["magaza"] == m) & (pro["mod"] == "politika")
              & (pro["talep_senaryo"] == "p50") & (pro["tarih"] == t)]
    if pol.empty:
        return _hata(f"{t.date()} icin magaza {m} projeksiyonu bulunamadi "
                     f"(magaza kapali olabilir veya tarih pencere disinda).")
    p = pol.iloc[0]

    # Uc talep senaryosunun o gunku satiri (politika modu, gercekci stok).
    def _senaryo(q):
        r = pro[(pro["magaza"] == m) & (pro["mod"] == "politika")
                & (pro["talep_senaryo"] == q) & (pro["tarih"] == t)]
        return None if r.empty else r.iloc[0]

    s10, s90 = _senaryo("p10"), _senaryo("p90")
    # Kotumser (p90) talepte BUGUN ne kadar eksik kalir. Karar degil
    # ACILIYET girdisidir: bugun verilen siparis bugun gelmez.
    stoksuz_p90 = float(s90["stoksuz"]) if s90 is not None else 0.0

    tah = _tahmin()
    tm = tah[(tah["magaza"] == m) & (tah["tarih"] == t)
             & (tah["senaryo"] == "planli")]
    if tm.empty:
        return _hata(f"{t.date()} icin magaza {m} tahmini bulunamadi.")
    ts = tm.iloc[0]
    p50 = round(float(ts["p50_adet"]), 1)
    alt, ust = _guven_araligi(ts, p50)
    bant = round(ust - alt, 1)
    belirsizlik = round(bant / p50, 3) if p50 > 0 else 1.0

    perf = _perf()
    pf = perf[perf["magaza"] == m]
    def _al(kolon):
        if pf.empty or kolon not in pf.columns:
            return None
        v = pf[kolon].iloc[0]
        return None if pd.isna(v) else float(v)

    # Karar ANINDA bilinen (val) metrikler -> kapi
    smape_val = _al("smape_val")
    kapsama_val = _al("kapsama_val")
    n_val = _al("n_val")
    # Geriye donuk (test) metrikler -> yalniz kayit/izleme, kararda KULLANILMAZ
    smape_test = _al("smape_test")
    kapsama_test = _al("kapsama_test")

    acilis = float(p["acilis"])
    # fire = BEKLENEN fire (Swanson). Plan (p50) ve dusuk talep (p10)
    # fireleri ayri alanlarda raporlanir.
    fire_plan = float(p["fire"])
    fire_p10 = float(s10["fire"]) if s10 is not None else fire_plan
    fire_p90 = float(s90["fire"]) if s90 is not None else fire_plan
    fire = (SWANSON["p10"] * fire_p10 + SWANSON["p50"] * fire_plan
            + SWANSON["p90"] * fire_p90)
    kalan_raf = int(p["kalan_raf"])
    politika_siparis = float(p.get("siparis", 0.0) or 0.0)
    gunluk_talep = max(float(o["tipik_gunluk_talep_adet"]), 1e-6)
    # O GUNE ait stok gun karsiligi. Magaza kapaliysa (p50=0) oran
    # TANIMSIZDIR; kucuk bir tabana bolmek sahte bir buyuk deger uretir ve
    # karar defterine yazilir. Bu yuzden None donulur.
    stok_gun = (acilis / p50) if p50 > 0 else None
    stok_gun_donem = float(o["stok_gun_karsiligi"])
    tedarikci = str(o["tedarikci"])
    lead = int(TEDARIKCILER.get(tedarikci, {}).get("lead", 0))
    varis = (t + pd.Timedelta(days=lead)).date()
    # Fire orani, satinalma politikasi Madde 6.1 ile AYNI paydayi kullanir.
    fire_oran = round(fire / gunluk_talep, 2)

    fire_riskli = fire_oran > FIRE_ORAN_ESIGI
    siparis_gunu = politika_siparis >= 1.0

    # --- KURAL MOTORU -----------------------------------------------------
    karar = "bekle"
    miktar = 0
    indirim_orani = 0.0
    net_etki = 0.0
    gerekceler: list[str] = []
    onerilen_aksiyon = ""
    cakisma = False

    yuzde50_serbest = _onceki_gun_indirimi(m, t) < 0.50

    # KAPALI GUN KONTROLU — diger tum kurallardan once.
    magaza_kapali = p50 <= 0
    if magaza_kapali:
        karar = "insana_sor"
        gerekceler.append(
            f"Magaza {t.date()} tarihinde KAPALI (tahmin 0 adet). Kapali "
            f"gunde siparis veya indirim uygulanamaz.")
        if fire > 0.5:
            gerekceler.append(
                f"Ancak elde {acilis:.0f} adet stok duruyor ve {fire:.0f} "
                f"adedi bu gun fire veriyor; magaza acilmadan once sevkiyat "
                f"veya baska magazaya aktarim degerlendirilmeli.")

    elif fire_riskli and siparis_gunu:
        # Iki kural ZIT yonde aksiyon istiyor: elde bozulan mal var (indirim)
        # ve politika motoru yeni mal istiyor (siparis). Eski surum sessizce
        # indirimi secip siparisi atliyordu; oysa siparis gelecekteki bir
        # pencereyi kurtarir. Celiskiyi insan cozer; iki oneri de korunur.
        sec = _en_iyi_indirim(fire, yuzde50_serbest)
        karar = "insana_sor"
        cakisma = True
        onerilen_aksiyon = (f"indirim_uygula (%{sec['oran'] * 100:.0f} indirim) + "
                            f"siparis_ver ({int(round(politika_siparis))} adet)")
        gerekceler.append(
            f"CELISKILI SINYAL: beklenen fire {fire:.0f} adet, gunluk talebin "
            f"{fire_oran:.2f} katina denk geliyor ({FIRE_ORAN_ESIGI} esigi "
            f"asildi) ve ayni gun politika motoru {politika_siparis:.0f} adet "
            f"siparis oneriyor (varis {varis}). Iki aksiyon da oneri olarak "
            f"tasiniyor.")

    elif fire_riskli:
        sec = _en_iyi_indirim(fire, yuzde50_serbest)
        karar = "indirim_uygula"
        indirim_orani = sec["oran"]
        net_etki = sec["net_etki_eur"]
        gerekceler.append(
            f"Beklenen fire {fire:.0f} adet (plan {fire_plan:.0f}, dusuk talepte "
            f"{fire_p10:.0f}); gunluk talebin {fire_oran:.2f} kati "
            f"({FIRE_ORAN_ESIGI} esigi asildi); kalan raf omru {kalan_raf} gun.")
        if not yuzde50_serbest:
            gerekceler.append("Onceki gun %50 uygulandigi icin politika geregi "
                              "%50 secenegi devre disi birakildi.")

    elif siparis_gunu:
        karar = "siparis_ver"
        miktar = int(round(politika_siparis))
        net_etki = round(miktar * BIRIM_KAR, 1)
        gerekceler.append(
            f"Politika motoru bugun {miktar} adet siparis oneriyor: tedarikci "
            f"{tedarikci}, teslim suresi {lead} gun, varis {varis}. Miktar, "
            f"varis penceresinin newsvendor hedefinden yoldaki siparis ve "
            f"ara talep dusulerek hesaplandi.")

    else:
        gerekceler.append(
            "Politika motoru bugun siparis ongormuyor (teslimat takvimi, "
            "yoldaki siparis veya yeterli stok).")
        if stoksuz_p90 > 0.5:
            gerekceler.append(
                f"UYARI: kotumser (p90) talepte bugun {stoksuz_p90:.0f} adet "
                f"eksik kalabilir; bugun verilecek siparis bugune yetismez.")
        else:
            gerekceler.append(f"Stok {stok_gun:.2f} gunluk talebi karsiliyor.")
        if fire > 0.5:
            gerekceler.append(
                f"Not: beklenen fire {fire:.0f} adet (gunluk talebin "
                f"{fire_oran:.2f} kati), otomatik indirim esigi "
                f"{FIRE_ORAN_ESIGI} asilmadigi icin aksiyon uretilmedi.")
        else:
            gerekceler.append("Fire riski yok.")

    # --- BELIRSIZLIK KAPISI ----------------------------------------------
    kapi = belirsizlik_kapisi(smape_val, kapsama_val, n_val)
    model_zayif = kapi["zayif"]
    if model_zayif and karar in ("siparis_ver", "indirim_uygula"):
        gerekceler.append(
            "Model bu magazada zayif (" + ", ".join(kapi["sebepler"]) +
            "); karar otomatik uygulanmiyor.")
        # Onerilen aksiyon KAYBOLMAZ, ayri alanda tasinir.
        onerilen_aksiyon = (f"{karar}"
                            + (f" (%{indirim_orani * 100:.0f} indirim)"
                               if indirim_orani else "")
                            + (f" ({miktar} adet)" if miktar else ""))
        karar = "insana_sor"
        miktar = 0
        indirim_orani = 0.0
    elif model_zayif:
        # bekle, kapali gun veya celiski: devir zaten var ya da aksiyon yok;
        # ama okuyucu modelin guvenilirligini bilmeli.
        gerekceler.append("Not: bu magazada model zayif (" +
                          ", ".join(kapi["sebepler"]) + ").")

    # --- ACILIYET ---------------------------------------------------------
    if magaza_kapali:
        aciliyet = "kritik" if fire > 0.5 else "dusuk"
    elif fire_riskli or stoksuz_p90 > p50:
        aciliyet = "kritik"
    elif stoksuz_p90 > 0.5 or (stok_gun is not None and stok_gun < 1.0):
        aciliyet = "yuksek"
    elif karar != "bekle":
        aciliyet = "orta"
    else:
        aciliyet = "dusuk"

    # --- ONAY KAPISI ------------------------------------------------------
    onay_sebepleri = []
    if abs(net_etki) > ONAY_ESIGI_EUR:
        onay_sebepleri.append(f"net etki {net_etki:.0f} EUR > {ONAY_ESIGI_EUR:.0f}")
    if fire_riskli:
        onay_sebepleri.append("satinalma politikasi Madde 6.1 (fire orani)")
    if magaza_kapali:
        onay_sebepleri.append("magaza kapali — stok insan degerlendirmesi gerektiriyor")
    elif cakisma:
        onay_sebepleri.append("celisen aksiyonlar (indirim + siparis)")
    elif model_zayif and karar == "insana_sor":
        onay_sebepleri.append("model belirsizligi")
    if bool(o.get("onay_gerekli", False)):
        onay_sebepleri.append(f"magaza bazli onay kaydi: {o.get('onay_sebep', '-')}")

    # 'bekle' AKSIYON YOKLUGUDUR; onaylanacak bir sey yoktur.
    onay_gerekli = bool(onay_sebepleri) and karar != "bekle"
    if karar == "bekle":
        onay_sebepleri = []

    # uygula=True yalnizca toplu modda gecilir. Telegram'dan gelen sorgular
    # indirim gecmisini KIRLETMEZ.
    if uygula and karar == "indirim_uygula" and indirim_orani > 0:
        _gecmise_yaz(m, str(t.date()), indirim_orani)

    # --- CIKTI (n8n_gonder.py payload'i ile ayni alan isimleri) -----------
    return json.dumps({
        "olay": "stok_karari",
        # Deterministik kimlik (magaza + gun): ayni gun yeniden karar
        # uretilirse AYNI id doner, defterde mukerrer satir olusmaz.
        "karar_id": f"{m}_{t:%Y%m%d}",
        "tarih": str(t.date()),
        "magaza": m,
        "kategori": "taze",

        "karar": karar,
        "aciliyet": aciliyet,
        "miktar": miktar,
        "tedarikci": tedarikci if karar == "siparis_ver" else "yok",
        "onay_gerekli_mi": onay_gerekli,
        "onay_sebebi": "; ".join(onay_sebepleri) if onay_sebepleri else "",
        "onerilen_aksiyon": onerilen_aksiyon,
        "indirim_orani": indirim_orani,
        "net_etki_eur": net_etki,
        "fire_riski_birim": round(fire, 1),          # beklenen fire
        "fire_plan_birim": round(fire_plan, 1),       # p50 senaryosu
        "fire_dusuk_talep_birim": round(fire_p10, 1),  # p10 senaryosu
        "gerekce": " ".join(gerekceler),

        "tahmin_satis": p50,
        "guven_alt": alt,
        "guven_ust": ust,
        "guven_bandi": bant,
        "belirsizlik_orani": belirsizlik,

        "mevcut_stok": round(acilis, 1),
        "magaza_kapali": magaza_kapali,
        "kalan_raf_omru": kalan_raf,
        "fire_gun_karsiligi": fire_oran,
        "stok_gun_karsiligi": round(stok_gun, 2) if stok_gun is not None else None,
        "stok_gun_karsiligi_donem_ort": round(stok_gun_donem, 2),
        # Siparis kararinin dayandigi politika motoru ciktilari
        "politika_siparis": round(politika_siparis, 1),
        "teslim_suresi_gun": lead,
        "varis_tarihi": str(varis),
        "stoksuz_p90_bugun": round(stoksuz_p90, 1),

        # Kapinin dayandigi metrikler (karar aninda bilinen, val donemi)
        "model_smape_val": None if smape_val is None else round(smape_val, 2),
        "model_kapsama_val": None if kapsama_val is None else round(kapsama_val, 1),
        "kapsama_binom_p": kapi["kapsama_p"],
        # GERIYE DONUK izleme alanlari: kararda KULLANILMAZ.
        "model_smape_test": None if smape_test is None else round(smape_test, 2),
        "model_kapsama_test": None if kapsama_test is None else round(kapsama_test, 1),

        "esikler": {
            "onay_esigi_eur": ONAY_ESIGI_EUR,
            "fire_oran_esigi": FIRE_ORAN_ESIGI,
            "stok_gun_alt_esik": STOK_GUN_ALT_ESIK,
            "smape_belirsiz_esik": SMAPE_BELIRSIZ_ESIK,
            "kapsama_hedef": KAPSAMA_HEDEF,
            "kapsama_p_esik": KAPSAMA_P_ESIK,
        },
    }, ensure_ascii=False)


# ===========================================================================
# Terminalden hizli dogrulama:  python karar_tool.py 264 2015-07-20
# ===========================================================================
if __name__ == "__main__":
    import sys

    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t = sys.argv[2] if len(sys.argv) > 2 else "2015-07-20"
    print(json.dumps(json.loads(karar_uret(m, t)), ensure_ascii=False, indent=2))
