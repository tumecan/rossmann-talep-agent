"""
agent_v6.py — FINAL Karar Agent'i (Rossmann bitirme projesi, Faz 5)

Karar seti: siparis_ver | bekle | indirim_uygula | insana_sor

TOOL'LAR (6)
    1) tahmin_getir_taze          -> taze kategori talep tahmini (ADET)
    2) stok_durumu_getir          -> FIFO simulasyonundan stok + kalan raf omru
    3) ihtiyac_hesapla            -> siparis ihtiyaci (deterministik)
    4) israf_riski_hesapla        -> KALAN RAF OMRU icinde eriyemeyecek fazla stok
    5) indirim_senaryosu_hesapla  -> bir indirimin EUR cinsinden net etkisi
    6) tedarikci_bilgisi_ara      -> RAG: sozlesme, politika, gecmis analiz
    7) onceki_indirim_getir       -> onceki is gununde uygulanan indirim

MIMARI ILKE (en pahali ders)
    LLM asla sayi URETMEZ ve kural motoru asla LLM'in URETTIGI sayiya
    DAYANMAZ. Deterministik olarak hesaplanabilen her deger kural motorunda
    yeniden hesaplanir; semadaki alan yalnizca agent'in ne dusundugunu
    gosterir, karari baglamaz. (i) ve (j) maddeleri bu ilkenin bedelidir.

ILK SURUMDE YAKALANAN HATALAR VE COZUMLERI
  (a) BIRIM UYUSMAZLIGI: v4'ten devralinan tahmin_getir ham yhat'i (EUR ciro)
      donduruyordu, stok ise adet cinsindendi. Agent ikisini ayni hesaba soktu
      ve 3950 birimlik sahte ihtiyac uretti; kural motoru 0'a cekerek kurtardi.
      Cozum: tum tool'lar ayni birimde (taze kategori adedi) konusur.
  (b) ISRAF RISKI HACME BAKIYORDU: fazla stok, tum raf omru penceresindeki
      talebe gore hesaplaniyordu. Az ama BAYAT stok "risksiz" gorunuyordu.
      Cozum: pencere artik elde duran partinin KALAN raf omru kadardir.
  (c) IHTIYAC YOKKEN "siparis_ver": v5'teki bekle kurali v6'ya tasinmamisti.
  (d) SEMA COKMESI: 11 zorunlu alan kucuk model icin fazlaydi. Cozum:
      varsayilan degerler + "aciliyet"in kural motorunda hesaplanmasi.
  (e) CHROMA THREAD HATASI: LangGraph tool'lari worker thread'de calistirir.
      Chroma istemcisi ilk kez orada olusursa chromadb'nin Rust baglayicisi
      cokuyor. Cozum: istemci modul yuklenirken ANA THREAD'de acilir.
  (g) POLITIKA KISITI VERISIZ UYGULANIYORDU: agent "pes pese iki gun %50
      indirim uygulanamaz" kuralini okudu ama onceki gun ne yapildigini
      BILMEDEN, ihtiyaten uyguladi. Cozum: indirim gecmisi diske yazilir ve
      7. tool ile sorgulanir; kural artik gercek veriye dayanir. Kural motoru
      da ayni kisiti bagimsiz olarak dayatir.
  (h) KAPALI GUN TARIHI: agent onceki_indirim_getir'e bugunun degil, kendi
      hesapladigi onceki TAKVIM gununun tarihini geciriyordu. Pazartesi
      gunlerinde bu Pazar'a denk geliyor ve magaza kapali oldugu icin veri
      setinde yok -> IndexError. Cozum: tarih arama toleransli hale getirildi,
      bilinmeyen tarih hata mesaji doner, cokmez.
  (i) FIRE RISKI VARKEN "bekle": agent bazi gunlerde indirim senaryosunu
      hesaplayip (net etki 400+ EUR) yine de "bekle" dedi. Ilk cozum kural
      motoruna bir kontrol ekledi, ama o kontrol "tahmini_net_etki_eur > 0"
      sartina baglanmisti: agent "bekle" derken bu alani doldurmadigi icin 0
      kaliyor ve kural TAM DA KORUMASI GEREKEN DURUMDA sessiz kaliyordu
      (2015-06-26 ve 2015-07-10). Nihai cozum: kosul yalnizca fire riskine
      bakar; net etkiyi kural motoru emniyet_indirim_sec ile kendisi hesaplar.
  (j) FIRE RISKININ KENDISI DE LLM'DEN GELIYORDU: (i) duzeltildikten sonra
      agent bir gun (2015-07-24) israf riskini hic hesaplamadan
      fire_riski_birim=0 dondurdu. Gercek deger 112 birim, atlanan kazanc
      ~377 EUR idi. Kural motoru 0 gordugu icin devreye girmedi, toplu
      kosudaki tutarlilik kontrolu de ayni alana baktigi icin kacirdi.
      Cozum: gercek_fire_riski() ile deger DETERMINISTIK hesaplanir ve
      semadaki alan kural motorunda uzerine yazilir. Tool ile kural motoru
      artik tek bir formul paylasir.
      Ders: savunma katmani, korumaya calistigi katmanin ciktisina bagimli
      olmamali — (i)'de ogrenildi, (j)'de bir kat asagida tekrar cikti.
  (f) YUZDE / ONDALIK KARISIKLIGI: belgede "%30" yazdigi icin model
      indirim_orani=30 dondurup semayi bozdu. Prompt'ta "ondalik yaz" demek
      kucuk modellerde yetmiyor. Cozum: savunmaci ayristirma — 1'den buyuk
      gelen oran otomatik 100'e bolunur (hem semada hem tool girisinde).

Onkosullar:
    python Rossman\\prophet_model.py
    python tedarikci_dokumanlari_olustur.py
    python politika_dokumanlari_olustur.py
    python rag_index.py

Calistirma:
    python agent_v6.py --tarih 2015-06-26 --trace
    python agent_v6.py --toplu
"""

import json
import os
import sys
from datetime import datetime
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from agent_v2 import CIKTI_DIR, hesapla, logla
from agent_v4_langchain import ihtiyac_hesapla
from agent_v5_rag import tedarikci_bilgisi_ara
from rag_index import indeks_yukle
from stok_simulasyon import (
    BIRIM_FIYAT,
    BIRIM_MALIYET,
    RAF_OMRU_GUN,
    KOK,
    Z_ARALIK,
    simule_et,
    veri_hazirla,
)

MODEL_ADI = "gpt-4o-mini"
PROMPT_SURUM = "v7"          # prompt v6'daki tedarikci sorgu kurallari duzeltildi
ONAY_ESIGI = 3000
VARSAYILAN_TEDARIKCI = "Schnellware"

PROMPT_DOSYA = CIKTI_DIR.parent / "prompts" / f"karar_promptu_{PROMPT_SURUM}.txt"
if not PROMPT_DOSYA.exists():
    PROMPT_DOSYA = CIKTI_DIR / "prompts" / f"karar_promptu_{PROMPT_SURUM}.txt"
KARAR_JSON = CIKTI_DIR / "agent_karar_v6.json"
TOPLU_CSV = CIKTI_DIR / "agent_kararlari_v6.csv"
GECMIS_JSON = CIKTI_DIR / "indirim_gecmisi.json"

TAHMIN = veri_hazirla(KOK / "prophet_forecast.csv")
STOK, _ = simule_et(TAHMIN, VARSAYILAN_TEDARIKCI, "ortalama")
DURUM = TAHMIN.join(STOK[["acilis_stok", "kapanis_stok", "fire",
                          "stoksuz", "en_eski_yas", "kalan_raf_omru"]])

# Chroma istemcisi ANA THREAD'de acilir. LangGraph tool'lari worker thread'de
# calistirdigi icin, istemci ilk kez orada olusturulursa chromadb'nin Rust
# baglayicisi cokuyor (RustBindingsAPI hatasi). Onbellegi burada dolduruyoruz.
indeks_yukle()

# Uygulanmis indirimlerin gecmisi: {"2015-06-26": 0.30, ...}
# Pes pese iki gun %50 yasagi bu kayda dayanir.
INDIRIM_GECMISI: dict[str, float] = (
    json.loads(GECMIS_JSON.read_text(encoding="utf-8"))
    if GECMIS_JSON.exists() else {}
)


def gecmise_yaz(tarih: str, oran: float) -> None:
    INDIRIM_GECMISI[tarih] = oran
    GECMIS_JSON.write_text(
        json.dumps(INDIRIM_GECMISI, ensure_ascii=False, indent=2), encoding="utf-8")


def onceki_is_gunu(tarih: str) -> str | None:
    """Bir onceki IS GUNU (kapali gunler veri setinde yok, atlanir)."""
    try:
        i = _idx(tarih)
    except ValueError:
        return None
    return DURUM.iloc[i - 1]["ds"].strftime("%Y-%m-%d") if i > 0 else None


def _satir(tarih: str) -> pd.Series:
    e = DURUM[DURUM["ds"] == pd.Timestamp(tarih)]
    if e.empty:
        raise ValueError(tarih)
    return e.iloc[0]


def _idx(tarih: str) -> int:
    """Tarihin satir indeksi. Bilinmeyen tarih icin ValueError."""
    eslesen = DURUM.index[DURUM["ds"] == pd.Timestamp(tarih)]
    if len(eslesen) == 0:
        raise ValueError(tarih)
    return int(eslesen[0])


def gercek_fire_riski(tarih: str) -> float:
    """Fire riski tasiyan fazla stok — TEK DOGRULUK KAYNAGI.
    Hem israf_riski_hesapla tool'u hem kural motoru bu fonksiyonu kullanir,
    boylece agent tool'u hic cagirmasa bile kural motoru dogru degeri bilir.
    Mantik: fazla = mevcut_stok - (KALAN raf omru gunlerindeki beklenen talep).
    Bilinmeyen tarihte 0 doner (cagiran taraf zaten hata mesaji uretir)."""
    try:
        s = _satir(tarih)
        idx = _idx(tarih)
    except ValueError:
        return 0.0
    kalan = int(s["kalan_raf_omru"])
    beklenen = float(DURUM.iloc[idx + 1: idx + 1 + kalan]["talep_tahmin"].sum())
    return round(max(0.0, float(s["kapanis_stok"]) - beklenen), 1)


# ============================================================================
# TOOL'LAR — hepsi ayni birimde konusur: taze kategori ADEDI
# ============================================================================
@tool
def tahmin_getir_taze(tarih: str) -> str:
    """Belirtilen tarih icin taze kategori talep tahminini ADET cinsinden
    dondurur. %90 guven araligi da adet cinsindendir. Tarih: YYYY-MM-DD.
    Doner: tahmin_satis, guven_alt, guven_ust, promo_var_mi (hepsi adet)."""
    try:
        s = _satir(tarih)
    except ValueError:
        return (f"HATA: {tarih} icin tahmin yok. Gecerli aralik: "
                f"{DURUM['ds'].min().date()} - {DURUM['ds'].max().date()}")
    tahmin = float(s["talep_tahmin"])
    sapma = Z_ARALIK * float(s["sigma"])
    return json.dumps({
        "tarih": tarih,
        "birim": "adet (taze kategori)",
        "tahmin_satis": round(tahmin, 1),
        "guven_alt": round(max(0.0, tahmin - sapma), 1),
        "guven_ust": round(tahmin + sapma, 1),
        "promo_var_mi": bool(s["promo"]),
    }, ensure_ascii=False)


@tool
def stok_durumu_getir(tarih: str) -> str:
    """Belirtilen tarihte magazadaki stok seviyesini ve elde duran malin
    KALAN RAF OMRUNU dondurur. Stok, FIFO parti takibi yapan bir simulasyondan
    gelir. Doner: mevcut_stok, en_eski_parti_yasi, kalan_raf_omru_gun."""
    try:
        s = _satir(tarih)
    except ValueError:
        return f"HATA: {tarih} icin stok verisi yok."
    return json.dumps({
        "tarih": tarih,
        "birim": "adet (taze kategori)",
        "mevcut_stok": round(float(s["kapanis_stok"]), 1),
        "gunluk_talep_tahmini": round(float(s["talep_tahmin"]), 1),
        "en_eski_parti_yasi_gun": int(s["en_eski_yas"]),
        "kalan_raf_omru_gun": int(s["kalan_raf_omru"]),
        "toplam_raf_omru_gun": RAF_OMRU_GUN,
    }, ensure_ascii=False)


@tool
def israf_riski_hesapla(tarih: str) -> str:
    """Elde duran malin KALAN raf omru icinde satilamayacak, yani fire riski
    tasiyan kismini hesaplar.
    Mantik: fazla = mevcut_stok - (kalan raf omru gunlerindeki beklenen talep).
    Stok hacmi kucuk olsa bile mal bayatsa risk cikar; bu yuzden pencere toplam
    raf omru degil KALAN raf omrudur.
    Doner: fazla_stok, fire_riski_var_mi, fire_halinde_zarar_eur."""
    try:
        s = _satir(tarih)
    except ValueError:
        return f"HATA: {tarih} icin veri yok."

    idx = _idx(tarih)
    kalan = int(s["kalan_raf_omru"])
    beklenen = float(DURUM.iloc[idx + 1: idx + 1 + kalan]["talep_tahmin"].sum())
    stok = float(s["kapanis_stok"])
    fazla = gercek_fire_riski(tarih)      # kural motoruyla ayni formul

    return json.dumps({
        "tarih": tarih,
        "birim": "adet (taze kategori)",
        "mevcut_stok": round(stok, 1),
        "en_eski_parti_yasi_gun": int(s["en_eski_yas"]),
        "kalan_raf_omru_gun": kalan,
        "kalan_surede_beklenen_talep": round(beklenen, 1),
        "fazla_stok": fazla,
        "fire_riski_var_mi": fazla > 0,
        "fire_halinde_zarar_eur": round(fazla * BIRIM_MALIYET, 1),
        "birim_maliyet_eur": BIRIM_MALIYET,
        "birim_satis_fiyati_eur": BIRIM_FIYAT,
    }, ensure_ascii=False)


@tool
def indirim_senaryosu_hesapla(fazla_stok: float, indirim_orani: float,
                              erime_orani: float) -> str:
    """Belirli bir indirim senaryosunun, hic aksiyon almamaya kiyasla net
    etkisini EUR cinsinden hesaplar.
    indirim_orani ve erime_orani KODDA TANIMLI DEGILDIR; bu degerleri once
    tedarikci_bilgisi_ara ile indirim politikasi belgesinden ogrenmelisin.
    Ornek: indirim_orani=0.30, erime_orani=0.60.
    Doner: aksiyon almama zarari, indirimli senaryo zarari ve aradaki fark."""
    # savunmaci ayristirma: "%30" niyetiyle 30 gelirse 0.30'a cevir
    if indirim_orani > 1:
        indirim_orani = indirim_orani / 100
    if erime_orani > 1:
        erime_orani = erime_orani / 100

    if not (0 < indirim_orani < 1) or not (0 < erime_orani <= 1):
        return ("HATA: indirim_orani 0-1 arasi ondalik olmali (or. 0.30), "
                "erime_orani 0-1 arasi olmali (or. 0.60).")
    if fazla_stok <= 0:
        return "HATA: fazla_stok sifir veya negatif; indirim senaryosu gereksiz."

    zarar_aksiyonsuz = fazla_stok * BIRIM_MALIYET
    eriyen = fazla_stok * erime_orani
    indirimli_fiyat = BIRIM_FIYAT * (1 - indirim_orani)
    gelir = eriyen * indirimli_fiyat
    zarar_indirimli = zarar_aksiyonsuz - gelir

    return json.dumps({
        "fazla_stok": round(fazla_stok, 1),
        "indirim_orani": indirim_orani,
        "erime_orani": erime_orani,
        "eriyen_birim": round(eriyen, 1),
        "yine_fire_olan_birim": round(fazla_stok - eriyen, 1),
        "indirimli_birim_fiyat_eur": round(indirimli_fiyat, 2),
        "aksiyon_yok_zarar_eur": round(zarar_aksiyonsuz, 1),
        "indirim_zarar_eur": round(zarar_indirimli, 1),
        "net_etki_eur": round(zarar_aksiyonsuz - zarar_indirimli, 1),
        "yorum": "net_etki_eur pozitifse indirim, aksiyon almamaktan iyidir",
    }, ensure_ascii=False)


@tool
def onceki_indirim_getir(tarih: str) -> str:
    """Bir onceki IS GUNUNDE uygulanmis indirim oranini dondurur.
    "Pes pese iki gun %50 indirim uygulanamaz" kuralini uygulayabilmek icin
    gereklidir; kurali varsayimla degil GERCEK GECMISLE uygula.
    Doner: onceki_is_gunu, onceki_indirim_orani, bugun_yuzde50_uygulanabilir_mi.
    NOT: Bu araca KARAR VERDIGIN GUNUN tarihini ver, onceki gunun degil.
    Onceki is gununu arac kendisi bulur (magazanin kapali oldugu gunler
    veri setinde yoktur, otomatik atlanir)."""
    try:
        _idx(tarih)
    except ValueError:
        return (f"HATA: {tarih} veri setinde yok (magaza kapali olabilir). "
                f"Bu araca karar verdigin gunun tarihini ver. Gecerli aralik: "
                f"{DURUM['ds'].min().date()} - {DURUM['ds'].max().date()}")

    onceki = onceki_is_gunu(tarih)
    if onceki is None:
        return json.dumps({
            "tarih": tarih,
            "onceki_is_gunu": None,
            "onceki_indirim_orani": 0.0,
            "bugun_yuzde50_uygulanabilir_mi": True,
            "not": "Bu ilk gun; gecmis kayit yok.",
        }, ensure_ascii=False)

    oran = float(INDIRIM_GECMISI.get(onceki, 0.0))
    return json.dumps({
        "tarih": tarih,
        "onceki_is_gunu": onceki,
        "onceki_indirim_orani": oran,
        "bugun_yuzde50_uygulanabilir_mi": oran < 0.5,
        "not": ("Onceki gun %50 uygulandi; bugun en fazla %30 uygulanabilir."
                if oran >= 0.5 else "Kisit yok."),
    }, ensure_ascii=False)


TOOLS = [tahmin_getir_taze, stok_durumu_getir, ihtiyac_hesapla,
         israf_riski_hesapla, indirim_senaryosu_hesapla,
         tedarikci_bilgisi_ara, onceki_indirim_getir]


# ============================================================================
# SEMA
# ============================================================================
class AgentKararV6(BaseModel):
    karar: Literal["siparis_ver", "bekle", "indirim_uygula", "insana_sor"] = Field(
        description="Alinan aksiyon")
    gerekce: str = Field(min_length=5, max_length=500,
                         description="En fazla 3 cumle Turkce gerekce")
    miktar: int = Field(default=0, ge=0,
                        description="Siparis miktari (adet); indirimde 0")
    tedarikci: str = Field(default="yok",
                           description="Secilen tedarikcinin tam adi")
    teslimat_notu: str = Field(default="", max_length=300,
                               description="SECILEN tedarikcinin lead-time, "
                                           "minimum miktar ve iskontosu")
    onay_gerekli_mi: bool = Field(default=False)
    fire_riski_birim: float = Field(default=0.0, ge=0)
    secilen_senaryo: str = Field(default="aksiyon yok", max_length=120)
    indirim_orani: float = Field(default=0.0, ge=0, le=1)
    tahmini_net_etki_eur: float = Field(default=0.0)
    aciliyet: Literal["yuksek", "orta", "dusuk"] = Field(default="orta")

    @field_validator("indirim_orani", mode="before")
    @classmethod
    def _orani_normalize(cls, deger):
        """Model "%30" niyetiyle 30 dondurebilir; 0.30'a cevir."""
        try:
            sayi = float(deger)
        except (TypeError, ValueError):
            return 0.0
        return sayi / 100 if sayi > 1 else sayi


def aciliyet_hesapla(hesap: dict, girdi: dict, fire_riski: float) -> str:
    if fire_riski > 0:
        return "yuksek"
    if hesap["ihtiyac"] <= 0:
        return "dusuk"
    if girdi["mevcut_stok"] > 0 and hesap["ihtiyac"] > girdi["mevcut_stok"] / 2:
        return "yuksek"
    return "orta"


# Kural motorunun emniyet senaryolari. Asil yol RAG'dir: agent oranlari
# indirim politikasi belgesinden okur ve indirim_senaryosu_hesapla ile
# karsilastirir. Bunlar sadece agent o degerlendirmeyi hic yapmadan "bekle"
# dediginde devreye giren son savunma katmanidir.
# (indirim_orani, erime_orani)
EMNIYET_SENARYOLARI = ((0.5, 0.85), (0.3, 0.60))


def emniyet_indirim_sec(fazla_stok: float, yuzde50_serbest: bool):
    """Politikanin izin verdigi senaryolar arasindan net etkisi en yuksek
    olani secer. Formul indirim_senaryosu_hesapla ile aynidir.
    Doner: (indirim_orani, erime_orani, net_etki_eur)."""
    en_iyi = None
    for indirim, erime in EMNIYET_SENARYOLARI:
        if indirim >= 0.5 and not yuzde50_serbest:
            continue
        etki = round(fazla_stok * erime * BIRIM_FIYAT * (1 - indirim), 1)
        if en_iyi is None or etki > en_iyi[2]:
            en_iyi = (indirim, erime, etki)
    return en_iyi


def kurallari_dayat(karar: AgentKararV6, hesap: dict, girdi: dict):
    mudahaleler = []
    v = karar.model_dump()

    # ADIM 0 — Deterministik degerleri LLM'den geri al.
    # Fire riski hesaplanabilir bir buyukluktur; agent tool'u cagirmayi
    # atlarsa alan 0 kalir ve asagidaki butun kurallar yanlis veriyle calisir.
    # Bu yuzden karar mantigina girmeden once dogru deger yazilir.
    gercek = gercek_fire_riski(girdi["tarih"])
    if abs(v["fire_riski_birim"] - gercek) > 0.5:
        mudahaleler.append(
            f"fire_riski_birim: {v['fire_riski_birim']:.1f} -> {gercek:.1f} "
            f"(deterministik yeniden hesap)")
        v["fire_riski_birim"] = gercek

    if v["karar"] != "indirim_uygula":
        if v["miktar"] != hesap["onerilen_miktar"]:
            mudahaleler.append(f"miktar: {v['miktar']} -> {hesap['onerilen_miktar']}")
            v["miktar"] = hesap["onerilen_miktar"]

        # ihtiyac yokken siparis verilemez
        if hesap["ihtiyac"] <= 0 and v["karar"] == "siparis_ver":
            mudahaleler.append("karar: siparis_ver -> bekle (ihtiyac yok)")
            v["karar"] = "bekle"
            v["tedarikci"] = "yok"
            v["teslimat_notu"] = ""

        if hesap["belirsiz_mi"] and v["karar"] == "siparis_ver":
            mudahaleler.append("karar: siparis_ver -> insana_sor (belirsizlik)")
            v["karar"] = "insana_sor"
    else:
        if v["miktar"] != 0:
            mudahaleler.append(f"miktar: {v['miktar']} -> 0 (indirim karari)")
            v["miktar"] = 0

        # politika: pes pese iki gun %50 indirim uygulanamaz
        onceki = onceki_is_gunu(girdi["tarih"])
        onceki_oran = float(INDIRIM_GECMISI.get(onceki, 0.0)) if onceki else 0.0
        if v["indirim_orani"] >= 0.5 and onceki_oran >= 0.5:
            mudahaleler.append(
                f"indirim_orani: {v['indirim_orani']} -> 0.3 "
                f"(pes pese %50 yasagi; onceki gun {onceki})")
            v["indirim_orani"] = 0.3
            v["secilen_senaryo"] = "%30 indirim (politika kisiti)"

    # Fire riski varken "aksiyon almamak" bir karar degildir.
    # Satinalma politikasi "Aksiyon Onceligi" maddesi: eritilebilen stok icin
    # fire yazilmadan once indirim degerlendirilir. Agent bu degerlendirmeyi
    # atlarsa kural motoru emniyet agi olarak devreye girer. Net etki burada
    # deterministik hesaplanir; LLM'in doldurduguna guvenilmez.
    if v["fire_riski_birim"] > 0 and v["karar"] != "indirim_uygula":
        onceki = onceki_is_gunu(girdi["tarih"])
        onceki_oran = float(INDIRIM_GECMISI.get(onceki, 0.0)) if onceki else 0.0
        indirim, _erime, etki = emniyet_indirim_sec(
            v["fire_riski_birim"], yuzde50_serbest=onceki_oran < 0.5)

        mudahaleler.append(
            f"karar: {v['karar']} -> indirim_uygula "
            f"(fire riski {v['fire_riski_birim']:.0f} birim; emniyet senaryosu "
            f"%{indirim * 100:.0f}, net etki +{etki:.0f} EUR)")
        v["karar"] = "indirim_uygula"
        v["miktar"] = 0
        v["tedarikci"] = "yok"
        v["teslimat_notu"] = ""
        if v["indirim_orani"] == 0:
            v["indirim_orani"] = indirim
            v["secilen_senaryo"] = f"%{indirim * 100:.0f} indirim (kural motoru)"
        if v["tahmini_net_etki_eur"] == 0:
            v["tahmini_net_etki_eur"] = etki

    dogru_aciliyet = aciliyet_hesapla(hesap, girdi, v["fire_riski_birim"])
    if v["aciliyet"] != dogru_aciliyet:
        mudahaleler.append(f"aciliyet: {v['aciliyet']} -> {dogru_aciliyet}")
        v["aciliyet"] = dogru_aciliyet

    gerekli = (v["miktar"] > ONAY_ESIGI
               or v["karar"] == "insana_sor"
               or v["indirim_orani"] >= 0.5)
    if v["onay_gerekli_mi"] != gerekli:
        mudahaleler.append(f"onay_gerekli_mi: {v['onay_gerekli_mi']} -> {gerekli}")
        v["onay_gerekli_mi"] = gerekli

    return AgentKararV6(**v), mudahaleler


# ============================================================================
def agent_olustur():
    load_dotenv()
    anahtar = os.getenv("openai_apikey")
    if not anahtar:
        sys.exit("HATA: .env icinde openai_apikey yok.")
    if not PROMPT_DOSYA.exists():
        sys.exit(f"HATA: {PROMPT_DOSYA} yok.")
    llm = ChatOpenAI(model=MODEL_ADI, temperature=0,
                     api_key=anahtar.strip().strip('"'))
    return create_agent(model=llm, tools=TOOLS,
                        system_prompt=PROMPT_DOSYA.read_text(encoding="utf-8"),
                        response_format=AgentKararV6)


def girdi_hazirla(tarih: str) -> dict:
    s = _satir(tarih)
    tahmin = float(s["talep_tahmin"])
    sapma = Z_ARALIK * float(s["sigma"])
    return {
        "magaza": 1, "tarih": tarih, "birim": "adet",
        "tahmin_satis": round(tahmin, 1),
        "guven_alt": round(max(0.0, tahmin - sapma), 1),
        "guven_ust": round(tahmin + sapma, 1),
        "mevcut_stok": round(float(s["kapanis_stok"]), 1),
        "kalan_raf_omru": int(s["kalan_raf_omru"]),
        "promo_var_mi": bool(s["promo"]),
    }


def karar_ver(agent, tarih: str, trace=False, yazdir=True) -> dict:
    soru = (
        f"{tarih} tarihi, Magaza 1, taze kategori. Tam bir stok karari ver: "
        "tahmin ve stok durumunu getir, israf riskini kontrol et, siparis "
        "ihtiyacini hesapla, fazla stok varsa indirim senaryolarini belgelerden "
        "bulup hesapla ve karsilastir, tedarikci ve onay durumunu belirle."
    )
    sonuc = agent.invoke({"messages": [{"role": "user", "content": soru}]})
    adimlar = [{"tool": c["name"], "args": c["args"]}
               for m in sonuc["messages"] for c in (getattr(m, "tool_calls", []) or [])]

    girdi = girdi_hazirla(tarih)
    hesap = hesapla(girdi)
    karar, mudahaleler = kurallari_dayat(sonuc["structured_response"], hesap, girdi)

    if karar.karar == "indirim_uygula" and karar.indirim_orani > 0:
        gecmise_yaz(tarih, karar.indirim_orani)

    if yazdir:
        print(f"\n{'='*72}\n{tarih} | Magaza 1 | taze kategori "
              f"| stok {girdi['mevcut_stok']:.0f} adet "
              f"| kalan raf omru {girdi['kalan_raf_omru']} gun\n{'='*72}")
        print(f"Tool zinciri ({len(adimlar)}): "
              f"{' -> '.join(a['tool'] for a in adimlar)}")
        if trace:
            print("\n--- TOOL TRACE ---")
            for i, a in enumerate(adimlar, 1):
                print(f"  {i}. {a['tool']}({json.dumps(a['args'], ensure_ascii=False)})")
        print("\n--- KARAR ---")
        print(json.dumps(karar.model_dump(), ensure_ascii=False, indent=2))
        if mudahaleler:
            print("\n--- Kural motoru ---")
            for m in mudahaleler:
                print("  *", m)
        else:
            print("\n(kural motoru mudahale etmedi)")

    kayit = {
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_ADI, "prompt_surum": PROMPT_SURUM,
        "mimari": "langchain create_agent (6 tool, RAG + israf analizi)",
        "tool_trace": adimlar, "girdi": girdi, "hesaplanan": hesap,
        "karar": karar.model_dump(), "mudahaleler": mudahaleler,
    }
    logla(kayit)
    return kayit


def toplu(agent) -> None:
    # Temiz kosu: gecmis sifirlanir, gunler sirayla islenir ve her indirim
    # karari bir sonraki gunun kisitini belirler.
    INDIRIM_GECMISI.clear()
    if GECMIS_JSON.exists():
        GECMIS_JSON.unlink()

    print(f"\n########## TOPLU KARAR v6 ({len(DURUM)} gun) ##########")
    satirlar = []
    for _, s in DURUM.iterrows():
        tarih = s["ds"].strftime("%Y-%m-%d")
        try:
            k = karar_ver(agent, tarih, yazdir=False)
        except Exception as e:
            print(f"{tarih} -> HATA: {type(e).__name__}")
            continue
        kr, g = k["karar"], k["girdi"]
        satirlar.append({
            "tarih": tarih, "stok": g["mevcut_stok"],
            "kalan_raf_omru": g["kalan_raf_omru"], "promo": int(g["promo_var_mi"]),
            "karar": kr["karar"], "miktar": kr["miktar"], "aciliyet": kr["aciliyet"],
            "tedarikci": kr["tedarikci"], "fire_riski": kr["fire_riski_birim"],
            "senaryo": kr["secilen_senaryo"], "indirim": kr["indirim_orani"],
            "net_etki_eur": kr["tahmini_net_etki_eur"],
            "onay": int(kr["onay_gerekli_mi"]), "tool_sayisi": len(k["tool_trace"]),
            "kural_mudahalesi": int(bool(k["mudahaleler"])),
            "gerekce": kr["gerekce"],
        })
        print(f"{tarih} -> {kr['karar']:15} miktar={kr['miktar']:6} "
              f"fire={kr['fire_riski_birim']:7.0f} "
              f"etki={kr['tahmini_net_etki_eur']:8.0f} EUR")

    df = pd.DataFrame(satirlar)
    df.to_csv(TOPLU_CSV, index=False, encoding="utf-8-sig")
    print("\n--- Karar dagilimi ---")
    print(df["karar"].value_counts().to_string())
    print(f"\nFire riski olan gun     : {(df['fire_riski'] > 0).sum()}")
    print(f"Indirim onerilen gun    : {(df['karar'] == 'indirim_uygula').sum()}")
    kurtarilan = df[df['karar'] == 'indirim_uygula']['net_etki_eur'].sum()
    print(f"Indirimle kurtarilan    : {kurtarilan:,.0f} EUR")
    print(f"Insan onayi gereken gun : {df['onay'].sum()}")
    print(f"Ortalama tool cagrisi   : {df['tool_sayisi'].mean():.1f}")
    print(f"Kural motoru mudahalesi : {df['kural_mudahalesi'].sum()} gun")

    # Tutarlilik kontrolu — fire riski artik deterministik oldugu icin bu
    # kontrol de guvenilir: LLM'in yazdigi degere degil, gercege bakiyor.
    kacak = df[(df["fire_riski"] > 0) & (df["karar"] != "indirim_uygula")]
    if kacak.empty:
        print("Tutarlilik              : OK (fire riskli hicbir gun aksiyonsuz degil)")
    else:
        print(f"Tutarlilik              : UYARI! {len(kacak)} gun aksiyonsuz:")
        print(kacak[["tarih", "karar", "fire_riski"]].to_string(index=False))

    print(f"\nKaydedildi: {TOPLU_CSV}")


if __name__ == "__main__":
    agent = agent_olustur()
    if "--toplu" in sys.argv:
        toplu(agent)
    else:
        tarih = (sys.argv[sys.argv.index("--tarih") + 1] if "--tarih" in sys.argv
                 else DURUM.iloc[-1]["ds"].strftime("%Y-%m-%d"))
        k = karar_ver(agent, tarih, trace="--trace" in sys.argv)
        KARAR_JSON.write_text(json.dumps(k, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"\nKaydedildi: {KARAR_JSON}")