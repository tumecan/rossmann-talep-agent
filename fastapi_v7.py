"""
fastapi_v7.py — agent_v7 icin FastAPI sarmalayici

Calistirma:  python fastapi_v7.py
Endpoint:    POST /soru        -> soru sor, gerekirse yapisal karar uret
             POST /onay        -> insan onayi/reddi kaydet (n8n cagirir)
             GET  /onay-ozet   -> onay istatistikleri
             GET  /ogrenme-raporu -> esik onerileri (ogrenme_dongusu)
             GET  /trend       -> haftalik model/karar trendi + alarmlar
             GET  /gunluk-rapor -> n8n Schedule'in cagirdigi paket
             GET  /saglik      -> servis ayakta mi

NEDEN ONAY KAYDI BURADA
    n8n Docker icinde calisiyor ve Google Sheets'e yaziyor. Ogrenme
    dongusunun girdisi olan onay kayitlarini Sheets'ten okumak, Python
    tarafinda ayri bir Google kimlik dogrulamasi gerektirirdi. Bunun
    yerine n8n her onay olayini buraya POST eder; kayit koddan okunabilir
    bir JSONL dosyasina duser. Sheets kullanici arayuzu, JSONL ise
    ogrenme dongusunun veri kaynagidir.
"""

import json
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from agent_v7 import agent_olustur, son_karar, sor
import contextlib
import io

import ogrenme_dongusu as O
import trend_analizi as TR

app = FastAPI(title="Rossmann Stok Asistani v7")

KOK = Path(__file__).resolve().parent
ROSSMAN = KOK / "Rossman"
ONAY_LOG = (ROSSMAN if ROSSMAN.exists() else KOK) / "onay_kayitlari.jsonl"

agent = None
gecmisler: dict[str, list] = {}

# Uretilen kararlar bellekte tutulur; onay geldiginde kararin PARAMETRELERI
# (esikler, karar turu, tutar) kayda eklenebilsin diye. Ogrenme dongusu
# "hangi esikle uretilen karar reddedildi" sorusunu ancak boyle cevaplar.
son_kararlar: dict[str, dict] = {}

# Red sebebi kodlari — serbest metin YERINE sabit kume.
# Sebep serbest metin olsaydi ogrenme dongusu bunlari gruplayamazdi;
# "tahmin sacma" ile "model guvenilmez" ayni kutuya girmezdi.
RED_SEBEPLERI = {
    "model": "Tahmin gercekci degil",
    "parametre": "Miktar veya oran yanlis",
    "zamanlama": "Zamanlama uygun degil",
    "diger": "Baska sebep",
}


class SoruIstegi(BaseModel):
    soru: str
    chat_id: str = "default"


class CevapYaniti(BaseModel):
    cevap: str
    tool_sayisi: int
    tool_isimleri: list[str]
    karar_var_mi: bool = False
    karar: dict | None = None


class OnayIstegi(BaseModel):
    karar_id: str
    durum: str                      # onaylandi | reddedildi
    sebep_kodu: str | None = None   # yalnizca reddedildi icin
    chat_id: str | None = None


class OnayYaniti(BaseModel):
    kaydedildi: bool
    karar_id: str
    durum: str
    sebep: str | None = None
    mesaj: str


@app.on_event("startup")
def baslat():
    global agent
    agent = agent_olustur()
    print("Agent hazirlandi.")


@app.get("/saglik")
def saglik():
    return {"durum": "ayakta", "agent_hazir": agent is not None,
            "bellekteki_karar": len(son_kararlar)}


@app.post("/soru", response_model=CevapYaniti)
def soru_sor(istek: SoruIstegi):
    gecmis = gecmisler.get(istek.chat_id, [])
    cevap, yeni_gecmis, adimlar = sor(agent, istek.soru, gecmis, trace=False)
    gecmisler[istek.chat_id] = yeni_gecmis

    # Karar, LLM'in metninden AYIKLANMAZ; dogrudan tool ciktisidir.
    karar = son_karar(adimlar)
    if karar and karar.get("karar_id"):
        son_kararlar[karar["karar_id"]] = karar

    return CevapYaniti(
        cevap=cevap,
        tool_sayisi=len(adimlar),
        tool_isimleri=[a["tool"] for a in adimlar],
        karar_var_mi=karar is not None,
        karar=karar,
    )


@app.post("/onay", response_model=OnayYaniti)
def onay_kaydet(istek: OnayIstegi):
    """Insan onayini/reddini kalici olarak kaydeder.

    Onaylanan bir INDIRIM karari, indirim gecmisine de yazilir: politika
    kisiti ("pes pese iki gun %50 uygulanamaz") ancak gercekten UYGULANAN
    indirimlere dayanmalidir. Yalnizca onerilmis ama onaylanmamis bir
    indirim gecmisi kirletirse kisit yanlis tetiklenir.
    """
    durum = istek.durum.strip().lower()
    if durum not in ("onaylandi", "reddedildi"):
        return OnayYaniti(kaydedildi=False, karar_id=istek.karar_id,
                          durum=durum, mesaj="Gecersiz durum.")

    karar = son_kararlar.get(istek.karar_id, {})
    sebep = RED_SEBEPLERI.get(istek.sebep_kodu or "")

    kayit = {
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "karar_id": istek.karar_id,
        "durum": durum,
        "sebep_kodu": istek.sebep_kodu,
        "sebep": sebep,
        "chat_id": istek.chat_id,
        # Kararin uretim parametreleri — ogrenme dongusunun girdisi
        "karar": karar.get("karar"),
        "magaza": karar.get("magaza"),
        "tarih": karar.get("tarih"),
        "aciliyet": karar.get("aciliyet"),
        "miktar": karar.get("miktar"),
        "indirim_orani": karar.get("indirim_orani"),
        "net_etki_eur": karar.get("net_etki_eur"),
        "onay_sebebi": karar.get("onay_sebebi"),
        "fire_gun_karsiligi": karar.get("fire_gun_karsiligi"),
        "stok_gun_karsiligi": karar.get("stok_gun_karsiligi"),
        "model_smape_val": karar.get("model_smape_val"),
        "model_kapsama_val": karar.get("model_kapsama_val"),
        "kapsama_binom_p": karar.get("kapsama_binom_p"),
        "model_smape_test": karar.get("model_smape_test"),
        "model_kapsama_test": karar.get("model_kapsama_test"),
        "esikler": karar.get("esikler"),
        # Karar bellekte bulunamadiysa (servis yeniden baslamis olabilir)
        # kayit yine tutulur; eksik alanlar ogrenme dongusunda elenir.
        "karar_bulundu": bool(karar),
    }

    ONAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ONAY_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")

    # Onaylanan indirim GERCEKTEN uygulanmis sayilir -> gecmise yaz
    if durum == "onaylandi" and karar.get("karar") == "indirim_uygula":
        try:
            import karar_tool as K
            K._gecmise_yaz(karar["magaza"], karar["tarih"],
                           float(karar.get("indirim_orani") or 0))
        except Exception as e:                       # noqa: BLE001
            print(f"Indirim gecmisine yazilamadi: {e}")

    mesaj = ("Karar onaylandi ve uygulandi." if durum == "onaylandi"
             else f"Karar reddedildi. Sebep: {sebep or 'belirtilmedi'}")
    return OnayYaniti(kaydedildi=True, karar_id=istek.karar_id,
                      durum=durum, sebep=sebep, mesaj=mesaj)


@app.get("/onay-ozet")
def onay_ozet():
    """Onay kayitlarinin hizli ozeti. Ayrintili analiz: ogrenme_dongusu.py"""
    if not ONAY_LOG.exists():
        return {"toplam": 0, "mesaj": "Henuz onay kaydi yok."}

    kayitlar = []
    for satir in ONAY_LOG.read_text(encoding="utf-8").splitlines():
        if satir.strip():
            try:
                kayitlar.append(json.loads(satir))
            except json.JSONDecodeError:
                continue

    onay = sum(1 for k in kayitlar if k["durum"] == "onaylandi")
    red = sum(1 for k in kayitlar if k["durum"] == "reddedildi")
    sebepler: dict[str, int] = {}
    for k in kayitlar:
        if k["durum"] == "reddedildi" and k.get("sebep"):
            sebepler[k["sebep"]] = sebepler.get(k["sebep"], 0) + 1

    return {
        "toplam": len(kayitlar),
        "onaylandi": onay,
        "reddedildi": red,
        "onay_orani": round(onay / len(kayitlar), 3) if kayitlar else None,
        "red_sebepleri": sebepler,
    }


@app.get("/ogrenme-raporu")
def ogrenme_raporu(min_gozlem: int = O.VARSAYILAN_MIN):
    """Onay/red kayitlarindan esik onerisi. Esikleri DEGISTIRMEZ."""
    if not O.LOG.exists():   # kayitlari_oku() dosya yoksa sys.exit yapar
        return {"benzersiz_karar": 0, "mesaj": "Henuz onay kaydi yok."}
    with contextlib.redirect_stdout(io.StringIO()):
        return O.rapor(O.kayitlari_oku(), min_gozlem)


@app.get("/trend")
def trend(yenile: bool = False):
    """Haftalik drift + karar trendi. Sonuc 24 saat onbellekte tutulur;
    yenile=true ile yeniden hesaplanir (train.csv okunur, ~10 sn)."""
    return TR.trend_yukle(yenile=yenile)


@app.get("/gunluk-rapor")
def gunluk_rapor():
    """n8n Schedule her sabah cagirir: alarm bayragi + Telegram metni +
    Sheets'e yazilacak gecmis satiri. Karar URETMEZ, esik DEGISTIRMEZ."""
    return TR.gunluk_rapor()


@app.post("/temizle")
def gecmis_temizle(chat_id: str = "default"):
    gecmisler.pop(chat_id, None)
    return {"durum": "temizlendi"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)