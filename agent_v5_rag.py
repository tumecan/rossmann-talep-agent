"""
agent_v5_rag.py — RAG destekli Karar Agent'i (Rossmann, Faz 5 - adim 4 / FINAL)

v4'e gore fark: dorduncu tool eklendi. Agent artik sadece "kac birim siparis
verilmeli" degil, "KIMDEN, ne zaman, hangi kosulla" sorusunu da cevapliyor.
Bu bilgi yapilandirilmis veride yok; tedarikci sozlesmelerinden hibrit
arama (BM25 + dense, RRF) ile geliyor.

TOOL'LAR
    1) tahmin_getir             -> Prophet tahmini + guven araligi
    2) stok_getir               -> mevcut stok (varsayim)
    3) ihtiyac_hesapla          -> deterministik aritmetik
    4) tedarikci_bilgisi_ara    -> RAG: sozlesme ve politika belgeleri

Onkosullar:
    python Rossman\\prophet_model.py
    python tedarikci_dokumanlari_olustur.py
    python rag_index.py
    pip install rank_bm25

Calistirma:
    python agent_v5_rag.py                      # son gun
    python agent_v5_rag.py --tarih 2015-07-09   # belirli gun
    python agent_v5_rag.py --trace              # tool cagrilarini goster
    python agent_v5_rag.py --toplu              # tum test gunleri + CSV
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
from pydantic import BaseModel, Field

from agent_v2 import CIKTI_DIR, hesapla, logla
from agent_v3 import forecast_yukle
from agent_v4_langchain import ihtiyac_hesapla, stok_getir, tahmin_getir
from rag_hybrid import hibrit_ara

MODEL_ADI = "gpt-4o-mini"
PROMPT_SURUM = "v5"
ONAY_ESIGI = 3000          # satinalma politikasi: ustunde bolge muduru onayi

PROMPT_DOSYA = CIKTI_DIR.parent / "prompts" / f"karar_promptu_{PROMPT_SURUM}.txt"
if not PROMPT_DOSYA.exists():
    PROMPT_DOSYA = CIKTI_DIR / "prompts" / f"karar_promptu_{PROMPT_SURUM}.txt"
KARAR_JSON = CIKTI_DIR / "agent_karar_v5.json"
TOPLU_CSV = CIKTI_DIR / "agent_kararlari_v5.csv"

# Ayni sizinti/coku deseni agent_v4_langchain'de de vardi (bkz. o dosyadaki
# not) ve bagimsiz olarak burada tekrarlanmis: agent_v7, tedarikci_bilgisi_ara
# icin bu modulu import ediyor ama DF'i (Prophet, magaza 1) HIC kullanmiyor.
# Tembel yukleme: import anında degil, ilk fiili kullanimda okunur.
_DF_ONBELLEK: pd.DataFrame | None = None


def _df() -> pd.DataFrame:
    global _DF_ONBELLEK
    if _DF_ONBELLEK is None:
        _DF_ONBELLEK = forecast_yukle()
    return _DF_ONBELLEK


# ============================================================================
# GENISLETILMIS CIKTI SEMASI
# ============================================================================
class AgentKararRAG(BaseModel):
    karar: Literal["siparis_ver", "bekle", "insana_sor"]
    miktar: int = Field(ge=0)
    aciliyet: Literal["yuksek", "orta", "dusuk"]
    gerekce: str = Field(min_length=10, max_length=400)
    # --- RAG ile gelen alanlar ---
    tedarikci: str = Field(description="Secilen tedarikcinin adi")
    teslimat_notu: str = Field(max_length=300,
                               description="Lead-time, minimum miktar, iskonto")
    onay_gerekli_mi: bool = Field(description="Insan onayi gerekiyor mu")


# ============================================================================
# 4. TOOL — RAG
# ============================================================================
@tool
def tedarikci_bilgisi_ara(soru: str) -> str:
    """Tedarikci sozlesmelerinde ve ic satinalma politikasinda arama yapar.
    Lead-time (teslim suresi), minimum siparis miktari, iskonto kademeleri,
    teslimat gunleri, onay esikleri ve tedarikci secim kurallari bu araçla
    ogrenilir. Dogal dilde soru sor, ornek: 'promosyon doneminde hangi
    tedarikci kullanilir', '3000 birim uzeri onay esigi'."""
    parcalar = hibrit_ara(soru, k=3)
    if not parcalar:
        return "Ilgili belge bulunamadi."
    bloklar = [
        f"[kaynak: {p.metadata.get('kaynak')}]\n{p.page_content.strip()}"
        for p in parcalar
    ]
    return "\n\n---\n\n".join(bloklar)


TOOLS = [tahmin_getir, stok_getir, ihtiyac_hesapla, tedarikci_bilgisi_ara]


# ============================================================================
# KURAL MOTORU — RAG alanlarini koruyarak
# ============================================================================
def kurallari_dayat_rag(karar: AgentKararRAG, hesap: dict):
    mudahaleler = []
    veri = karar.model_dump()

    if veri["miktar"] != hesap["onerilen_miktar"]:
        mudahaleler.append(
            f"miktar duzeltildi: {veri['miktar']} -> {hesap['onerilen_miktar']}")
        veri["miktar"] = hesap["onerilen_miktar"]

    if hesap["belirsiz_mi"] and veri["karar"] != "insana_sor":
        mudahaleler.append(
            f"karar duzeltildi: {veri['karar']} -> insana_sor (belirsizlik)")
        veri["karar"] = "insana_sor"

    if not hesap["belirsiz_mi"] and hesap["ihtiyac"] <= 0 and veri["karar"] == "siparis_ver":
        mudahaleler.append("karar duzeltildi: siparis_ver -> bekle (ihtiyac yok)")
        veri["karar"] = "bekle"
        veri["miktar"] = 0

    # Politika kurali: esik ustu VEYA belirsiz karar -> otomatik onay yok
    gerekli = veri["miktar"] > ONAY_ESIGI or veri["karar"] == "insana_sor"
    if veri["onay_gerekli_mi"] != gerekli:
        mudahaleler.append(
            f"onay_gerekli_mi duzeltildi: {veri['onay_gerekli_mi']} -> {gerekli}")
        veri["onay_gerekli_mi"] = gerekli

    return AgentKararRAG(**veri), mudahaleler


# ============================================================================
# AGENT
# ============================================================================
def agent_olustur():
    load_dotenv()
    anahtar = os.getenv("openai_apikey")
    if not anahtar:
        sys.exit("HATA: .env icinde openai_apikey yok.")
    if not PROMPT_DOSYA.exists():
        sys.exit(f"HATA: {PROMPT_DOSYA} yok. Once prompt dosyasini olustur.")

    llm = ChatOpenAI(model=MODEL_ADI, temperature=0,
                     api_key=anahtar.strip().strip('"'))
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=PROMPT_DOSYA.read_text(encoding="utf-8"),
        response_format=AgentKararRAG,
    )


def girdi_hazirla(tarih: str) -> dict:
    satir = _df()[_df()["ds"] == pd.Timestamp(tarih)].iloc[0]
    return {
        "magaza": 1,
        "tarih": tarih,
        "tahmin_satis": round(float(satir["yhat"]), 1),
        "guven_alt": round(float(satir["yhat_lower"]), 1),
        "guven_ust": round(float(satir["yhat_upper"]), 1),
        "mevcut_stok": float(satir["mevcut_stok"]),
        "promo_var_mi": bool(satir["promo"]),
    }


def karar_ver(agent, tarih: str, trace: bool = False, yazdir: bool = True) -> dict:
    soru = (
        f"{tarih} tarihi icin Magaza 1'de siparis karari ver. "
        "Once tahmini ve stogu getir, ihtiyaci hesapla, sonra hangi tedarikciden "
        "alinacagini ve onay gerekip gerekmedigini belgelerden arastir."
    )
    sonuc = agent.invoke({"messages": [{"role": "user", "content": soru}]})

    adimlar = [
        {"tool": c["name"], "args": c["args"]}
        for m in sonuc["messages"] for c in (getattr(m, "tool_calls", []) or [])
    ]
    ham_karar: AgentKararRAG = sonuc["structured_response"]

    girdi = girdi_hazirla(tarih)
    hesap = hesapla(girdi)
    karar, mudahaleler = kurallari_dayat_rag(ham_karar, hesap)

    if yazdir:
        print(f"\n========== {tarih} | Magaza 1 ==========")
        print(f"Tool zinciri ({len(adimlar)} cagri): "
              f"{' -> '.join(a['tool'] for a in adimlar)}")
        if trace:
            print("\n--- TOOL TRACE ---")
            for i, a in enumerate(adimlar, 1):
                print(f"  {i}. {a['tool']}({json.dumps(a['args'], ensure_ascii=False)})")
        print("\n--- Agent karari ---")
        print(json.dumps(karar.model_dump(), ensure_ascii=False, indent=2))
        if mudahaleler:
            print("\n--- Kural motoru mudahaleleri ---")
            for m in mudahaleler:
                print("  *", m)
        else:
            print("\n(kural motoru mudahale etmedi)")

    kayit = {
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_ADI,
        "prompt_surum": PROMPT_SURUM,
        "mimari": "langchain create_agent (4 tool, RAG dahil)",
        "tool_trace": adimlar,
        "girdi": girdi,
        "hesaplanan": hesap,
        "karar": karar.model_dump(),
        "mudahaleler": mudahaleler,
    }
    logla(kayit)
    return kayit


def toplu_calistir(agent) -> None:
    df = _df()
    print(f"\n########## TOPLU KARAR — RAG ({len(df)} gun) ##########")
    satirlar = []
    for _, s in df.iterrows():
        tarih = s["ds"].strftime("%Y-%m-%d")
        kayit = karar_ver(agent, tarih, yazdir=False)
        k, h, g = kayit["karar"], kayit["hesaplanan"], kayit["girdi"]
        satirlar.append({
            "tarih": tarih,
            "tahmin": g["tahmin_satis"],
            "stok": g["mevcut_stok"],
            "promo": int(g["promo_var_mi"]),
            "karar": k["karar"],
            "miktar": k["miktar"],
            "aciliyet": k["aciliyet"],
            "tedarikci": k["tedarikci"],
            "onay_gerekli": int(k["onay_gerekli_mi"]),
            "tool_sayisi": len(kayit["tool_trace"]),
            "teslimat_notu": k["teslimat_notu"],
            "gerekce": k["gerekce"],
            "gercek": round(float(s["gercek"]), 1),
        })
        print(f"{tarih} -> {k['karar']:12} {k['miktar']:6} | "
              f"{k['tedarikci'][:28]:28} | onay={'E' if k['onay_gerekli_mi'] else 'H'}")

    sonuc = pd.DataFrame(satirlar)
    sonuc.to_csv(TOPLU_CSV, index=False, encoding="utf-8-sig")

    print("\n--- Karar dagilimi ---")
    print(sonuc["karar"].value_counts().to_string())
    print("\n--- Tedarikci dagilimi ---")
    print(sonuc["tedarikci"].value_counts().to_string())
    print("\n--- Onay gerekliligi ---")
    print(f"  otomatik islenebilir : {(sonuc['onay_gerekli'] == 0).sum()} gun")
    print(f"  insan onayi gerekli  : {(sonuc['onay_gerekli'] == 1).sum()} gun")

    onayli = sonuc[sonuc["karar"] == "siparis_ver"]
    print(f"\nOtomatik onayli siparis : {onayli['miktar'].sum():,} ({len(onayli)} gun)")
    bekleyen = sonuc[sonuc["karar"] == "insana_sor"]
    print(f"Insan onayi bekleyen    : {bekleyen['miktar'].sum():,} ({len(bekleyen)} gun)")
    print(f"\nKaydedildi: {TOPLU_CSV}")


if __name__ == "__main__":
    agent = agent_olustur()

    if "--toplu" in sys.argv:
        toplu_calistir(agent)
    else:
        if "--tarih" in sys.argv:
            tarih = sys.argv[sys.argv.index("--tarih") + 1]
        else:
            tarih = _df().iloc[-1]["ds"].strftime("%Y-%m-%d")
        kayit = karar_ver(agent, tarih, trace="--trace" in sys.argv)
        KARAR_JSON.write_text(json.dumps(kayit, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"\nKaydedildi: {KARAR_JSON}")