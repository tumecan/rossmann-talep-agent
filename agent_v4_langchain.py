"""
agent_v4_langchain.py — Tool kullanan Karar Agent'i (Rossmann, Faz 5 - adim 3)

v3'e gore fark: artik tek prompt degil, GERCEK AGENT.
LangChain 1.x create_agent ile kurulu, uc tool'u kendi karariyla cagiriyor:
    1) tahmin_getir      -> Prophet tahmini + guven araligi
    2) stok_getir        -> o gunun mevcut stogu (varsayim)
    3) ihtiyac_hesapla   -> deterministik aritmetik (LLM sayi uretmez)

Aritmetik hala Python'da; fark su ki agent bu hesabi bir ARAC olarak cagiriyor.
Kural motoru (agent_v2.kurallari_dayat) arkada savunma katmani olarak duruyor.

Onkosul:
    python Rossman\\prophet_model.py   -> Rossman/prophet_forecast.csv

Calistirma:
    python agent_v4_langchain.py                      # son gun
    python agent_v4_langchain.py --tarih 2015-07-15   # belirli gun
    python agent_v4_langchain.py --trace              # tool cagrilarini goster
"""

import json
import os
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# v2/v3'teki katmanlar aynen kullaniliyor
from agent_v2 import (
    AgentKarar,
    CIKTI_DIR,
    hesapla,
    kurallari_dayat,
    logla,
)
from agent_v3 import STOK_ORANI, forecast_yukle

MODEL_ADI = "gpt-4o-mini"
PROMPT_SURUM = "v4"
PROMPT_DOSYA = CIKTI_DIR.parent / "prompts" / f"karar_promptu_{PROMPT_SURUM}.txt"
if not PROMPT_DOSYA.exists():                      # Rossman altklasoru yoksa
    PROMPT_DOSYA = CIKTI_DIR / "prompts" / f"karar_promptu_{PROMPT_SURUM}.txt"
KARAR_JSON = CIKTI_DIR / "agent_karar_v4.json"

# Tool'larin erisecegi veri — TEMBEL yuklenir. v4'un yasadigi donemde bu
# modul seviyesinde tek seferlik yukleniyordu; ama v5/v6/v7 zinciri bu
# dosyayi yalniz ihtiyac_hesapla() (saf aritmetik, DF kullanmaz) icin
# import ediyor. Modul seviyesinde forecast_yukle() cagirmak, prophet
# tabanli tahmin_getir/stok_getir HIC cagirilmasa bile SADECE IMPORT ETMEK
# icin Rossman/prophet_forecast.csv'nin var olmasini sart kosuyordu — bu
# dosya standart kurulum siralamasinda (README SS.8) hic uretilmiyor.
# Sonuc: agent_v7 -> agent_v5_rag -> agent_v4_langchain zinciri, hic
# kullanilmayan bir Prophet artefaktinin yokluğunda import anında
# sys.exit ile cokuyordu. Cozum: DF'i ilk fiilen kullanildigi anda yukle.
_DF_ONBELLEK: pd.DataFrame | None = None


def _df() -> pd.DataFrame:
    global _DF_ONBELLEK
    if _DF_ONBELLEK is None:
        _DF_ONBELLEK = forecast_yukle()
    return _DF_ONBELLEK


# ============================================================================
# TOOL'LAR — agent bunlari kendi karariyla cagirir
# ============================================================================
@tool
def tahmin_getir(tarih: str) -> str:
    """Belirtilen tarih icin Prophet modelinin satis tahminini ve %90 guven
    araligini dondurur. Tarih formati: YYYY-MM-DD.
    Doner: tahmin_satis, guven_alt, guven_ust, promo_var_mi."""
    df = _df()
    satir = df[df["ds"] == pd.Timestamp(tarih)]
    if satir.empty:
        return (f"HATA: {tarih} icin tahmin yok. "
                f"Gecerli aralik: {df['ds'].min().date()} - {df['ds'].max().date()}")
    s = satir.iloc[0]
    return json.dumps({
        "tarih": tarih,
        "tahmin_satis": round(float(s["yhat"]), 1),
        "guven_alt": round(float(s["yhat_lower"]), 1),
        "guven_ust": round(float(s["yhat_upper"]), 1),
        "promo_var_mi": bool(s["promo"]),
    }, ensure_ascii=False)


@tool
def stok_getir(tarih: str) -> str:
    """Belirtilen tarihte magazada mevcut olan stok miktarini dondurur.
    Tarih formati: YYYY-MM-DD. Doner: mevcut_stok."""
    df = _df()
    satir = df[df["ds"] == pd.Timestamp(tarih)]
    if satir.empty:
        return f"HATA: {tarih} icin stok bilgisi yok."
    return json.dumps({
        "tarih": tarih,
        "mevcut_stok": float(satir.iloc[0]["mevcut_stok"]),
        "not": f"Bir onceki gunun satisinin %{int(STOK_ORANI*100)}'i varsayimi",
    }, ensure_ascii=False)


@tool
def ihtiyac_hesapla(tahmin_satis: float, guven_alt: float, guven_ust: float,
                    mevcut_stok: float, promo_var_mi: bool) -> str:
    """Tahmin, guven araligi ve stok bilgisinden siparis ihtiyacini hesaplar.
    Emniyet stogunu, gereken toplami, acigi ve onerilen miktari dondurur.
    Ayrica guven araliginin iki ucunda karar degisiyorsa belirsiz_mi=true doner.
    ASLA bu hesabi kendin yapma, her zaman bu araci kullan."""
    hesap = hesapla({
        "tahmin_satis": tahmin_satis,
        "guven_alt": guven_alt,
        "guven_ust": guven_ust,
        "mevcut_stok": mevcut_stok,
        "promo_var_mi": promo_var_mi,
    })
    return json.dumps(hesap, ensure_ascii=False)


TOOLS = [tahmin_getir, stok_getir, ihtiyac_hesapla]


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

    llm = ChatOpenAI(
        model=MODEL_ADI,
        temperature=0,
        api_key=anahtar.strip().strip('"'),
    )
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=PROMPT_DOSYA.read_text(encoding="utf-8"),
        response_format=AgentKarar,     # cikti semaya zorlanir
    )


def trace_yazdir(mesajlar) -> list[dict]:
    """Agent'in hangi tool'u hangi argumanla cagirdigini gosterir."""
    adimlar = []
    for m in mesajlar:
        for cagri in getattr(m, "tool_calls", []) or []:
            adimlar.append({"tool": cagri["name"], "args": cagri["args"]})
    return adimlar


def karar_ver(agent, tarih: str, trace: bool = False) -> dict:
    soru = (
        f"{tarih} tarihi icin Magaza 1'de siparis verilmeli mi? "
        "Once tahmini, sonra stogu getir, sonra ihtiyaci hesapla ve karar ver."
    )
    sonuc = agent.invoke({"messages": [{"role": "user", "content": soru}]})

    adimlar = trace_yazdir(sonuc["messages"])
    ham_karar: AgentKarar = sonuc["structured_response"]

    # Bagimsiz dogrulama: kod kendi hesabini yapar, kural motoru uygular
    satir = _df()[_df()["ds"] == pd.Timestamp(tarih)].iloc[0]
    girdi = {
        "magaza": 1,
        "tarih": tarih,
        "tahmin_satis": round(float(satir["yhat"]), 1),
        "guven_alt": round(float(satir["yhat_lower"]), 1),
        "guven_ust": round(float(satir["yhat_upper"]), 1),
        "mevcut_stok": float(satir["mevcut_stok"]),
        "promo_var_mi": bool(satir["promo"]),
    }
    hesap = hesapla(girdi)
    karar, mudahaleler = kurallari_dayat(ham_karar, hesap)

    print(f"\n========== {tarih} | Magaza 1 ==========")
    print(f"Agent {len(adimlar)} tool cagrisi yapti: "
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
        print("\n(kural motoru mudahale etmedi - agent ciktisi tutarli)")

    kayit = {
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_ADI,
        "prompt_surum": PROMPT_SURUM,
        "mimari": "langchain create_agent (3 tool)",
        "tool_trace": adimlar,
        "girdi": girdi,
        "hesaplanan": hesap,
        "karar": karar.model_dump(),
        "mudahaleler": mudahaleler,
    }
    KARAR_JSON.write_text(json.dumps(kayit, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    logla(kayit)
    print(f"\nKaydedildi: {KARAR_JSON}")
    return kayit


if __name__ == "__main__":
    if "--tarih" in sys.argv:
        tarih = sys.argv[sys.argv.index("--tarih") + 1]
    else:
        tarih = _df().iloc[-1]["ds"].strftime("%Y-%m-%d")

    agent = agent_olustur()
    karar_ver(agent, tarih, trace="--trace" in sys.argv)