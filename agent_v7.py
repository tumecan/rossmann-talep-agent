"""
agent_v7.py — Cok magazali soru-cevap ajani (Rossmann stok karar sistemi)

v6 ILE ILISKISI
    v6 KARAR ajanidir: tek magaza, tek gun, sabit semali JSON uretir, kural
    motoru denetler. Yerinde duruyor, degismedi.
    v7 SORU-CEVAP ajanidir: 1115 magaza uzerinde serbest soru alir, dogru
    tool'u secer, cevabi duz metin verir. Sema yok cunku cikti bir karar
    degil bir rapordur.

MIMARI ILKE (v6'dan devralindi, burada daha da katidir)
    LLM hicbir sayi URETMEZ. Tum sayilar tools_toplu.py'deki deterministik
    tablolardan gelir. LLM'in tek isi YONLENDIRME'dir: hangi soruya hangi
    tool. Bu yuzden v7'de kural motoru yoktur — denetlenecek uretilmis sayi
    yoktur. Denetim ROUTING EVAL SETI ile yapilir.

IKINCI ILKE — kritik bilgi ayri tool'a birakilmaz
    v6'da ogrenilen ders ((i) ve (j) maddeleri): savunma katmani, korudugu
    katmanin davranisina bagimli olmamali. Burada karsiligi su: magaza
    listeleyen her tool, o magazanin tahmin hatasini (tahmin_zayif_mi) da
    dondurur. Ajan model_performansi'yi cagirmayi unutsa bile guvenilirlik
    bilgisi cevapta olur.

TOOL'LAR (10)
    1) tukenme_riski          -> hangi magazalarda stok tukenir
    2) sevkiyat_plani_getir   -> belirli gun kimlere ne gonderilecek
    3) en_cok_ihtiyac_duyan   -> onceligi kime verelim
    4) segment_ozeti          -> magaza tipi / urun yelpazesi kirilimi
    5) model_performansi      -> bu magazanin tahmini ne kadar guvenilir
    6) raf_omru_durumu        -> kimde raf omru doluyor, fire ne kadar
    7) dagitim_plani          -> elimde X birim var, nasil dagitayim
    8) magaza_karsilastir     -> iki magazayi yan yana koy
    9) promo_senaryosu        -> promosyon yapsak talep ne olur
   10) tedarikci_bilgisi_ara  -> RAG: sozlesme, politika, gecmis analiz

CANLI DEMO DAYANIKLILIGI (juri ilk 5 mesajda bunlari dener)
    - Olmayan magaza (1116, 0, "Migros")   -> tool net hata doner
    - Pencere disi tarih (2015-06-26)      -> tool net hata doner
    - Kapsam disi soru ("hava nasil")      -> prompt reddeder
    - Prompt injection                     -> tool ciktisi VERI sayilir
    - Konusma hafizasi ("peki 732?")       -> mesaj gecmisi tasinir

Calistirma:
    python agent_v7.py                      # etkilesimli sohbet
    python agent_v7.py --soru "..."         # tek soru
    python agent_v7.py --trace              # tool cagrilarini goster
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import tools_toplu as T
import karar_tool as K
import trend_analizi as TR
from agent_v5_rag import tedarikci_bilgisi_ara
from rag_index import indeks_yukle

MODEL_ADI = "gpt-5.5"
PROMPT_SURUM = "asistan_promptu_v2"
KOK = Path(__file__).resolve().parent
ROSSMAN = KOK / "Rossman"
LOG_JSONL = ROSSMAN / "agent_v7_log.jsonl"

# Chroma istemcisi ANA THREAD'de acilir (v6 hata (e): LangGraph tool'lari
# worker thread'de calistirir, chromadb'nin Rust baglayicisi orada cokuyor).
indeks_yukle()

_BAS, _BIT = T._pencere()
DEMO_BUGUN = T._meta().get("demo_bugun", "2015-07-03")


# ===========================================================================
# TOOL SARMALAYICILARI
# Gercek is tools_toplu.py'de. Docstring'ler LLM'in tool secimini belirler.
# ===========================================================================


@tool
def karar_uret(magaza: int, tarih: str) -> str:
    """Belirli bir magaza ve tarih icin YAPISAL STOK KARARI uretir:
    siparis_ver, bekle, indirim_uygula veya insana_sor. Miktar, aciliyet,
    tedarikci, net etki, onay gerekip gerekmedigi ve gerekce dahil tek bir
    karar dondurur. "X magazasi icin ne yapmaliyim", "siparis verelim mi",
    "karar ver", "aksiyon onerisi" gibi KARAR sorularinda bu araci kullan.
    Sadece durum bilgisi isteniyorsa diger araclari kullan."""
    return K.karar_uret(magaza, tarih)

@tool
def tukenme_riski(baslangic: str = "", bitis: str = "", n: int = 10) -> str:
    """Belirtilen tarih araliginda stogu tukenecek magazalari, en erken
    tukenenden baslayarak listeler. YUKSEK TALEP (p90) senaryosuna ve siparis
    verilmezse varsayimina dayanir; en kotu durum analizidir. Tarihler
    YYYY-MM-DD; bos birakilirsa tum tahmin penceresi. n en fazla 25.
    "Hangi magazalarda stok tukenir", "2 hafta icinde kim tukenir",
    "risk altindaki magazalar" sorularinda bu araci kullan."""
    return T.tukenme_riski(baslangic, bitis, n)


@tool
def sevkiyat_plani_getir(tarih: str, n: int = 15) -> str:
    """Belirtilen SIPARIS TARIHINDE hangi magazaya ne kadar gonderilecegini,
    hangi tedarikciden, ne zaman varacagini, tutari ve onay durumunu listeler.
    Tarih YYYY-MM-DD. n en fazla 25.
    "Kimlere ne gonderelim", "bugun sevkiyat var mi", "6 Temmuz plani"
    sorularinda bu araci kullan."""
    return T.sevkiyat_plani_getir(tarih, n)


@tool
def en_cok_ihtiyac_duyan(baslangic: str = "", bitis: str = "", n: int = 10) -> str:
    """Belirtilen aralikta ihtiyaci en buyuk olan magazalari siralar. Olcut,
    kacirilan satisin magazanin KENDI gunluk talebine oranidir; boylece buyuk
    magazalar listeyi doldurmaz, kitlik yasayan kucuk magazalar da gorunur.
    n en fazla 25.
    "En cok ihtiyaci olan magazalar", "onceligi kime verelim", tek bir
    magazanin ihtiyacli olup olmadigi sorularinda bu araci kullan."""
    return T.en_cok_ihtiyac_duyan(baslangic, bitis, n)


@tool
def segment_ozeti(magaza_tipi: str = "", urun_yelpazesi: str = "") -> str:
    """Bir magaza segmentinin talep, stok, tahmin dogrulugu ve onay profilini
    ozetler. magaza_tipi 'a','b','c','d'; urun_yelpazesi 'a','b','c' olabilir.
    Ikisi de bos birakilirsa tum segmentler karsilastirmali gelir.
    "B tipi magazalar nasil", "hangi segment daha riskli", "magaza tipleri
    arasinda fark var mi" sorularinda bu araci kullan."""
    return T.segment_ozeti(magaza_tipi, urun_yelpazesi)


@tool
def model_performansi(magaza: int) -> str:
    """Belirli bir magaza icin tahmin modelinin ne kadar guvenilir oldugunu
    raporlar: sMAPE hata orani, kapsama ve zincir ortalamasiyla kiyas, ayrica
    1115 magaza icindeki hata siralamasi.
    "Bu magazanin tahmini ne kadar guvenilir", "bu sayilara guvenebilir
    miyim" sorularinda kullan. Bir magaza icin ayrintili guvenilirlik
    dokumu gerekiyorsa bu araci cagir."""
    return T.model_performansi(magaza)


@tool
def raf_omru_durumu(tarih: str, n: int = 10) -> str:
    """Belirtilen tarihte hangi magazalarda malin raf omru dolmak uzere
    oldugunu, o gun ne kadar fire verilecegini ve zincir genelinde kalan raf
    omru dagilimini listeler. Tarih YYYY-MM-DD. n en fazla 25.
    "Kimde raf omru doluyor", "bugun ne kadar fire var", "bayat stok kimde"
    sorularinda bu araci kullan."""
    return T.raf_omru_durumu(tarih, n)


@tool
def dagitim_plani(toplam_stok: float, tarih: str, n: int = 10) -> str:
    """Elde belirli bir miktar stok varsa bunun magazalara nasil
    dagitilacagini hesaplar. Dagitim, magazalarin o gunku karsilanmamis
    ihtiyaci oraninda yapilir. toplam_stok adet, tarih YYYY-MM-DD.
    "Elimde 50 bin birim var nasil dagitayim", "kisitli stogu kime
    verelim" sorularinda bu araci kullan."""
    return T.dagitim_plani(toplam_stok, tarih, n)


@tool
def magaza_karsilastir(magaza_a: int, magaza_b: int,
                       baslangic: str = "", bitis: str = "") -> str:
    """Iki magazayi talep, stok, risk, siparis, fire ve tahmin dogrulugu
    bakimindan yan yana karsilastirir. Tarihler bos birakilirsa tum pencere.
    "A ile B'yi karsilastir", "hangisi daha riskli", "530 mu 1 mi daha
    kotu durumda" sorularinda bu araci kullan."""
    return T.magaza_karsilastir(magaza_a, magaza_b, baslangic, bitis)


@tool
def promo_senaryosu(magaza: int, tarih: str = "") -> str:
    """Bir magazada promosyon yapilmasi durumunda talebin ne kadar
    artacagini gosterir. Tarih bos birakilirsa tum pencere ortalamasi doner.
    "Promosyon yapsak ne olur", "kampanya talebi ne kadar artirir"
    sorularinda bu araci kullan."""
    return T.promo_senaryosu(magaza, tarih)


@tool
def trend_ozeti() -> str:
    """Zincir genelinde tahmin modelinin ve kararlarin ZAMAN ICINDEKI
    degisimini haftalik olarak raporlar: sMAPE, kapsama, p90 ustu ihlal,
    bias, onay yuku, kacirilan risk ve drift alarmlari. Parametre almaz.
    "Model zamanla kotulesiyor mu", "haftalik trend nasil", "drift var
    mi", "tahmin performansi haftadan haftaya nasil degisti" sorularinda
    bu araci kullan. Tek magaza guvenilirligi icin model_performansi
    kullanilir; bu arac zincir geneli ve zaman eksenlidir."""
    return TR.trend_ozeti()


TOOLS = [tukenme_riski, sevkiyat_plani_getir, en_cok_ihtiyac_duyan,
         segment_ozeti, model_performansi, raf_omru_durumu, dagitim_plani,
         magaza_karsilastir, promo_senaryosu, tedarikci_bilgisi_ara, karar_uret,
         trend_ozeti]


# ===========================================================================
# SISTEM PROMPTU — dosyadan okunur, kodda gomulu degildir
# ===========================================================================
# Prompt bir KOD PARCASI degil, bir YAPILANDIRMADIR. Dosyaya alinmasinin
# sebepleri: (1) surumlenebilir — degistiginde dosya adi degisir ve her
# karar logunda hangi surumle uretildigi kayitli kalir, (2) prompt'u
# degistirmek icin kod dosyasina dokunmak gerekmez, (3) teslim kalemi
# "agent prompt'lari ayri dosyada" bunu sart kosuyor.
PROMPT_SURUM = "asistan_promptu_v2"
PROMPT_DOSYA = KOK / "prompts" / f"{PROMPT_SURUM}.txt"
if not PROMPT_DOSYA.exists():
    raise SystemExit(f"HATA: prompt dosyasi bulunamadi: {PROMPT_DOSYA}")

SISTEM_PROMPT = PROMPT_DOSYA.read_text(encoding="utf-8").format(
    bugun=DEMO_BUGUN,
    pencere_bas=_BAS.date(),
    pencere_bit=_BIT.date(),
)


# ===========================================================================
def agent_olustur():
    load_dotenv()
    anahtar = os.getenv("openai_apikey")
    if not anahtar:
        sys.exit("HATA: .env icinde openai_apikey yok.")
    llm = ChatOpenAI(model=MODEL_ADI, temperature=0,
                     api_key=anahtar.strip().strip('"'))
    return create_agent(model=llm, tools=TOOLS, system_prompt=SISTEM_PROMPT)


def logla(kayit: dict) -> None:
    ROSSMAN.mkdir(parents=True, exist_ok=True)
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")


def sor(agent, soru: str, gecmis: list, trace=False) -> tuple[str, list, list]:
    """Doner: (cevap, guncel_gecmis, tool_adimlari)

    tool_adimlari artik her adim icin CIKTIYI da tasir. Eslestirme
    tool_call_id uzerinden yapilir; sirali eslestirme paralel tool
    cagrilarinda kayar.
    """
    sonuc = agent.invoke({"messages": gecmis + [{"role": "user", "content": soru}]})

    harita: dict[str, dict] = {}
    sira: list[str] = []
    for m in sonuc["messages"]:
        for c in (getattr(m, "tool_calls", []) or []):
            harita[c["id"]] = {"tool": c["name"], "args": c["args"], "cikti": None}
            sira.append(c["id"])
    for m in sonuc["messages"]:
        tcid = getattr(m, "tool_call_id", None)
        if tcid in harita:
            harita[tcid]["cikti"] = m.content
    adimlar = [harita[i] for i in sira]

    cevap = sonuc["messages"][-1].content
    yeni_gecmis = gecmis + [
        {"role": "user", "content": soru},
        {"role": "assistant", "content": cevap},
    ]
    if len(yeni_gecmis) > 12:
        yeni_gecmis = yeni_gecmis[-12:]

    logla({
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_ADI, "prompt_surum": PROMPT_SURUM,
        "soru": soru, "cevap": cevap, "tool_sayisi": len(adimlar),
        # cikti log'da kisaltilir — dosya sismesin
        "tool_trace": [{"tool": a["tool"], "args": a["args"],
                        "cikti_ozet": (a["cikti"] or "")[:500]} for a in adimlar],
    })
    if trace and adimlar:
        print("\n--- TOOL TRACE ---")
        for i, a in enumerate(adimlar, 1):
            print(f"  {i}. {a['tool']}({json.dumps(a['args'], ensure_ascii=False)})")
    return cevap, yeni_gecmis, adimlar


def son_karar(adimlar: list) -> dict | None:
    """karar_uret cagrildiysa yapisal karari dondurur, yoksa None."""
    for a in reversed(adimlar):
        if a.get("tool") == "karar_uret" and a.get("cikti"):
            try:
                k = json.loads(a["cikti"])
            except (json.JSONDecodeError, TypeError):
                return None
            return k if isinstance(k, dict) and k.get("olay") == "stok_karari" else None
    return None


def sohbet(agent, trace=False) -> None:
    print("=" * 70)
    print(f"Rossmann Stok Asistani v7 | bugun {DEMO_BUGUN} | "
          f"pencere {_BAS.date()} - {_BIT.date()} | {len(TOOLS)} arac")
    print("Cikis: 'q' | Hafizayi temizle: 'yeni'")
    print("=" * 70)
    gecmis: list = []
    while True:
        try:
            soru = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\ngorusuruz.")
            return
        if not soru:
            continue
        if soru.lower() in {"q", "quit", "exit", "cikis"}:
            print("gorusuruz.")
            return
        if soru.lower() in {"yeni", "temizle", "reset"}:
            gecmis = []
            print("(hafiza temizlendi)")
            continue
        try:
            cevap, gecmis, adimlar = sor(agent, soru, gecmis, trace)
        except Exception as e:
            print(f"HATA: {type(e).__name__}: {e}")
            continue
        print(f"\n{cevap}")
        if not trace and adimlar:
            print(f"\n[{len(adimlar)} arac: "
                  f"{', '.join(a['tool'] for a in adimlar)}]")


if __name__ == "__main__":
    trace = "--trace" in sys.argv
    agent = agent_olustur()
    if "--soru" in sys.argv:
        s = sys.argv[sys.argv.index("--soru") + 1]
        cevap, _, adimlar = sor(agent, s, [], trace)
        print(f"\n{cevap}")
        print(f"\n[{len(adimlar)} arac: "
              f"{', '.join(a['tool'] for a in adimlar) or 'yok'}]")
    else:
        sohbet(agent, trace)