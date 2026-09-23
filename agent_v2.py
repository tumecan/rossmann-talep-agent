"""
agent_v2.py — Güvenilir Karar Agent'ı (Rossmann bitirme projesi, Faz 5)

v1'e göre farklar:
  1) Sayıyı Python hesaplar, LLM sadece yorumlar  -> aynı girdi = aynı miktar
  2) Pydantic ile şema doğrulama + hatada 1 kez tekrar deneme
  3) Güven aralığı genişse karar zorla "insana_sor" (eskalasyon)
  4) Prompt ayrı dosyada versiyonlu + her çalıştırma agent_log.jsonl'e yazılır

Çalıştırma:
    python agent_v2.py            # tek karar üretir
    python agent_v2.py --test     # 3 kez çalıştırıp tutarlılığı kanıtlar
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# ----------------------------------------------------------------------------
# AYARLAR  (buradaki eşikleri sunumda "iş kuralı" olarak anlatabilirsin)
# ----------------------------------------------------------------------------
MODEL_ADI = "gpt-4o-mini"
PROMPT_SURUM = "v2"

EMNIYET_STOGU_ORANI = 0.15   # tahminin %15'i kadar tampon
PROMO_EK_ORAN = 0.10         # promosyon varsa %10 daha ekle
BELIRSIZLIK_ESIGI = 0.30     # (üst-alt)/tahmin bunu aşarsa -> insana_sor

BURASI = Path(__file__).resolve().parent
CIKTI_DIR = BURASI / "Rossman" if (BURASI / "Rossman").is_dir() else BURASI
PROMPT_DOSYA = BURASI / "prompts" / f"karar_promptu_{PROMPT_SURUM}.txt"
KARAR_JSON = CIKTI_DIR / "agent_karar.json"
LOG_DOSYA = CIKTI_DIR / "agent_log.jsonl"


# ----------------------------------------------------------------------------
# 1) ÇIKTI ŞEMASI — LLM ne dönerse dönsün buna uymak zorunda
# ----------------------------------------------------------------------------
class AgentKarar(BaseModel):
    karar: Literal["siparis_ver", "bekle", "insana_sor"]
    miktar: int = Field(ge=0)
    aciliyet: Literal["yuksek", "orta", "dusuk"]
    gerekce: str = Field(min_length=10, max_length=400)


# ----------------------------------------------------------------------------
# 2) DETERMİNİSTİK KATMAN — tüm aritmetik burada, LLM'e sayı yaptırmıyoruz
# ----------------------------------------------------------------------------
def hesapla(girdi: dict) -> dict:
    tahmin = float(girdi["tahmin_satis"])
    alt = float(girdi["guven_alt"])
    ust = float(girdi["guven_ust"])
    stok = float(girdi["mevcut_stok"])
    promo = bool(girdi["promo_var_mi"])

    tampon_oran = EMNIYET_STOGU_ORANI + (PROMO_EK_ORAN if promo else 0.0)
    emniyet_stogu = tahmin * tampon_oran

    gereken_toplam = tahmin + emniyet_stogu
    ihtiyac = gereken_toplam - stok
    onerilen_miktar = max(0, int(round(ihtiyac / 50.0) * 50))  # 50'lik paketler

    # --- KARAR DAYANIKLILIGI ---
    # Guven araliginin iki ucunda da ayni karara variyor muyuz?
    ihtiyac_alt = (alt + emniyet_stogu) - stok   # kotu senaryo (dusuk satis)
    ihtiyac_ust = (ust + emniyet_stogu) - stok   # iyi senaryo (yuksek satis)
    # Isaret degisiyorsa karar aralik icinde donuyor -> insana sor
    belirsiz_mi = (ihtiyac_alt <= 0) != (ihtiyac_ust <= 0)

    # bilgi amacli: aralik tahminin yuzde kaci (artik tetikleyici degil)
    belirsizlik_orani = (ust - alt) / tahmin if tahmin > 0 else 1.0

    return {
        "belirsizlik_orani": round(belirsizlik_orani, 4),
        "emniyet_stogu": round(emniyet_stogu, 1),
        "gereken_toplam": round(gereken_toplam, 1),
        "ihtiyac": round(ihtiyac, 1),
        "ihtiyac_alt": round(ihtiyac_alt, 1),
        "ihtiyac_ust": round(ihtiyac_ust, 1),
        "belirsiz_mi": belirsiz_mi,
        "onerilen_miktar": onerilen_miktar,
    }


# ----------------------------------------------------------------------------
# 3) LLM KATMANI — sadece karar + gerekçe üretir
# ----------------------------------------------------------------------------
def istemci_olustur() -> OpenAI:
    load_dotenv()
    anahtar = os.getenv("openai_apikey")
    if not anahtar:
        sys.exit("HATA: .env içinde openai_apikey bulunamadi.")
    return OpenAI(api_key=anahtar.strip().strip('"'))


def llm_cagir(client: OpenAI, sistem_prompt: str, kullanici_mesaji: str,
              ek_uyari: str = "") -> str:
    mesajlar = [
        {"role": "system", "content": sistem_prompt},
        {"role": "user", "content": kullanici_mesaji},
    ]
    if ek_uyari:
        mesajlar.append({"role": "user", "content": ek_uyari})

    cevap = client.chat.completions.create(
        model=MODEL_ADI,
        messages=mesajlar,
        response_format={"type": "json_object"},
        temperature=0,      # kararlılık için
        seed=42,            # aynı girdi -> aynı çıktı (best-effort)
    )
    return cevap.choices[0].message.content


def karar_uret(client: OpenAI, girdi: dict, hesap: dict) -> tuple[AgentKarar, int, str]:
    """Doğrulanmış karar döndürür. (karar, deneme_sayisi, ham_cikti)"""
    sistem_prompt = PROMPT_DOSYA.read_text(encoding="utf-8")

    kullanici_mesaji = (
        "girdi:\n" + json.dumps(girdi, ensure_ascii=False, indent=2) +
        "\n\nhesaplanan (bunlari aynen kullan, yeniden hesaplama):\n" +
        json.dumps(hesap, ensure_ascii=False, indent=2)
    )

    ek_uyari = ""
    ham = ""
    for deneme in (1, 2):
        ham = llm_cagir(client, sistem_prompt, kullanici_mesaji, ek_uyari)
        try:
            return AgentKarar.model_validate_json(ham), deneme, ham
        except ValidationError as e:
            print(f"[uyari] Sema dogrulama hatasi (deneme {deneme}). Tekrar deneniyor.")
            ek_uyari = (
                "Onceki cevabin sema dogrulamasindan gecmedi. Hatalar:\n"
                f"{e}\nSadece gecerli JSON dondur."
            )

    # İki denemede de olmadıysa güvenli varsayılan
    print("[uyari] LLM sema uretemedi -> guvenli varsayilana dusuldu.")
    return (
        AgentKarar(
            karar="insana_sor",
            miktar=hesap["onerilen_miktar"],
            aciliyet="orta",
            gerekce="Agent gecerli bir karar uretemedi, karar insana yonlendirildi.",
        ),
        3,
        ham,
    )


# ----------------------------------------------------------------------------
# 4) GÜVENLİK KATMANI — LLM ne derse desin iş kuralları galip
# ----------------------------------------------------------------------------
def kurallari_dayat(karar: AgentKarar, hesap: dict) -> tuple[AgentKarar, list[str]]:
    mudahaleler = []
    veri = karar.model_dump()

    if veri["miktar"] != hesap["onerilen_miktar"]:
        mudahaleler.append(
            f"miktar duzeltildi: {veri['miktar']} -> {hesap['onerilen_miktar']}"
        )
        veri["miktar"] = hesap["onerilen_miktar"]

    if hesap["belirsiz_mi"] and veri["karar"] != "insana_sor":
        mudahaleler.append(f"karar duzeltildi: {veri['karar']} -> insana_sor (belirsizlik)")
        veri["karar"] = "insana_sor"

    if not hesap["belirsiz_mi"] and hesap["ihtiyac"] <= 0 and veri["karar"] == "siparis_ver":
        mudahaleler.append("karar duzeltildi: siparis_ver -> bekle (ihtiyac yok)")
        veri["karar"] = "bekle"
        veri["miktar"] = 0

    return AgentKarar(**veri), mudahaleler


# ----------------------------------------------------------------------------
# 5) LOG — izlenebilirlik
# ----------------------------------------------------------------------------
def logla(kayit: dict) -> None:
    with open(LOG_DOSYA, "a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# ANA AKIŞ
# ----------------------------------------------------------------------------
def calistir(client: OpenAI, girdi: dict, yazdir: bool = True) -> AgentKarar:
    t0 = time.time()
    hesap = hesapla(girdi)
    ham_karar, deneme, ham_cikti = karar_uret(client, girdi, hesap)
    karar, mudahaleler = kurallari_dayat(ham_karar, hesap)
    sure = round(time.time() - t0, 2)

    if yazdir:
        print("\n========== HESAPLANAN (Python) ==========")
        for k, v in hesap.items():
            print(f"{k:22}: {v}")

        print("\n========== AGENT KARARI ==========")
        print(json.dumps(karar.model_dump(), ensure_ascii=False, indent=2))

        if mudahaleler:
            print("\n--- Kural motoru mudahaleleri ---")
            for m in mudahaleler:
                print(" *", m)

        print("\n--- Ozet ---")
        print(f"Karar    : {karar.karar}")
        print(f"Miktar   : {karar.miktar}")
        print(f"Aciliyet : {karar.aciliyet}")
        print(f"Gerekce  : {karar.gerekce}")
        print(f"Sure     : {sure} sn | deneme: {deneme}")

    cikti = {
        "zaman": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_ADI,
        "prompt_surum": PROMPT_SURUM,
        "girdi": girdi,
        "hesaplanan": hesap,
        "karar": karar.model_dump(),
        "mudahaleler": mudahaleler,
        "deneme_sayisi": deneme,
        "sure_sn": sure,
    }

    KARAR_JSON.write_text(
        json.dumps(cikti, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logla({**cikti, "llm_ham_cikti": ham_cikti})

    if yazdir:
        print(f"\nKarar kaydedildi : {KARAR_JSON}")
        print(f"Log eklendi      : {LOG_DOSYA}")

    return karar


def tutarlilik_testi(client: OpenAI, girdi: dict, n: int = 3) -> None:
    """Aynı girdiyle n kez çalıştır — sunumda gösterilecek kanıt."""
    print(f"\n########## TUTARLILIK TESTI ({n} calistirma) ##########")
    sonuclar = []
    for i in range(1, n + 1):
        k = calistir(client, girdi, yazdir=False)
        sonuclar.append(k)
        print(f"{i}. calistirma -> karar={k.karar:12} miktar={k.miktar:6} aciliyet={k.aciliyet}")

    miktarlar = {s.miktar for s in sonuclar}
    kararlar = {s.karar for s in sonuclar}
    print("\nSONUC:", "TUTARLI (miktar ve karar sabit)"
          if len(miktarlar) == 1 and len(kararlar) == 1
          else f"TUTARSIZ -> miktarlar={miktarlar}, kararlar={kararlar}")


if __name__ == "__main__":
    # ---- Örnek girdi (Faz 5-adim 2'de burasi Prophet ciktisiyla degisecek) ----
    ORNEK_GIRDI = {
        "magaza": 1,
        "tarih": "2015-07-31",
        "tahmin_satis": 5800.0,
        "guven_alt": 4200.0,
        "guven_ust": 7400.0,
        "mevcut_stok": 4200.0,
        "promo_var_mi": True,
    }

    client = istemci_olustur()

    if "--test" in sys.argv:
        tutarlilik_testi(client, ORNEK_GIRDI)
    else:
        calistir(client, ORNEK_GIRDI)