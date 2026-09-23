"""
test_agent.py — Karar katmani degerlendirme seti.

NEYI TEST EDIYOR
LLM'in kendisini degil, LLM hata yaptiginda devreye giren KURAL MOTORUNU.
Sebebi su: LLM ciktisi olasiliksaldir, ayni girdiye farkli gunlerde farkli
cevap verebilir (bu projede iki kez yasandi). Guvence veremeyecegin bir
katmani test etmek yerine, ondan gelen HER TURLU ciktiyi dogru karara
cevirmesi gereken deterministik katmani test ediyoruz.

Her senaryo sunu yapar:
  1) Agent'in donebilecegi (bazen hatali) bir karar nesnesi kurar
  2) kurallari_dayat'i cagirir
  3) Sonucun is kurallarina uydugunu dogrular

LLM cagrilmadigi icin ucretsiz, saniyeler surer ve TEKRARLANABILIR.
Canli agent testi icin: python test_agent.py --canli
(bu mod gercek API cagrisi yapar, ucretlidir, birkac dakika surer)

Calistirma:
    python test_agent.py
    python test_agent.py --canli
"""

import sys

import agent_v6 as av
from agent_v6 import AgentKararV6, gercek_fire_riski, kurallari_dayat

ONAY_ESIGI = av.ONAY_ESIGI


# ============================================================================
# YARDIMCILAR
# ============================================================================
def gun_bul(fire_riski_olsun: bool) -> str:
    """Veri setinden, fire riski olan / olmayan gercek bir tarih secer.
    Tarihleri koda gommuyoruz: veri degisirse test kendini uyarlar."""
    for _, s in av.DURUM.iterrows():
        tarih = s["ds"].strftime("%Y-%m-%d")
        varmi = gercek_fire_riski(tarih) > 0
        if varmi == fire_riski_olsun:
            return tarih
    raise RuntimeError(
        f"Veri setinde fire riski {'olan' if fire_riski_olsun else 'olmayan'} gun yok.")


def girdi_yap(tarih: str) -> dict:
    return av.girdi_hazirla(tarih)


def hesap_yap(ihtiyac: float, onerilen: int, belirsiz: bool = False) -> dict:
    """Deterministik hesap katmaninin ciktisini taklit eder."""
    return {"ihtiyac": ihtiyac, "onerilen_miktar": onerilen, "belirsiz_mi": belirsiz}


def karar_yap(**alanlar) -> AgentKararV6:
    """Agent'in donebilecegi bir karar nesnesi kurar."""
    varsayilan = {
        "karar": "bekle",
        "gerekce": "Test senaryosu icin uretilmis gerekce metni.",
        "miktar": 0,
        "tedarikci": "yok",
        "onay_gerekli_mi": False,
        "fire_riski_birim": 0.0,
        "secilen_senaryo": "aksiyon yok",
        "indirim_orani": 0.0,
        "tahmini_net_etki_eur": 0.0,
        "aciliyet": "orta",
    }
    varsayilan.update(alanlar)
    return AgentKararV6(**varsayilan)


SONUCLAR = []


def dogrula(ad: str, beklenen: str, kosul: bool, detay: str = "") -> None:
    SONUCLAR.append({"ad": ad, "beklenen": beklenen, "gecti": kosul, "detay": detay})
    isaret = "GECTI" if kosul else "KALDI"
    print(f"  [{isaret}] {ad}")
    if not kosul and detay:
        print(f"          -> {detay}")


# ============================================================================
# SENARYOLAR
# ============================================================================
def senaryolari_calistir() -> None:
    riskli = gun_bul(True)
    temiz = gun_bul(False)
    riskli_deger = gercek_fire_riski(riskli)

    print(f"\nSecilen test gunleri: fire riskli={riskli} ({riskli_deger:.0f} birim), "
          f"temiz={temiz}\n")

    # ---- 1. Fazla stok -> indirim ----------------------------------------
    print("1) FAZLA STOK / ISRAF RISKI")
    av.INDIRIM_GECMISI.clear()

    k, m = kurallari_dayat(
        karar_yap(karar="bekle", fire_riski_birim=riskli_deger),
        hesap_yap(-50, 0), girdi_yap(riskli))
    dogrula("Fire riski varken 'bekle' -> indirim_uygula'ya cevrilir",
            "indirim_uygula", k.karar == "indirim_uygula", f"gelen: {k.karar}")

    av.INDIRIM_GECMISI.clear()
    k, m = kurallari_dayat(
        karar_yap(karar="bekle", fire_riski_birim=0.0),   # agent riski atladi
        hesap_yap(-50, 0), girdi_yap(riskli))
    dogrula("Agent fire riskini 0 yazsa bile kural motoru gercegi hesaplar",
            "indirim_uygula", k.karar == "indirim_uygula" and k.fire_riski_birim > 0,
            f"karar={k.karar}, fire={k.fire_riski_birim}")

    av.INDIRIM_GECMISI.clear()
    k, m = kurallari_dayat(
        karar_yap(karar="indirim_uygula", fire_riski_birim=riskli_deger,
                  indirim_orani=0.3, miktar=500),
        hesap_yap(-50, 0), girdi_yap(riskli))
    dogrula("Indirim kararinda siparis miktari sifirlanir",
            "miktar=0", k.miktar == 0, f"miktar={k.miktar}")

    # ---- 2. Politika kisiti ----------------------------------------------
    print("\n2) POLITIKA KISITI (pes pese iki gun %50 yasagi)")
    onceki = av.onceki_is_gunu(riskli)
    if onceki:
        av.INDIRIM_GECMISI.clear()
        av.INDIRIM_GECMISI[onceki] = 0.5      # dun %50 uygulandi
        k, m = kurallari_dayat(
            karar_yap(karar="indirim_uygula", fire_riski_birim=riskli_deger,
                      indirim_orani=0.5),
            hesap_yap(-50, 0), girdi_yap(riskli))
        dogrula("Onceki gun %50 ise bugun %50 uygulanamaz",
                "indirim_orani=0.3", abs(k.indirim_orani - 0.3) < 0.01,
                f"oran={k.indirim_orani}")

        av.INDIRIM_GECMISI.clear()
        av.INDIRIM_GECMISI[onceki] = 0.3      # dun %30 uygulandi
        k, m = kurallari_dayat(
            karar_yap(karar="indirim_uygula", fire_riski_birim=riskli_deger,
                      indirim_orani=0.5),
            hesap_yap(-50, 0), girdi_yap(riskli))
        dogrula("Onceki gun %30 ise bugun %50 serbesttir",
                "indirim_orani=0.5", abs(k.indirim_orani - 0.5) < 0.01,
                f"oran={k.indirim_orani}")

    # ---- 3. Siparis mantigi ----------------------------------------------
    print("\n3) SIPARIS MANTIGI")
    av.INDIRIM_GECMISI.clear()

    k, m = kurallari_dayat(
        karar_yap(karar="siparis_ver", miktar=500),
        hesap_yap(-120, 0), girdi_yap(temiz))
    dogrula("Ihtiyac yokken 'siparis_ver' -> bekle",
            "bekle", k.karar == "bekle" and k.miktar == 0,
            f"karar={k.karar}, miktar={k.miktar}")

    k, m = kurallari_dayat(
        karar_yap(karar="siparis_ver", miktar=9999),
        hesap_yap(150, 150), girdi_yap(temiz))
    dogrula("Agent'in uydurdugu miktar deterministik hesaba cekilir",
            "miktar=150", k.miktar == 150, f"miktar={k.miktar}")

    k, m = kurallari_dayat(
        karar_yap(karar="siparis_ver", miktar=100),
        hesap_yap(150, 150, belirsiz=True), girdi_yap(temiz))
    dogrula("Belirsizlik varken siparis -> insana_sor",
            "insana_sor", k.karar == "insana_sor", f"karar={k.karar}")

    # ---- 4. Onay esigi ----------------------------------------------------
    print("\n4) ONAY ESIGI VE ESKALASYON")
    av.INDIRIM_GECMISI.clear()

    buyuk = ONAY_ESIGI + 500
    k, m = kurallari_dayat(
        karar_yap(karar="siparis_ver", miktar=buyuk, onay_gerekli_mi=False),
        hesap_yap(buyuk, buyuk), girdi_yap(temiz))
    dogrula(f"{ONAY_ESIGI} birim ustu siparis onay ister",
            "onay_gerekli_mi=True", k.onay_gerekli_mi is True,
            f"onay={k.onay_gerekli_mi}, miktar={k.miktar}")

    k, m = kurallari_dayat(
        karar_yap(karar="siparis_ver", miktar=100, onay_gerekli_mi=True),
        hesap_yap(100, 100), girdi_yap(temiz))
    dogrula("Kucuk siparis gereksiz yere onaya gitmez",
            "onay_gerekli_mi=False", k.onay_gerekli_mi is False,
            f"onay={k.onay_gerekli_mi}")

    av.INDIRIM_GECMISI.clear()
    k, m = kurallari_dayat(
        karar_yap(karar="indirim_uygula", fire_riski_birim=riskli_deger,
                  indirim_orani=0.5, onay_gerekli_mi=False),
        hesap_yap(-50, 0), girdi_yap(riskli))
    dogrula("%50 indirim her zaman onay ister",
            "onay_gerekli_mi=True", k.onay_gerekli_mi is True,
            f"onay={k.onay_gerekli_mi}, oran={k.indirim_orani}")

    # ---- 5. Aciliyet tutarliligi -----------------------------------------
    print("\n5) ACILIYET TUTARLILIGI")
    av.INDIRIM_GECMISI.clear()

    k, m = kurallari_dayat(
        karar_yap(karar="bekle", fire_riski_birim=riskli_deger, aciliyet="dusuk"),
        hesap_yap(-50, 0), girdi_yap(riskli))
    dogrula("Fire riski varken aciliyet 'yuksek' olmali",
            "aciliyet=yuksek", k.aciliyet == "yuksek", f"aciliyet={k.aciliyet}")

    k, m = kurallari_dayat(
        karar_yap(karar="bekle", aciliyet="yuksek"),
        hesap_yap(-120, 0), girdi_yap(temiz))
    dogrula("Ihtiyac ve risk yokken aciliyet 'dusuk' olmali",
            "aciliyet=dusuk", k.aciliyet == "dusuk", f"aciliyet={k.aciliyet}")

    # ---- 6. Sema savunmasi ------------------------------------------------
    print("\n6) SEMA SAVUNMASI")

    k = karar_yap(indirim_orani=30)          # model "%30" niyetiyle 30 dondu
    dogrula("Yuzde/ondalik karisikligi otomatik duzelir",
            "indirim_orani=0.3", abs(k.indirim_orani - 0.3) < 0.01,
            f"oran={k.indirim_orani}")

    try:
        karar_yap(karar="stok_yak")          # sema disi karar
        dogrula("Sema disi karar degeri reddedilir", "ValidationError", False,
                "kabul edildi, reddedilmeliydi")
    except Exception:
        dogrula("Sema disi karar degeri reddedilir", "ValidationError", True)

    try:
        karar_yap(miktar=-100)               # negatif miktar
        dogrula("Negatif miktar reddedilir", "ValidationError", False,
                "kabul edildi, reddedilmeliydi")
    except Exception:
        dogrula("Negatif miktar reddedilir", "ValidationError", True)


# ============================================================================
# CANLI AGENT TESTI (opsiyonel, ucretli)
# ============================================================================
def canli_test() -> None:
    print("\n" + "=" * 72)
    print("CANLI AGENT TESTI (gercek API cagrisi)")
    print("=" * 72)
    agent = av.agent_olustur()

    riskli = gun_bul(True)
    temiz = gun_bul(False)

    for tarih, beklenen in ((riskli, "indirim_uygula"), (temiz, ("siparis_ver", "bekle"))):
        k = av.karar_ver(agent, tarih, yazdir=False)
        alinan = k["karar"]["karar"]
        uygun = alinan == beklenen if isinstance(beklenen, str) else alinan in beklenen
        dogrula(f"Canli: {tarih} -> {beklenen}", str(beklenen), uygun,
                f"alinan={alinan}, tool={len(k['tool_trace'])}, "
                f"mudahale={len(k['mudahaleler'])}")


# ============================================================================
def main() -> None:
    print("=" * 72)
    print("KARAR KATMANI DEGERLENDIRME SETI")
    print("=" * 72)

    senaryolari_calistir()
    if "--canli" in sys.argv:
        canli_test()

    gecen = sum(1 for s in SONUCLAR if s["gecti"])
    toplam = len(SONUCLAR)
    print("\n" + "=" * 72)
    print(f"SONUC: {gecen}/{toplam} senaryo gecti")
    if gecen < toplam:
        print("\nKALAN SENARYOLAR:")
        for s in SONUCLAR:
            if not s["gecti"]:
                print(f"  - {s['ad']}")
                print(f"    beklenen: {s['beklenen']} | {s['detay']}")
    print("=" * 72)
    sys.exit(0 if gecen == toplam else 1)


if __name__ == "__main__":
    main()