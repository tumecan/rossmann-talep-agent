"""
routing_eval.py — agent_v7 yonlendirme degerlendirmesi

NE OLCULUYOR
    "Agent dogru ARACI secti mi?" Bu, "verdigi KARAR dogru muydu?"
    sorusundan farklidir; onu gercek_backtest.py olcer.

KARAR_URET VE YASAKLI ARAC KONTROLU
    karar_uret diger araclarla kavramsal olarak ortusuyor:
    "en cok ihtiyaci olan magazalar" bir BILGI sorusu, "264'e siparis
    gecelim mi" bir KARAR sorusudur. Ikisi de stok ve siparis hakkinda.
    Bu yuzden eval setinde iki sey var:

      1. karar_uret icin 6 pozitif soru
      2. YASAKLI ARAC kontrolu: bazi sorularda dogru araci cagirmak
         yetmez, YANLIS araci cagirmamak da gerekir. Bilgi sorusunda
         karar_uret cagrilirsa gereksiz karar uretilir ve karar defteri
         kirlenir — bu sessiz bir hatadir, ciktiya bakarak anlasilmaz.

CALISTIRMA
    python routing_eval.py
    python routing_eval.py --hizli        # bekleme suresini 3 sn'ye dusurur
    python routing_eval.py --sadece karar_uret
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agent_v7 import agent_olustur, sor

# (soru, beklenen_tool, yasakli_tool)
# yasakli_tool None ise yalnizca beklenen aracin cagrilmasi yeterlidir.
EVAL_SETI = [
    # === TUKENME_RISKI (5) ===
    ("Hangi mağazalarda stok tükeniyor?", "tukenme_riski", "karar_uret"),
    ("Önümüzdeki iki haftada risk altındaki mağazalar hangileri?", "tukenme_riski", "karar_uret"),
    ("En erken hangi mağaza tükenir?", "tukenme_riski", None),
    ("10-20 Temmuz arasında kaç mağaza stoksuz kalır?", "tukenme_riski", None),
    ("Stok tükenen mağazalara promosyon yapılmalı mı?", "tukenme_riski", None),

    # === SEVKIYAT_PLANI_GETIR (5) ===
    ("7 Temmuz sevkiyat planı ne?", "sevkiyat_plani_getir", None),
    ("Bugün kimlere mal gönderilecek?", "sevkiyat_plani_getir", None),
    ("Yarın teslim edilecek siparişler?", "sevkiyat_plani_getir", None),
    ("15 Temmuz'da hangi tedarikçiden sevkiyat var?", "sevkiyat_plani_getir", None),
    ("Bu hafta toplam kaç sipariş gönderilecek?", "sevkiyat_plani_getir", None),

    # === EN_COK_IHTIYAC_DUYAN (4) ===
    # Bu grup karar_uret ile en cok karisan gruptur: ikisi de "ihtiyac"
    # ve "siparis" kavramlari etrafinda doner. Hepsinde yasakli isaretli.
    ("En çok ihtiyacı olan mağazalar hangileri?", "en_cok_ihtiyac_duyan", "karar_uret"),
    ("Önceliği kime verelim?", "en_cok_ihtiyac_duyan", "karar_uret"),
    ("Acil stok gereken mağazalar?", "en_cok_ihtiyac_duyan", "karar_uret"),
    ("Hangi mağaza en çok satış kaçırıyor?", "en_cok_ihtiyac_duyan", "karar_uret"),

    # === SEGMENT_OZETI (5) ===
    ("B tipi mağazalar nasıl?", "segment_ozeti", None),
    ("Mağaza tipleri arasında fark var mı?", "segment_ozeti", None),
    ("Hangi segment daha riskli?", "segment_ozeti", None),
    ("A tipi ile D tipi mağazaları karşılaştır", "segment_ozeti", None),
    ("Ürün yelpazesi c olan mağazaların durumu ne?", "segment_ozeti", None),

    # === MODEL_PERFORMANSI (4) ===
    ("Mağaza 530'un tahmini ne kadar güvenilir?", "model_performansi", "karar_uret"),
    ("Bu sayılara güvenebilir miyim, mağaza 100?", "model_performansi", None),
    ("Mağaza 850'nin hata oranı kaç?", "model_performansi", None),
    ("Mağaza 730'un tahmini güvenilir mi?", "model_performansi", None),

    # === RAF_OMRU_DURUMU (5) ===
    ("Kimde raf ömrü doluyor?", "raf_omru_durumu", "karar_uret"),
    ("Bayat mal kimde birikmiş?", "raf_omru_durumu", None),
    ("10 Temmuz'da fire riski olan mağazalar?", "raf_omru_durumu", "karar_uret"),
    ("4 Temmuz'da hangi mağazalarda ürün çöpe gidecek?", "raf_omru_durumu", None),
    ("4 Temmuz'da ne kadar fire var?", "raf_omru_durumu", None),

    # === DAGITIM_PLANI (4) ===
    ("Elimde 50 bin birim var nasıl dağıtayım, 10 Temmuz?", "dagitim_plani", None),
    ("Kısıtlı stoğu kime verelim, 30 bin birim, 10 Temmuz?", "dagitim_plani", None),
    ("20 bin birim dağıtmam lazım, 15 Temmuz için", "dagitim_plani", None),
    ("100 bin birimlik dağıtım planı oluştur, 10 Temmuz", "dagitim_plani", None),

    # === MAGAZA_KARSILASTIR (4) ===
    ("Mağaza 530 ile 1'i karşılaştır", "magaza_karsilastir", None),
    ("Hangisi daha riskli, 250 mi 800 mü?", "magaza_karsilastir", None),
    ("Mağaza 400 mü 900 mü daha kötü?", "magaza_karsilastir", None),
    ("Mağaza 100 ile mağaza 200 arasındaki fark ne?", "magaza_karsilastir", None),

    # === PROMO_SENARYOSU (4) ===
    ("Mağaza 200'de promosyon yapsak talep ne olur?", "promo_senaryosu", None),
    ("Kampanya talebi ne kadar artırır, mağaza 50?", "promo_senaryosu", None),
    ("Promosyonun etkisi ne kadar, mağaza 730?", "promo_senaryosu", None),
    ("Mağaza 1'de kampanya yaparsak ne olur?", "promo_senaryosu", None),

    # === TEDARIKCI_BILGISI_ARA / RAG (5) ===
    ("Schnellware'ın minimum sipariş miktarı ne?", "tedarikci_bilgisi_ara", None),
    ("İndirim oranı politikası ne diyor?", "tedarikci_bilgisi_ara", "karar_uret"),
    ("Nordmann hakkında ne biliyorsun?", "tedarikci_bilgisi_ara", None),
    ("Onay kuralları nedir?", "tedarikci_bilgisi_ara", "karar_uret"),
    ("ExpressLog ile Rheinland'ı karşılaştırabilir misin?", "tedarikci_bilgisi_ara", None),

    # === TREND_OZETI (4) ===
    # Zaman ekseni + zincir geneli. model_performansi ile karisma riski
    # oldugu icin iki soru bilincli olarak "model" kelimesi icerir.
    ("Model zamanla kötüleşiyor mu?", "trend_ozeti", "karar_uret"),
    ("Haftalık trend nasıl gidiyor?", "trend_ozeti", "karar_uret"),
    ("Tahmin performansı haftadan haftaya nasıl değişti?", "trend_ozeti", None),
    ("Modelde drift var mı?", "trend_ozeti", "karar_uret"),

    # === KARAR_URET (6) ===
    # Ortak nokta: TEK magaza + TEK tarih + aksiyon talebi.
    ("264 numaralı mağaza için 20 Temmuz'da karar ver", "karar_uret", None),
    ("Mağaza 1100'e 6 Temmuz'da sipariş geçelim mi?", "karar_uret", None),
    ("Mağaza 418 için 6 Temmuz'da ne yapmalıyım?", "karar_uret", None),
    ("500 numaralı mağazaya 15 Temmuz için aksiyon önerisi ver", "karar_uret", None),
    ("Mağaza 23 için 5 Temmuz'da indirim yapmalı mıyız?", "karar_uret", None),
    ("617 numaralı mağazanın 6 Temmuz kararını üret", "karar_uret", None),

    # === KAPSAM DISI — tool cagirmamali (4) ===
    ("Hava nasıl?", "__reddedilmeli__", None),
    ("Bana bir şiir yaz", "__reddedilmeli__", None),
    ("Python'da liste nasıl sıralanır?", "__reddedilmeli__", None),
    ("Türkiye'nin başkenti neresi?", "__reddedilmeli__", None),
]

BEKLEME = 8          # rate limit onlemi; --hizli ile 3 sn
KOK = Path(__file__).resolve().parent
LOG = KOK / "Rossman" / "routing_eval_sonuc.json"
if not LOG.parent.exists():
    LOG = KOK / "routing_eval_sonuc.json"


def _normalize(kayit):
    """Eski 2'li tuple'lari da kabul et."""
    if len(kayit) == 2:
        return kayit[0], kayit[1], None
    return kayit


def calistir(bekleme: int, filtre: str | None) -> None:
    agent = agent_olustur()

    seti = [_normalize(k) for k in EVAL_SETI]
    if filtre:
        seti = [k for k in seti if k[1] == filtre]
        if not seti:
            sys.exit(f"HATA: '{filtre}' icin soru yok.")

    sonuclar = []
    dogru = 0
    toplam = len(seti)
    # Hangi arac yerine hangisi cagrildi — hata desenini gorebilmek icin
    karisma = defaultdict(int)

    for i, (soru, beklenen, yasakli) in enumerate(seti, 1):
        print(f"\n[{i:2d}/{toplam}] {soru}")
        yasakli_ihlali = False
        try:
            cevap, _, adimlar = sor(agent, soru, [], trace=False)
            gelen = [a["tool"] for a in adimlar]

            if beklenen == "__reddedilmeli__":
                eslesme = len(gelen) == 0
            else:
                eslesme = beklenen in gelen

            # Yasakli arac cagrildiysa soru basarisiz sayilir; dogru araci
            # da cagirmis olmasi kurtarmaz, cunku yan etki olusmustur
            # (gereksiz karar uretimi karar defterine satir yazar).
            if yasakli and yasakli in gelen:
                yasakli_ihlali = True
                eslesme = False

            if not eslesme and beklenen != "__reddedilmeli__":
                for g in gelen:
                    if g != beklenen:
                        karisma[f"{beklenen} -> {g}"] += 1
            if beklenen == "__reddedilmeli__" and gelen:
                for g in gelen:
                    karisma[f"reddedilmeli -> {g}"] += 1

        except Exception as e:
            gelen = [f"HATA: {type(e).__name__}: {e}"]
            eslesme = False

        if eslesme:
            dogru += 1
        durum = "OK " if eslesme else "HATA"
        ek = "  [YASAKLI ARAC CAGRILDI]" if yasakli_ihlali else ""
        print(f"  {durum}  beklenen: {beklenen} | gelen: {gelen}{ek}")

        sonuclar.append({
            "soru": soru,
            "beklenen": beklenen,
            "yasakli": yasakli,
            "gelen": gelen,
            "eslesme": eslesme,
            "yasakli_ihlali": yasakli_ihlali,
        })

        if i < toplam:
            time.sleep(bekleme)

    # --- arac bazli dokum -------------------------------------------------
    arac = defaultdict(lambda: [0, 0])
    for s in sonuclar:
        arac[s["beklenen"]][1] += 1
        if s["eslesme"]:
            arac[s["beklenen"]][0] += 1

    oran = dogru / toplam * 100
    print("\n" + "=" * 62)
    print(f"SONUC: {dogru}/{toplam} dogru — %{oran:.1f}")
    print("=" * 62)
    print("\n--- Arac bazli ---")
    for ad, (d, t) in sorted(arac.items(), key=lambda x: x[1][0] / x[1][1]):
        print(f"  {ad:24s} {d}/{t}  ({d/t:.0%})")

    ihlal = sum(1 for s in sonuclar if s["yasakli_ihlali"])
    print(f"\nYasakli arac ihlali: {ihlal}")
    if ihlal:
        print("  UYARI: bilgi sorusunda karar uretilmis. Bu sessiz bir")
        print("  hatadir — cevap dogru gorunur ama karar defterine")
        print("  gereksiz satir yazilir. Prompt'taki ARAC SECIMI bolumu")
        print("  netlestirilmelidir.")

    if karisma:
        print("\n--- Karisma deseni (beklenen -> cagrilan) ---")
        for k, v in sorted(karisma.items(), key=lambda x: -x[1]):
            print(f"  {k:44s} {v}")

    rapor = {
        "tarih": datetime.now().isoformat(timespec="seconds"),
        "toplam": toplam,
        "dogru": dogru,
        "oran": round(oran, 1),
        "yasakli_ihlali": ihlal,
        "arac_bazli": {k: {"dogru": v[0], "toplam": v[1]} for k, v in arac.items()},
        "karisma": dict(karisma),
        "detay": sonuclar,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSonuc kaydedildi: {LOG}")


if __name__ == "__main__":
    a = sys.argv
    calistir(
        bekleme=3 if "--hizli" in a else BEKLEME,
        filtre=a[a.index("--sadece") + 1] if "--sadece" in a else None,
    )