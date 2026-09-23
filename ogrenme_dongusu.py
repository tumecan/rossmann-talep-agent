"""
ogrenme_dongusu.py — Onay kayitlarindan esik kalibrasyonu

DONGU
    karar uret -> insan onayi/reddi -> kayit -> ANALIZ -> esik onerisi
                                                  ^
                                            bu script

NE YAPAR, NE YAPMAZ
    YAPAR: reddedilen kararlarin hangi PARAMETRELERLE uretildigine bakar,
           red sebeplerini gruplar ve somut esik onerisi uretir.
    YAPMAZ: esikleri kendiliginden DEGISTIRMEZ. Oneri uretir, karari
           insan verir. Otomatik esik degisimi, birkac reddin sistemi
           savurmasina yol acardi; uretimde bu bir guvenlik zafiyetidir.

NEDEN SEBEP KODLARI SABIT
    Red sebebi serbest metin olsaydi gruplanamazdi. Dort sabit kod her
    biri FARKLI bir esige isaret eder:
      model      -> belirsizlik kapisi cok GEVSEK (zayif modelli karar
                    otomatik uretilmis, insan guvenmemis)
      parametre  -> miktar/oran formulu yanlis (indirim orani secimi,
                    siparis miktari hesabi)
      zamanlama  -> karar dogru ama gun yanlis (raf omru/lead-time)
      diger      -> siniflandirilamiyor; cok artarsa sebep kumesi eksik

ISTATISTIKSEL DURUSTLUK
    Az kayitla esik onerisi uretmek uydurmadir. Script her oneri icin
    minimum gozlem sayisi arar ve altinda kalirsa "yeterli veri yok" der.

Calistirma:
    python ogrenme_dongusu.py
    python ogrenme_dongusu.py --min 5      # minimum gozlem esigini degistir
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

KOK = Path(__file__).resolve().parent
ROSSMAN = KOK / "Rossman"
LOG = (ROSSMAN if ROSSMAN.exists() else KOK) / "onay_kayitlari.jsonl"
RAPOR = (ROSSMAN if ROSSMAN.exists() else KOK) / "ogrenme_raporu.json"

# Bir oneri uretmek icin gereken minimum gozlem. Altinda "yeterli veri yok".
VARSAYILAN_MIN = 5


def kayitlari_oku() -> list[dict]:
    if not LOG.exists():
        sys.exit(f"Henuz onay kaydi yok: {LOG}\n"
                 "Telegram'dan bir karar onaylayin veya reddedin.")
    k = []
    for satir in LOG.read_text(encoding="utf-8").splitlines():
        if satir.strip():
            try:
                k.append(json.loads(satir))
            except json.JSONDecodeError:
                continue
    return k


def son_durumlar(kayitlar: list[dict]) -> list[dict]:
    """Ayni karar birden cok kez oylanmis olabilir (fikir degistirme).

    Esik analizinde her karar BIR kez sayilmali, yoksa cok oylanan bir
    karar sonucu orantisiz etkiler. Son oy gecerli sayilir.
    """
    son: dict[str, dict] = {}
    for k in kayitlar:
        son[k["karar_id"]] = k
    return list(son.values())


def rapor(kayitlar: list[dict], min_gozlem: int) -> dict:
    hepsi = son_durumlar(kayitlar)
    n = len(hepsi)
    onay = [k for k in hepsi if k["durum"] == "onaylandi"]
    red = [k for k in hepsi if k["durum"] == "reddedildi"]

    print("=" * 72)
    print("OGRENME DONGUSU RAPORU")
    print("=" * 72)
    print(f"Toplam olay kaydi     : {len(kayitlar)}")
    print(f"Benzersiz karar       : {n}   (son oy gecerli)")
    print(f"  Onaylandi           : {len(onay)}")
    print(f"  Reddedildi          : {len(red)}")
    if n:
        print(f"  Onay orani          : {len(onay)/n:.1%}")

    if n < min_gozlem:
        print(f"\nUYARI: {n} karar, minimum {min_gozlem} gozlem esiginin")
        print("altinda. Asagidaki dokum bilgi amaclidir; ESIK ONERISI")
        print("uretilmemistir. Az veriyle esik degistirmek, gurultuyu")
        print("sinyal saymaktir.")

    # --- karar turune gore -----------------------------------------------
    print("\n--- Karar turune gore onay orani ---")
    tur = defaultdict(lambda: {"onay": 0, "red": 0})
    for k in hepsi:
        if k.get("karar"):
            tur[k["karar"]]["onay" if k["durum"] == "onaylandi" else "red"] += 1
    if tur:
        for ad, s in sorted(tur.items()):
            t = s["onay"] + s["red"]
            print(f"  {ad:16s} {s['onay']}/{t} onay ({s['onay']/t:.0%})"
                  f"{'   [az veri]' if t < min_gozlem else ''}")
    else:
        print("  (karar turu bilgisi olan kayit yok)")

    # --- red sebepleri ----------------------------------------------------
    print("\n--- Red sebepleri ---")
    sebepler = Counter(k.get("sebep_kodu") or "belirtilmemis" for k in red)
    if sebepler:
        for kod, adet in sebepler.most_common():
            print(f"  {kod:16s} {adet}")
    else:
        print("  (red yok)")

    # --- ONERILER ---------------------------------------------------------
    print("\n--- ESIK ONERILERI ---")
    oneriler = []

    def oneri(baslik: str, gozlem: int, metin: str, alan: str | None = None,
              mevcut=None, onerilen=None):
        yeterli = gozlem >= min_gozlem
        oneriler.append({"baslik": baslik, "gozlem": gozlem,
                         "yeterli_veri": yeterli, "aciklama": metin,
                         "alan": alan, "mevcut": mevcut, "onerilen": onerilen})
        isaret = "->" if yeterli else "  "
        print(f"\n{isaret} {baslik}  (gozlem: {gozlem})")
        print(f"   {metin}")
        if not yeterli:
            print(f"   YETERLI VERI YOK (min {min_gozlem}); oneri uygulanmamali.")
        elif alan:
            print(f"   {alan}: {mevcut} -> {onerilen}")

    # 1) model sebebiyle red -> belirsizlik kapisi gevsek
    model_red = [k for k in red if k.get("sebep_kodu") == "model"]
    if model_red:
        # Kapi val sMAPE ile calisir; esik onerisi de ayni metrikten turer.
        # Eski kayitlarda (kapi duzeltmesi oncesi) yalniz test alani vardir.
        smapeler = [k.get("model_smape_val", k.get("model_smape_test"))
                    for k in model_red]
        smapeler = [x for x in smapeler if x is not None]
        mevcut = model_red[0].get("esikler", {}).get("smape_belirsiz_esik")
        if smapeler and mevcut:
            # Reddedilenlerin EN DUSUK sMAPE'i, esigin inmesi gereken yeri
            # gosterir: o karar otomatik uretilmis ama insan guvenmemis.
            yeni = round(min(smapeler) - 1, 1)
            oneri("Belirsizlik kapisi cok gevsek", len(model_red),
                  f"{len(model_red)} karar 'tahmin gercekci degil' ile "
                  f"reddedildi. Bu kararlarin sMAPE araligi "
                  f"{min(smapeler):.1f}-{max(smapeler):.1f}. Esik bu araligin "
                  f"altina cekilirse benzer kararlar otomatik uretilmez, "
                  f"insana devredilir.",
                  "SMAPE_BELIRSIZ_ESIK", mevcut, yeni)

    # 2) parametre sebebiyle red -> miktar/oran formulu
    par_red = [k for k in red if k.get("sebep_kodu") == "parametre"]
    if par_red:
        indirimli = [k for k in par_red if k.get("karar") == "indirim_uygula"]
        siparisli = [k for k in par_red if k.get("karar") == "siparis_ver"]
        metin = (f"{len(par_red)} karar 'miktar/oran yanlis' ile reddedildi "
                 f"({len(indirimli)} indirim, {len(siparisli)} siparis). ")
        if indirimli:
            oranlar = [k.get("indirim_orani") for k in indirimli
                       if k.get("indirim_orani")]
            if oranlar:
                metin += (f"Reddedilen indirim oranlari: "
                          f"{sorted(set(oranlar))}. Elastikiyet varsayimi "
                          f"(1.5) gozden gecirilmeli — net etki hesabi buna "
                          f"dayaniyor ve gercek fiyat testi verisi yok.")
        if siparisli:
            metin += (" Siparis miktari formulu (p50 x emniyet esigi - stok) "
                      "gozden gecirilmeli.")
        oneri("Miktar/oran formulu sorgulaniyor", len(par_red), metin)

    # 3) zamanlama sebebiyle red -> raf omru / lead-time
    zam_red = [k for k in red if k.get("sebep_kodu") == "zamanlama"]
    if zam_red:
        oneri("Zamanlama mantigi eksik", len(zam_red),
              f"{len(zam_red)} karar 'zamanlama uygun degil' ile reddedildi. "
              f"Kural motoru yalnizca O GUNE bakiyor; tedarikci lead-time'i "
              f"ve kalan raf omru ile birlikte degerlendirmiyor. Kararin "
              f"bir gun once/sonra alinmasi gerekiyorsa bu kurala girmeli.")

    # 4) onay yuku
    onay_gerekenler = [k for k in hepsi if k.get("onay_sebebi")]
    if onay_gerekenler:
        oran = len(onay_gerekenler) / n if n else 0
        if oran > 0.6:
            oneri("Onay yuku yuksek", len(onay_gerekenler),
                  f"Kararlarin %{oran*100:.0f}'i onaya gidiyor. Bu oran "
                  f"yuksek kalirsa sistem 'karar veren' degil 'oneri "
                  f"hazirlayan' konuma duser. Onay esigi (EUR) yukseltilebilir "
                  f"veya belirsizlik kapisi daraltilabilir; ikisi de kacirilan "
                  f"riski artirir — karar_degerlendirme.py --duyarlilik ile "
                  f"takas olculmelidir.")

    # 5) 'diger' cok ise sebep kumesi eksik
    diger = sebepler.get("diger", 0)
    if diger and red and diger / len(red) > 0.3:
        oneri("Sebep kumesi yetersiz", diger,
              f"Redlerin %{diger/len(red)*100:.0f}'i 'baska sebep' ile "
              f"isaretlendi. Mevcut dort sebep kodu gercek red gerekcelerini "
              f"karsilamiyor; kume genisletilmeli.")

    if not oneriler:
        print("\n  Oneri uretilecek desen bulunamadi.")

    print("\n" + "=" * 72)
    print("NOT: Bu script esikleri DEGISTIRMEZ. Oneriler karar_tool.py")
    print("icindeki sabitlere elle uygulanir ve sonrasinda")
    print("karar_degerlendirme.py ile etkisi olculur.")

    cikti = {
        "toplam_olay": len(kayitlar),
        "benzersiz_karar": n,
        "onaylandi": len(onay),
        "reddedildi": len(red),
        "onay_orani": round(len(onay) / n, 3) if n else None,
        "karar_turune_gore": {a: dict(s) for a, s in tur.items()},
        "red_sebepleri": dict(sebepler),
        "min_gozlem": min_gozlem,
        "oneriler": oneriler,
    }
    RAPOR.write_text(json.dumps(cikti, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"Rapor kaydedildi: {RAPOR}")
    return cikti


def main() -> None:
    a = sys.argv
    mn = int(a[a.index("--min") + 1]) if "--min" in a else VARSAYILAN_MIN
    rapor(kayitlari_oku(), mn)


if __name__ == "__main__":
    main()