"""
n8n_gonder.py — Agent kararlarini n8n webhook'una gonderen kopru.

Kaynak: Rossman/agent_log.jsonl  (CSV'de olmayan guven araligi burada)
Cikti : n8n webhook + Rossman/n8n_gonderim_log.jsonl (izlenebilirlik)

Kullanim:
    python .\\n8n_gonder.py --liste                  # log'daki gunleri listeler
    python .\\n8n_gonder.py --kuru                   # hicbir yere gondermez, payload'i yazar
    python .\\n8n_gonder.py --tarih 2015-06-26       # tek gun gonderir
    python .\\n8n_gonder.py --toplu                  # tum gunleri sirayla gonderir
    python .\\n8n_gonder.py --toplu --gecikme 1.5    # gunler arasi bekleme (sn)
    python .\\n8n_gonder.py --tarih 2015-06-26 --url http://localhost:5678/webhook-test/stok-karar

URL onceligi: --url  >  .env icindeki n8n_webhook_url  >  VARSAYILAN_URL
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------------------------
# YOLLAR
# ----------------------------------------------------------------------------
KOK_DIZIN = Path(__file__).resolve().parent
CIKTI_DIR = KOK_DIZIN / "Rossman"
if not CIKTI_DIR.exists():
    CIKTI_DIR = KOK_DIZIN

LOG_JSONL = CIKTI_DIR / "agent_log.jsonl"
GONDERIM_LOG = CIKTI_DIR / "n8n_gonderim_log.jsonl"

# n8n'de "Test workflow" ile dinlerken /webhook-test/, aktif workflow'da /webhook/
VARSAYILAN_URL = "http://localhost:5678/webhook-test/stok-karar"
ZAMAN_ASIMI = 20


# ----------------------------------------------------------------------------
# LOG OKUMA
# ----------------------------------------------------------------------------
def kayitlari_oku() -> dict:
    """agent_log.jsonl'i okur, ayni tarihin EN SON kaydini tutar."""
    if not LOG_JSONL.exists():
        sys.exit(f"HATA: {LOG_JSONL} yok. Once 'python .\\agent_v6.py --toplu' calistir.")

    kayitlar = {}
    bozuk = 0
    for satir in LOG_JSONL.read_text(encoding="utf-8").splitlines():
        satir = satir.strip()
        if not satir:
            continue
        try:
            k = json.loads(satir)
        except json.JSONDecodeError:
            bozuk += 1
            continue
        tarih = (k.get("girdi") or {}).get("tarih")
        if tarih:
            kayitlar[tarih] = k          # sonraki kosu oncekini ezer
    if bozuk:
        print(f"UYARI: {bozuk} bozuk satir atlandi.")
    if not kayitlar:
        sys.exit("HATA: log'da gecerli kayit yok.")
    return dict(sorted(kayitlar.items()))


# ----------------------------------------------------------------------------
# PAYLOAD
# ----------------------------------------------------------------------------
def payload_yap(kayit: dict) -> dict:
    """Agent log kaydini n8n'in rahat okuyacagi DUZ bir sozluge cevirir."""
    g = kayit.get("girdi", {})
    kr = kayit.get("karar", {})
    iz = kayit.get("tool_trace", []) or []
    mud = kayit.get("mudahaleler", []) or []

    tahmin = float(g.get("tahmin_satis", 0) or 0)
    alt = float(g.get("guven_alt", 0) or 0)
    ust = float(g.get("guven_ust", 0) or 0)
    bant = round(ust - alt, 1)
    # Belirsizlik olcusu: guven bandinin tahmine orani. n8n'de routing esigi.
    belirsizlik = round(bant / tahmin, 3) if tahmin > 0 else 1.0

    return {
        "olay": "stok_karari",
        "tarih": g.get("tarih"),
        "magaza": g.get("magaza", 1),
        "kategori": "taze",

        # --- karar ---
        "karar": kr.get("karar"),
        "aciliyet": kr.get("aciliyet"),
        "miktar": int(kr.get("miktar", 0) or 0),
        "tedarikci": kr.get("tedarikci", "yok"),
        "teslimat_notu": kr.get("teslimat_notu", ""),
        "onay_gerekli_mi": bool(kr.get("onay_gerekli_mi", False)),
        "indirim_orani": float(kr.get("indirim_orani", 0) or 0),
        "secilen_senaryo": kr.get("secilen_senaryo", "aksiyon yok"),
        "net_etki_eur": float(kr.get("tahmini_net_etki_eur", 0) or 0),
        "fire_riski_birim": float(kr.get("fire_riski_birim", 0) or 0),
        "gerekce": kr.get("gerekce", ""),

        # --- tahmin ve belirsizlik (routing icin) ---
        "tahmin_satis": tahmin,
        "guven_alt": alt,
        "guven_ust": ust,
        "guven_bandi": bant,
        "belirsizlik_orani": belirsizlik,

        # --- durum ---
        "mevcut_stok": float(g.get("mevcut_stok", 0) or 0),
        "kalan_raf_omru": int(g.get("kalan_raf_omru", 0) or 0),
        "promo": int(bool(g.get("promo_var_mi", False))),

        # --- izlenebilirlik ---
        "tool_sayisi": len(iz),
        "tool_zinciri": " -> ".join(a.get("tool", "?") for a in iz),
        "kural_mudahalesi": bool(mud),
        "mudahaleler": mud,
        "model": kayit.get("model"),
        "prompt_surum": kayit.get("prompt_surum"),
        "karar_zamani": kayit.get("zaman"),
    }


# ----------------------------------------------------------------------------
# GONDERIM
# ----------------------------------------------------------------------------
def gonder(url: str, payload: dict, kuru: bool) -> bool:
    etiket = f"{payload['tarih']} | {payload['karar']:15}"
    if kuru:
        print(f"\n--- {etiket} (KURU KOSU, gonderilmedi) ---")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return True

    try:
        y = requests.post(url, json=payload, timeout=ZAMAN_ASIMI)
        basarili = 200 <= y.status_code < 300
        print(f"{etiket} -> HTTP {y.status_code} {'OK' if basarili else 'HATA'}"
              f"  {y.text[:120].strip()}")
    except requests.exceptions.RequestException as e:
        basarili = False
        print(f"{etiket} -> BAGLANTI HATASI: {type(e).__name__} ({e})")
        print("   ipucu: n8n calisiyor mu? Test webhook'u dinlemede mi?")

    with GONDERIM_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "gonderim_zamani": datetime.now().isoformat(timespec="seconds"),
            "url": url, "basarili": basarili, "payload": payload,
        }, ensure_ascii=False) + "\n")
    return basarili


# ----------------------------------------------------------------------------
def main() -> None:
    load_dotenv()
    a = sys.argv

    url = (a[a.index("--url") + 1] if "--url" in a
           else os.getenv("n8n_webhook_url") or VARSAYILAN_URL).strip().strip('"')
    gecikme = float(a[a.index("--gecikme") + 1]) if "--gecikme" in a else 1.0
    kuru = "--kuru" in a

    kayitlar = kayitlari_oku()

    if "--liste" in a:
        print(f"\nLog'da {len(kayitlar)} gun var:")
        for t, k in kayitlar.items():
            print(f"  {t}  {k['karar']['karar']:15} "
                  f"onay={int(k['karar'].get('onay_gerekli_mi', False))}")
        return

    if "--toplu" in a:
        hedef = list(kayitlar.values())
    else:
        tarih = a[a.index("--tarih") + 1] if "--tarih" in a else list(kayitlar)[-1]
        if tarih not in kayitlar:
            sys.exit(f"HATA: {tarih} log'da yok. '--liste' ile bak.")
        hedef = [kayitlar[tarih]]

    print(f"\nHedef URL : {url}{'  (KURU KOSU)' if kuru else ''}")
    print(f"Gonderilecek gun sayisi: {len(hedef)}\n")

    basari = 0
    for i, k in enumerate(hedef):
        if gonder(url, payload_yap(k), kuru):
            basari += 1
        if not kuru and i < len(hedef) - 1:
            time.sleep(gecikme)

    print(f"\n--- Ozet: {basari}/{len(hedef)} basarili ---")
    if not kuru:
        print(f"Gonderim log: {GONDERIM_LOG}")


if __name__ == "__main__":
    main()