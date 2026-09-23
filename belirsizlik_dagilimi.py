"""Belirsizlik oraninin 36 gunluk dagilimi — n8n routing esigini
tahminle degil veriyle secmek icin."""

import json
from pathlib import Path

KOK = Path(__file__).resolve().parent
LOG = KOK / "Rossman" / "agent_log.jsonl"

kayitlar = {}
for satir in LOG.read_text(encoding="utf-8").splitlines():
    if not satir.strip():
        continue
    try:
        k = json.loads(satir)
    except json.JSONDecodeError:
        continue
    t = (k.get("girdi") or {}).get("tarih")
    if t:
        kayitlar[t] = k

satirlar = []
for t, k in sorted(kayitlar.items()):
    g, kr = k["girdi"], k["karar"]
    tahmin = float(g["tahmin_satis"])
    bant = float(g["guven_ust"]) - float(g["guven_alt"])
    satirlar.append((t, kr["karar"], tahmin, bant,
                     bant / tahmin if tahmin > 0 else 1.0,
                     bool(kr["onay_gerekli_mi"])))

print(f"{'tarih':12} {'karar':15} {'tahmin':>8} {'bant':>8} {'oran':>7}  onay")
for t, karar, tahmin, bant, oran, onay in satirlar:
    print(f"{t:12} {karar:15} {tahmin:8.1f} {bant:8.1f} {oran:7.3f}  {int(onay)}")

oranlar = sorted(s[4] for s in satirlar)
n = len(oranlar)
print(f"\nGun sayisi : {n}")
print(f"min / max  : {oranlar[0]:.3f} / {oranlar[-1]:.3f}")
print(f"medyan     : {oranlar[n // 2]:.3f}")
print(f"%75 dilim  : {oranlar[int(n * 0.75)]:.3f}")
print(f"%90 dilim  : {oranlar[int(n * 0.90)]:.3f}")

print("\nEsik denemeleri (kac gun belirsizlik nedeniyle onaya giderdi):")
for esik in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    kac = sum(1 for s in satirlar if s[4] > esik)
    print(f"  esik {esik:.2f} -> {kac:2} / {n} gun")