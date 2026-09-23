"""teshis_siparis.py — karar_tool siparis kurali ufka gore dejenere mi?

karar_tool siparis kararini mod='siparis_yok' + p90 stoksuzluguna bakarak
veriyor. Bu mod 28 gun boyunca HIC siparis vermiyor; karar_degerlendirme
referansi ayni sebeple (%80.5 dejenere) bu moddan vazgecmisti.
Hipotez: 3-4. gunden sonra kural her gun kendiliginden tetikleniyor.

Calistirma:  python teshis_siparis.py
"""
from pathlib import Path

import pandas as pd

from tools_toplu import ROSSMAN, _projeksiyon

p = _projeksiyon()
yok = p[(p["mod"] == "siparis_yok") & (p["talep_senaryo"] == "p90") & (p["acik"] == 1)]
pol = p[(p["mod"] == "politika") & (p["talep_senaryo"] == "p50") & (p["acik"] == 1)]

t = pd.DataFrame({
    "yok_tetik_%": (yok.assign(x=yok["stoksuz"] > 0.5).groupby("h")["x"].mean() * 100).round(1),
    "yok_acilis_medyan": yok.groupby("h")["acilis"].median().round(0),
    "politika_acilis_medyan": pol.groupby("h")["acilis"].median().round(0),
})
print("Ufuk gunune gore (h=1 -> 07-04):")
print(t.to_string())

kd = Path(ROSSMAN) / "karar_degerlendirme.csv"
if kd.exists():
    d = pd.read_csv(kd, parse_dates=["tarih"])
    d = d[~d["magaza_kapali"].astype(bool)]
    s = (d.groupby("tarih")["agent"].apply(lambda x: (x == "siparis_ver").mean()) * 100).round(1)
    print("\nAgent siparis_ver orani, gune gore:")
    print(s.to_string())