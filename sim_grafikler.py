"""
sim_grafikler.py — Stok simulasyonu sonuclarindan sunum grafikleri uretir.

Uretilen grafikler (Rossman/figs/):
    20_tedarikci_net_kar.png      : manset bulgu — hangi tedarikci kazaniyor
    21_fire_stoksuz.png           : mekanizma — kar farki nereden geliyor
    22_raf_omru_duyarlilik.png    : dogru tedarikci raf omrune gore degisiyor
    23_stok_seyri.png             : stok seviyesi zaman icinde, tedarikci bazinda

Onkosul:
    python stok_simulasyon.py    (en az bir kez calismis olmali)

Calistirma:
    python sim_grafikler.py
"""

import matplotlib
matplotlib.use("Agg")            # pencere acma, dosyaya yaz (plt.show() takilmasin)
import matplotlib.pyplot as plt

from stok_simulasyon import (
    KOK,
    TEDARIKCILER,
    RAF_OMRU_GUN,
    simule_et,
    veri_hazirla,
)

FIG_DIR = KOK / "figs"
RENKLER = {
    "Schnellware": "#4C72B0",
    "Nordmann": "#C44E52",
    "ExpressLog": "#55A868",
    "Rheinland": "#DD8452",
}


def en_iyi_ozet(df, ad: str, raf: int = RAF_OMRU_GUN) -> dict:
    adaylar = [simule_et(df, ad, p, raf_omru=raf)[1] for p in ("ortalama", "newsvendor")]
    return max(adaylar, key=lambda x: x["net_kar"])


def grafik_net_kar(df) -> None:
    ozetler = {ad: en_iyi_ozet(df, ad) for ad in TEDARIKCILER}
    adlar = list(ozetler)
    karlar = [ozetler[a]["net_kar"] for a in adlar]

    fig, ax = plt.subplots(figsize=(9, 5))
    barlar = ax.bar(adlar, karlar, color=[RENKLER[a] for a in adlar])
    ax.axhline(0, color="black", linewidth=0.9)

    for bar, kar, ad in zip(barlar, karlar, adlar):
        y = kar + (400 if kar >= 0 else -900)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{kar:,.0f} €",
                ha="center", fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2,
                kar / 2 if kar > 2000 else 900,
                f"birim fiyat\n×{TEDARIKCILER[ad]['carpan']:.2f}",
                ha="center", fontsize=8, color="white" if kar > 2000 else "black")

    ax.set_title(f"Tedarikçi bazında net kâr — 36 gün, {RAF_OMRU_GUN} günlük raf ömrü\n"
                 "En pahalı tedarikçi (ExpressLog) en kârlısı", fontsize=12)
    ax.set_ylabel("Net kâr (€)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "20_tedarikci_net_kar.png", dpi=130)
    plt.close(fig)


def grafik_fire_stoksuz(df) -> None:
    ozetler = {ad: en_iyi_ozet(df, ad) for ad in TEDARIKCILER}
    adlar = list(ozetler)
    fire = [ozetler[a]["fire_birim"] for a in adlar]
    stoksuz = [ozetler[a]["stoksuz_birim"] for a in adlar]

    x = range(len(adlar))
    genislik = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - genislik / 2 for i in x], fire, genislik,
           label="Fire (raf ömrü doldu)", color="#C44E52")
    ax.bar([i + genislik / 2 for i in x], stoksuz, genislik,
           label="Stoksuz kalma (talep karşılanamadı)", color="#8C8C8C")

    ax.set_xticks(list(x))
    ax.set_xticklabels(adlar)
    ax.set_ylabel("Birim")
    ax.set_title("Kâr farkı nereden geliyor: fire ve stoksuz kalma\n"
                 "ExpressLog her ikisinde de sıfıra yakın", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "21_fire_stoksuz.png", dpi=130)
    plt.close(fig)


def grafik_tarama(df) -> None:
    raflar = [2, 3, 4, 5, 7]
    fig, ax = plt.subplots(figsize=(9, 5))

    for ad in TEDARIKCILER:
        karlar = [en_iyi_ozet(df, ad, raf)["net_kar"] for raf in raflar]
        ax.plot(raflar, karlar, marker="o", label=ad, color=RENKLER[ad], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.axvline(RAF_OMRU_GUN, color="gray", linestyle=":", alpha=0.7)
    ax.text(RAF_OMRU_GUN + 0.08, ax.get_ylim()[0] * 0.9,
            f"seçilen: {RAF_OMRU_GUN} gün", fontsize=8, color="gray")

    ax.set_xlabel("Raf ömrü (gün)")
    ax.set_ylabel("Net kâr (€)")
    ax.set_title("Doğru tedarikçi raf ömrüne göre değişiyor\n"
                 "Sabit bir 'en iyi tedarikçi' yok — karar bağlama bağlı", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "22_raf_omru_duyarlilik.png", dpi=130)
    plt.close(fig)


def grafik_stok_seyri(df) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    for ad in TEDARIKCILER:
        detay, _ = simule_et(df, ad, "ortalama")
        ax.plot(range(len(detay)), detay["kapanis_stok"],
                label=ad, color=RENKLER[ad], linewidth=1.8)

    ax.plot(range(len(df)), df["talep_gercek"], color="black",
            linestyle="--", linewidth=1.2, label="Günlük talep (gerçek)")

    ax.set_xlabel("Gün (test dönemi)")
    ax.set_ylabel("Kapanış stoğu (birim)")
    ax.set_title("Stok seyri: yüksek minimum sipariş ve uzun lead-time stok yığıyor\n"
                 "Nordmann talebin çok üstünde stok taşıyor, ExpressLog talebe yapışık",
                 fontsize=12)
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "23_stok_seyri.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = veri_hazirla(KOK / "prophet_forecast.csv")

    grafik_net_kar(df)
    grafik_fire_stoksuz(df)
    grafik_tarama(df)
    grafik_stok_seyri(df)

    print(f"4 grafik kaydedildi -> {FIG_DIR}")
    for ad in ("20_tedarikci_net_kar.png", "21_fire_stoksuz.png",
               "22_raf_omru_duyarlilik.png", "23_stok_seyri.png"):
        print(f"  {ad}")