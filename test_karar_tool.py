"""
test_karar_tool.py — karar_uret kural motoru testleri

NEYI TEST EDIYOR
    LLM'i degil, karar_tool.py icindeki DETERMINISTIK kural motorunu.
    test_agent.py ayni seyi agent_v6 icin yapar; bu dosya v7'nin magaza
    bazli karar aracini kapsar. LLM cagrilmadigi icin ucretsiz, saniyeler
    surer ve tekrarlanabilir.

NEDEN AYRI DOSYA
    test_agent.py agent_v6'nin kurallari_dayat fonksiyonunu test eder ve
    16/16 geciyor. Arayuzler farkli (biri Pydantic nesnesi alip duzeltir,
    digeri sifirdan JSON uretir), tek dosyada birlestirmek ikisini de
    okunmaz yapardi.

TEST GUNLERI KODA GOMULU DEGIL
    Veri degisirse testler kendini uyarlar: uygun magaza-gun
    kombinasyonlari projeksiyon tablosundan taranarak bulunur.

Fire kosullari BEKLENEN fireye (Swanson: 0.3 p10 + 0.4 p50 + 0.3 p90)
    gore secilir; karar_tool ile ayni tanim.

Calistirma:
    python test_karar_tool.py
"""

import json
import sys

import pandas as pd

import karar_tool as K
from tools_toplu import _ozet, _perf, _projeksiyon, _tahmin

SONUCLAR = []


def dogrula(ad: str, beklenen: str, kosul: bool, detay: str = "") -> None:
    SONUCLAR.append({"ad": ad, "beklenen": beklenen, "gecti": bool(kosul)})
    print(f"  [{'GECTI' if kosul else 'KALDI'}] {ad}")
    if not kosul and detay:
        print(f"          -> {detay}")


def karar(magaza: int, tarih: str, **kw) -> dict:
    """karar_uret'i cagirip JSON'u sozluge cevirir."""
    ham = K.karar_uret(magaza, tarih, **kw)
    try:
        return json.loads(ham)
    except json.JSONDecodeError:
        return {"__hata_metni__": ham}


def hata_mi(k: dict) -> bool:
    """Arac katmani hatayi YAPISAL olarak dondurur: {"hata": "..."}.
    Duz metin bicimi de kabul edilir; onemli olan kararin URETILMEMIS
    olmasi ve sebebin okunabilir sekilde donmesidir."""
    if "__hata_metni__" in k:
        return True
    if "hata" in k and "karar" not in k:
        return bool(k["hata"])
    return False


# ============================================================================
# TEST GUNU SECIMI — veriden taranarak
# ============================================================================
def senaryo_tablosu() -> pd.DataFrame:
    """Politika modu p50 satirlari + p10/p90 fireleri + beklenen fire.
    Beklenen fire karar_tool.SWANSON ile AYNI tanimdan hesaplanir."""
    pro = _projeksiyon()
    pol = pro[pro["mod"] == "politika"]
    g = pol[pol["talep_senaryo"] == "p50"].copy()
    for q in ("p10", "p90"):
        f = pol[pol["talep_senaryo"] == q][["magaza", "tarih", "fire"]] \
            .rename(columns={"fire": f"fire_{q}"})
        g = g.merge(f, on=["magaza", "tarih"], how="left")
    g["fire_p10"] = g["fire_p10"].fillna(g["fire"])
    g["fire_p90"] = g["fire_p90"].fillna(g["fire"])
    g["fire_beklenen"] = (K.SWANSON["p10"] * g["fire_p10"]
                          + K.SWANSON["p50"] * g["fire"]
                          + K.SWANSON["p90"] * g["fire_p90"])
    return g


def gun_sec() -> dict:
    """Her senaryo icin uygun bir magaza-gun cifti bulur."""
    tah = _tahmin()
    tah = tah[tah["senaryo"] == "planli"]
    ozet = _ozet()
    perf = _perf()

    g = senaryo_tablosu().merge(tah[["magaza", "tarih", "p50_adet"]],
                                on=["magaza", "tarih"])
    g = g.merge(ozet[["magaza", "tipik_gunluk_talep_adet"]], on="magaza")
    # Kapi hukmu karar_tool'daki AYNI fonksiyondan gelir (kural tek yerde).
    perf = perf.copy()
    perf["zayif"] = [K.belirsizlik_kapisi(r.smape_val, r.kapsama_val, r.n_val)["zayif"]
                     for r in perf.itertuples()]
    g = g.merge(perf[["magaza", "zayif"]], on="magaza")
    payda = g["tipik_gunluk_talep_adet"].clip(lower=1e-6)
    g["fire_oran"] = g["fire_beklenen"] / payda
    g["fire_oran_plan"] = g["fire"] / payda

    secim = {}

    # 1) Kapali gun: tahmin 0 ve elde stok var
    kapali = g[(g["p50_adet"] <= 0) & (g["acilis"] > 0)]
    if len(kapali):
        s = kapali.iloc[0]
        secim["kapali"] = (int(s.magaza), str(s.tarih.date()))

    # 2) Beklenen fire yuksek + model GUCLU + o gun siparis yok -> indirim_uygula
    fire_guclu = g[(g["fire_oran"] > K.FIRE_ORAN_ESIGI) & (g["p50_adet"] > 0)
                   & (~g["zayif"]) & (g["siparis"] < 1)]
    if len(fire_guclu):
        s = fire_guclu.sort_values("fire_oran", ascending=False).iloc[0]
        secim["fire_guclu"] = (int(s.magaza), str(s.tarih.date()))

    # 3) Beklenen fire yuksek + model ZAYIF -> insana_sor
    fire_zayif = g[(g["fire_oran"] > K.FIRE_ORAN_ESIGI) & (g["p50_adet"] > 0)
                   & (g["zayif"])]
    if len(fire_zayif):
        s = fire_zayif.sort_values("fire_oran", ascending=False).iloc[0]
        secim["fire_zayif"] = (int(s.magaza), str(s.tarih.date()))

    # 4) Sakin gun: beklenen fire yok, stok bol, model guclu, siparis yok -> bekle
    sakin = g[(g["fire_beklenen"] < 0.5) & (g["p50_adet"] > 0)
              & (g["acilis"] > g["p50_adet"] * 2)
              & (~g["zayif"]) & (g["siparis"] < 1)]
    if len(sakin):
        s = sakin.iloc[0]
        secim["sakin"] = (int(s.magaza), str(s.tarih.date()))

    # 5) Plan fireyi esik altinda goruyor ama beklenen fire esigi asiyor
    #    (tahmin hatasindan dogacak fire) — model guclu, siparis yok
    gizli = g[(g["fire_oran_plan"] <= K.FIRE_ORAN_ESIGI)
              & (g["fire_oran"] > K.FIRE_ORAN_ESIGI) & (g["p50_adet"] > 0)
              & (~g["zayif"]) & (g["siparis"] < 1)]
    if len(gizli):
        s = gizli.sort_values("fire_oran", ascending=False).iloc[0]
        secim["gizli_fire"] = (int(s.magaza), str(s.tarih.date()))

    # 6) Genel amacli gecerli bir gun
    normal = g[g["p50_adet"] > 0]
    s = normal.iloc[0]
    secim["normal"] = (int(s.magaza), str(s.tarih.date()))
    return secim


# ============================================================================
# SENARYOLAR
# ============================================================================
GECERLI_KARARLAR = {"siparis_ver", "bekle", "indirim_uygula", "insana_sor"}
ZORUNLU_ALANLAR = ["olay", "tarih", "magaza", "karar", "aciliyet", "miktar",
                   "tedarikci", "onay_gerekli_mi", "indirim_orani",
                   "net_etki_eur", "gerekce", "tahmin_satis", "guven_alt",
                   "guven_ust", "mevcut_stok", "esikler"]


def senaryolar() -> None:
    g = gun_sec()
    print("\nSecilen test gunleri:")
    for ad, (m, t) in g.items():
        print(f"  {ad:12s} magaza {m:4d}  {t}")
    print()

    # ---- 1. SEMA BUTUNLUGU ----------------------------------------------
    print("1) SEMA BUTUNLUGU")
    m, t = g["normal"]
    k = karar(m, t)

    eksik = [a for a in ZORUNLU_ALANLAR if a not in k]
    dogrula("Zorunlu alanlarin tamami donuyor", "eksik yok", not eksik,
            f"eksik: {eksik}")

    dogrula("olay alani sabit", "stok_karari", k.get("olay") == "stok_karari",
            f"gelen: {k.get('olay')}")

    dogrula("karar dort degerden biri", str(GECERLI_KARARLAR),
            k.get("karar") in GECERLI_KARARLAR, f"gelen: {k.get('karar')}")

    dogrula("aciliyet tanimli kumeden", "kritik/yuksek/orta/dusuk",
            k.get("aciliyet") in {"kritik", "yuksek", "orta", "dusuk"},
            f"gelen: {k.get('aciliyet')}")

    ESIK_ALANLARI = {"onay_esigi_eur", "fire_oran_esigi", "stok_gun_alt_esik",
                     "smape_belirsiz_esik", "kapsama_hedef", "kapsama_p_esik"}
    dogrula("esikler ciktida tasiniyor (izlenebilirlik)", "6 esik",
            isinstance(k.get("esikler"), dict) and ESIK_ALANLARI <= set(k["esikler"]),
            f"gelen: {k.get('esikler')}")

    dogrula("Kapi yalniz val metrigiyle calisir (sizinti yok)", "val alanlari dolu",
            "model_smape_val" in k and "model_kapsama_val" in k
            and "test sMAPE" not in k.get("gerekce", ""),
            f"gerekce: {k.get('gerekce', '')[:120]}")

    # ---- 2. KAPALI MAGAZA ------------------------------------------------
    print("\n2) KAPALI MAGAZA (kural sirasinda ilk kontrol)")
    if "kapali" in g:
        m, t = g["kapali"]
        k = karar(m, t)
        dogrula("Kapali gunde karar insana devredilir", "insana_sor",
                k.get("karar") == "insana_sor", f"gelen: {k.get('karar')}")
        dogrula("Kapali gunde siparis miktari sifir", "miktar=0",
                k.get("miktar") == 0, f"miktar={k.get('miktar')}")
        dogrula("Kapali gunde indirim orani sifir", "0.0",
                k.get("indirim_orani") == 0.0,
                f"oran={k.get('indirim_orani')}")
        dogrula("magaza_kapali bayragi true", "True",
                k.get("magaza_kapali") is True,
                f"gelen: {k.get('magaza_kapali')}")
        # Bu, 20 Eylul'de yakalanan gercek bir hatanin regresyon testidir:
        # kapali magazada onay sebebi "model belirsizligi" yaziliyordu.
        sebep = (k.get("onay_sebebi") or "").lower()
        dogrula("Onay sebebi kapali magazayi dogru gosterir (regresyon)",
                "'magaza kapali' gecmeli", "kapali" in sebep,
                f"gelen: {k.get('onay_sebebi')}")
        dogrula("Kapali gun gerekcesinde 'KAPALI' gecer", "gecmeli",
                "KAPALI" in (k.get("gerekce") or ""),
                f"gerekce: {(k.get('gerekce') or '')[:80]}")
    else:
        print("  (veri setinde kapali gun bulunamadi, atlandi)")

    # ---- 3. FIRE KURALI --------------------------------------------------
    print("\n3) FIRE KURALI (model guclu iken)")
    if "fire_guclu" in g:
        m, t = g["fire_guclu"]
        k = karar(m, t)
        dogrula("Fire esigi asildiginda indirim karari uretilir",
                "indirim_uygula", k.get("karar") == "indirim_uygula",
                f"gelen: {k.get('karar')}, fire_oran="
                f"{k.get('fire_gun_karsiligi')}")
        dogrula("Indirim orani ONDALIK yazilir (0-1)", "0 < oran <= 1",
                0 < (k.get("indirim_orani") or 0) <= 1,
                f"oran={k.get('indirim_orani')}")
        dogrula("Indirim kararinda siparis miktari sifir", "miktar=0",
                k.get("miktar") == 0, f"miktar={k.get('miktar')}")
        dogrula("Fire orani esigi asmis olarak raporlanir", "> esik",
                (k.get("fire_gun_karsiligi") or 0) > K.FIRE_ORAN_ESIGI,
                f"oran={k.get('fire_gun_karsiligi')}")
    else:
        print("  (fire esigini asan ve modeli guclu magaza-gun yok, atlandi)")

    # ---- 4. BELIRSIZLIK KAPISI ------------------------------------------
    print("\n4) BELIRSIZLIK KAPISI (model zayif iken)")
    if "fire_zayif" in g:
        m, t = g["fire_zayif"]
        k = karar(m, t)
        dogrula("Model zayifken karar otomatik uygulanmaz", "insana_sor",
                k.get("karar") == "insana_sor", f"gelen: {k.get('karar')}")
        dogrula("Devredilen kararda indirim orani sifirlanir (celiski yok)",
                "0.0", k.get("indirim_orani") == 0.0,
                f"oran={k.get('indirim_orani')}")
        dogrula("Onerilen aksiyon kaybolmaz, ayri alanda tasinir",
                "bos olmamali", bool(k.get("onerilen_aksiyon")),
                f"gelen: '{k.get('onerilen_aksiyon')}'")
    else:
        print("  (fire riski olan zayif model magazasi yok, atlandi)")

    # ---- 5. SAKIN GUN ----------------------------------------------------
    print("\n5) SAKIN GUN")
    if "sakin" in g:
        m, t = g["sakin"]
        k = karar(m, t)
        dogrula("Fire yok ve stok bolken aksiyon uretilmez", "bekle",
                k.get("karar") == "bekle",
                f"gelen: {k.get('karar')}, gerekce="
                f"{(k.get('gerekce') or '')[:70]}")
        dogrula("Bekle kararinda onay gerekmez", "False",
                k.get("onay_gerekli_mi") is False,
                f"sebep: {k.get('onay_sebebi')}")
    else:
        print("  (sakin gun bulunamadi, atlandi)")

    # ---- 6. HATALI GIRDI -------------------------------------------------
    print("\n6) HATALI GIRDI — sessizce gecmemeli")
    k = karar(99999, g["normal"][1])
    dogrula("Gecersiz magaza numarasi hata dondurur", "yapisal hata",
            hata_mi(k), f"gelen: {str(k)[:90]}")

    k = karar(g["normal"][0], "2016-01-15")
    dogrula("Pencere disi tarih hata dondurur", "yapisal hata",
            hata_mi(k), f"gelen: {str(k)[:90]}")

    k = karar(g["normal"][0], "gecersiz-tarih")
    dogrula("Bozuk tarih bicimi hata dondurur", "yapisal hata",
            hata_mi(k), f"gelen: {str(k)[:90]}")

    # ---- 7. TUTARLILIK — coklu gun --------------------------------------
    print("\n7) TUTARLILIK (20 magaza-gun uzerinde)")
    pro = _projeksiyon()
    ornek = pro[(pro["mod"] == "politika")
                & (pro["talep_senaryo"] == "p50")].head(20)

    bozuk_sema, bozuk_karar, celiski = 0, 0, 0
    for _, s in ornek.iterrows():
        k = karar(int(s.magaza), str(s.tarih.date()))
        if hata_mi(k):
            continue
        if any(a not in k for a in ZORUNLU_ALANLAR):
            bozuk_sema += 1
        if k.get("karar") not in GECERLI_KARARLAR:
            bozuk_karar += 1
        # insana_sor veya bekle iken aksiyon alanlari dolu olmamali
        if k.get("karar") in ("insana_sor", "bekle") and (
                k.get("miktar") or k.get("indirim_orani")):
            celiski += 1

    dogrula("Sema hicbir gunde bozulmuyor", "0 bozuk", bozuk_sema == 0,
            f"bozuk: {bozuk_sema}")
    dogrula("Karar seti disina cikilmiyor", "0 gecersiz", bozuk_karar == 0,
            f"gecersiz: {bozuk_karar}")
    dogrula("Aksiyonsuz kararlarda miktar/oran dolu degil", "0 celiski",
            celiski == 0, f"celiskili satir: {celiski}")

    # ---- 8. SIZINTI REGRESYONU -------------------------------------------
    # Test donemi metrikleri karar aninda bilinemez. Onlari bozdugumuzda
    # hicbir karar degismemeli; degisiyorsa kural gelecege bakiyor demektir.
    print("\n8) SIZINTI REGRESYONU (test metrikleri bozulunca karar degismemeli)")
    import tools_toplu as T
    once = [karar(int(s.magaza), str(s.tarih.date())) for _, s in ornek.iterrows()]
    asil_perf = T._perf
    bozuk = asil_perf().copy()
    bozuk["smape_test"] = 99.0
    bozuk["kapsama_test"] = 1.0
    T._perf = lambda: bozuk
    K._perf = T._perf
    try:
        sonra = [karar(int(s.magaza), str(s.tarih.date())) for _, s in ornek.iterrows()]
    finally:
        T._perf = asil_perf
        K._perf = asil_perf
    alanlar = ("karar", "miktar", "indirim_orani", "onay_gerekli_mi", "gerekce")
    fark = sum(1 for a, b in zip(once, sonra)
               if any(a.get(x) != b.get(x) for x in alanlar))
    dogrula("Test metrikleri kararlari etkilemiyor", "0 fark", fark == 0,
            f"degisen karar: {fark}")

    # ---- 9. SIPARIS KURALI ---------------------------------------------
    # Regresyon: onceki kural siparis_yok modunun stoksuzluguna bakiyordu ve
    # 4. gunden sonra her acik magazada tetikleniyordu (%94.5 siparis).
    # Yeni kural politika motorunun o gunku siparisine baglidir.
    print("\n9) SIPARIS KURALI (politika motoruna bagli, ufukla dejenere olmaz)")
    tah = _tahmin()
    tah = tah[tah["senaryo"] == "planli"][["magaza", "tarih", "p50_adet"]]
    gec = senaryo_tablosu().merge(tah, on=["magaza", "tarih"])
    gec = gec[(gec["tarih"] >= gec["tarih"].min() + pd.Timedelta(days=5))
              & (gec["p50_adet"] > 0)]
    orn = gec.sample(n=min(60, len(gec)), random_state=7)

    yanlis, eksik, miktar_farki, n_sip = 0, 0, 0, 0
    swanson_farki = 0
    for s in orn.itertuples():
        k = karar(int(s.magaza), str(s.tarih.date()))
        if hata_mi(k):
            continue
        if k.get("karar") == "siparis_ver":
            n_sip += 1
            if s.siparis < 1:
                yanlis += 1
            if k.get("miktar") != int(round(s.siparis)):
                miktar_farki += 1
        if s.siparis >= 1 and k.get("karar") == "bekle":
            eksik += 1
        if abs((k.get("fire_riski_birim") or 0) - round(s.fire_beklenen, 1)) > 0.11:
            swanson_farki += 1
    print(f"  (bilgi) orneklemde siparis_ver orani: {n_sip}/{len(orn)}")
    dogrula("Politika siparis vermiyorsa agent da vermiyor (dejenerasyon regresyonu)",
            "0 yanlis", yanlis == 0, f"yanlis siparis: {yanlis}")
    dogrula("Politika siparis veriyorsa agent 'bekle' demiyor", "0 eksik",
            eksik == 0, f"eksik siparis: {eksik}")
    dogrula("Siparis miktari politika motoruyla ayni", "0 fark",
            miktar_farki == 0, f"farkli miktar: {miktar_farki}")

    # Celiskili sinyal: beklenen fire esigi asti VE ayni gun politika siparisi var
    oz = _ozet()[["magaza", "tipik_gunluk_talep_adet"]]
    c = gec.merge(oz, on="magaza")
    c = c[(c["fire_beklenen"] / c["tipik_gunluk_talep_adet"].clip(lower=1e-6)
           > K.FIRE_ORAN_ESIGI) & (c["siparis"] >= 1)]
    if len(c):
        s = c.iloc[0]
        k = karar(int(s.magaza), str(s.tarih.date()))
        dogrula("Celiskili sinyalde karar insana devredilir", "insana_sor",
                k.get("karar") == "insana_sor", f"gelen: {k.get('karar')}")
        dogrula("Celiskide iki oneri de korunur", "'+' iceren oneri",
                "+" in (k.get("onerilen_aksiyon") or ""),
                f"gelen: '{k.get('onerilen_aksiyon')}'")
    else:
        print("  (celiskili gun bulunamadi, atlandi)")

    # ---- 10. BEKLENEN FIRE ----------------------------------------------
    # Yalniz plana (p50) bakan sinyal tahmin hatasindan dogan fireyi goremez;
    # sinyal Swanson kuraliyla beklenen fireye bakar.
    print("\n10) BEKLENEN FIRE (Swanson: 0.3 p10 + 0.4 p50 + 0.3 p90)")
    dogrula("Swanson agirliklari toplami 1", "1.0",
            abs(sum(K.SWANSON.values()) - 1) < 1e-9, f"toplam={sum(K.SWANSON.values())}")
    dogrula("fire_riski_birim beklenen fireye esit", "0 fark",
            swanson_farki == 0, f"farkli gun: {swanson_farki}")
    if "gizli_fire" in g:
        m, t = g["gizli_fire"]
        k = karar(m, t)
        dogrula("Planin gormedigi fire (dusuk talep riski) sinyal uretir",
                "indirim_uygula", k.get("karar") == "indirim_uygula",
                f"gelen: {k.get('karar')}, plan={k.get('fire_plan_birim')}, "
                f"beklenen={k.get('fire_riski_birim')}")
        dogrula("Gerekce plan ve dusuk talep firesini ayri gosterir", "'dusuk talepte' gecmeli",
                "dusuk talepte" in (k.get("gerekce") or ""),
                f"gerekce: {(k.get('gerekce') or '')[:100]}")
    else:
        print("  (plan esik alti / beklenen esik ustu gun bulunamadi, atlandi)")


# ============================================================================
def main() -> None:
    print("=" * 72)
    print("KARAR_TOOL KURAL MOTORU TESTLERI")
    print("=" * 72)

    senaryolar()

    gecen = sum(1 for s in SONUCLAR if s["gecti"])
    toplam = len(SONUCLAR)
    print("\n" + "=" * 72)
    print(f"SONUC: {gecen}/{toplam} senaryo gecti")
    if gecen < toplam:
        print("\nKALAN SENARYOLAR:")
        for s in SONUCLAR:
            if not s["gecti"]:
                print(f"  - {s['ad']}  (beklenen: {s['beklenen']})")
    print("=" * 72)
    sys.exit(0 if gecen == toplam else 1)


if __name__ == "__main__":
    main()
