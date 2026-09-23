"""senaryo_uret.py — TELEGRAM_TEST_SENARYOLARI.md'yi GERCEK karar ciktisindan uretir.

B grubundaki magaza-gun ciftleri elle secilmez; karar_tool'un urettigi
Rossman/karar_onbellek.csv'den (gercek_backtest.py yazar) her karar tipi icin
bir ornek alinir. Karar kurali degisirse bu script yeniden calistirilir.

Calistirma:  python senaryo_uret.py
"""
from pathlib import Path

import pandas as pd

from tools_toplu import ROSSMAN

d = pd.read_csv(Path(ROSSMAN) / "karar_onbellek.csv", parse_dates=["tarih"])
d["oner"] = d["onerilen_aksiyon"].fillna("").astype(str)
d["onay"] = d["onay_gerekli_mi"].astype(str).str.lower().eq("true")
d["kapali"] = d["magaza_kapali"].astype(str).str.lower().eq("true")
d = d[d["tarih"] >= "2015-07-06"]          # ilk iki gun baslangic stoku etkisi

KATEGORI = [
    ("B1", "siparis_ver, onay YOK", "Otomatik Siparis -> Telegram Aksiyon Bildirimi (✅, varis tarihi yazar)",
     (d.karar == "siparis_ver") & ~d.onay),
    ("B2", "siparis_ver, onay VAR", "Siparis Eskalasyon -> Telegram Onay Bildirimi (🔴 + butonlar)",
     (d.karar == "siparis_ver") & d.onay),
    ("B3", "indirim_uygula, onay VAR", "Indirim Onaya -> Telegram Onay Bildirimi",
     d.karar == "indirim_uygula"),
    ("B4", "bekle (sakin gun)", "Bekle Log -> Telegram Bekle Bildirimi (ℹ️ aksiyon gerekmiyor)",
     (d.karar == "bekle") & (d.stoksuz_p90_bugun.fillna(0) < 0.5) & (d.fire_riski_birim.fillna(0) < 0.5)),
    ("B5", "bekle + stok UYARISI", "Bekle Log -> Telegram Bekle Bildirimi (gerekcede 'UYARI: kotumser talepte...')",
     (d.karar == "bekle") & (d.stoksuz_p90_bugun.fillna(0) > 0.5)),
    ("B6", "insana_sor (model zayif)", "Insan Onayi -> Telegram Onay Bildirimi (Sebep: model belirsizligi)",
     (d.karar == "insana_sor") & ~d.kapali & (d.oner != "") & ~d.oner.str.contains(r"\+", regex=True)),
    ("B7", "insana_sor (magaza KAPALI)", "Insan Onayi; miktar 0, gerekcede 'KAPALI'",
     (d.karar == "insana_sor") & d.kapali),
    ("B8", "insana_sor (CELISKILI SINYAL)", "Insan Onayi; onerilen aksiyonda 'indirim + siparis'",
     (d.karar == "insana_sor") & d.oner.str.contains(r"\+", regex=True)),
]

satirlar, secilen = [], {}
for kod, beklenen, dal, maske in KATEGORI:
    aday = d[maske]
    if aday.empty:
        satirlar.append(f"| {kod} | _(veride bu karar tipi yok — atla)_ | {beklenen} | {dal} |")
        continue
    s = aday.sample(1, random_state=42).iloc[0]
    secilen[kod] = s
    mesaj = f"Mağaza {int(s.magaza)} için {s.tarih.day} Temmuz'da karar ver"
    satirlar.append(f"| {kod} | `{mesaj}` | {beklenen} | {dal} |")

B_TABLO = "\n".join(satirlar)
onayli = [k for k in ("B2", "B3", "B6") if k in secilen]

MD = f"""# TELEGRAM & n8n TEST SENARYOLARI

Amaç: n8n'deki **45 node'un tamamını** en az bir kez çalıştırmak ve
agent'ın 12 aracının hepsini tetiklemek.

**Ön koşul:** Sistem ayakta (Docker, ngrok, n8n workflow Active, FastAPI çalışıyor).
Tahmin penceresi **2015-07-04 – 2015-07-31**. Bu aralık dışı tarihler
bilinçli olarak reddedilir.

**Nasıl kullanılır:** Her satırdaki mesajı Telegram'a yaz, "Beklenen"
sütunuyla karşılaştır. Sapma varsa n8n → Executions sekmesinde kırmızı X
olan node'a bak.

> B grubu çiftleri elle seçilmez; `senaryo_uret.py` gerçek karar çıktısından
> her karar tipi için bir örnek seçer. Karar kuralı değişirse betik yeniden
> çalıştırılır.

---

## A. ARAÇ SEÇİMİ — 12 aracın her biri (A1–A12)

| # | Telegram mesajı | Beklenen araç | Beklenen cevap |
|---|---|---|---|
| A1 | `Hangi mağazalarda stok tükeniyor?` | tukenme_riski | Mağaza listesi (en kötü durum: sipariş verilmezse) |
| A2 | `7 Temmuz sevkiyat planı ne?` | sevkiyat_plani_getir | O günkü sevkiyatlar |
| A3 | `En çok ihtiyacı olan mağazalar hangileri?` | en_cok_ihtiyac_duyan | Öncelik sıralı liste |
| A4 | `B tipi mağazalar nasıl?` | segment_ozeti | Segment ortalamaları |
| A5 | `Mağaza 530'un tahmini ne kadar güvenilir?` | model_performansi | val sMAPE, kapsama |
| A6 | `10 Temmuz'da fire riski olan mağazalar?` | raf_omru_durumu | Bayat mal biriken mağazalar |
| A7 | `Elimde 50 bin birim var nasıl dağıtayım, 10 Temmuz?` | dagitim_plani | Mağazalara dağılım |
| A8 | `Mağaza 530 ile 1'i karşılaştır` | magaza_karsilastir | İki mağaza yan yana |
| A9 | `Mağaza 200'de promosyon yapsak talep ne olur?` | promo_senaryosu | promo_var / promo_yok farkı |
| A10 | `Schnellware'ın minimum sipariş miktarı ne?` | tedarikci_bilgisi_ara | RAG'den politika cevabı |
| A11 | `Model zamanla kötüleşiyor mu?` | trend_ozeti | Haftalık drift + model hatası EUR/100 adet |
| A12 | `264 numaralı mağaza için 20 Temmuz'da karar ver` | karar_uret | **Karar üretir → B grubu hattı** |

---

## B. KARAR YOLU — karar tipleri ve dallar

Her biri Google Sheets `Sayfa1`'e satır yazmalı (`karar_id` = mağaza_tarih).

| # | Telegram mesajı | Beklenen karar | Beklenen n8n dalı |
|---|---|---|---|
{B_TABLO}

---

## C. ONAY DÖNGÜSÜ — buton akışı

{", ".join(onayli) or "B2/B3/B6"} mesajlarındaki butonları kullan.

| # | Aksiyon | Beklenen |
|---|---|---|
| C1 | **✅ Onayla** | `✅ Karar ... ONAYLANDI`; Sheets `durum` = onaylandi, `onay_zamani` dolu |
| C2 | Başka bir kararda **❌ Reddet** | 4 seçenekli sebep menüsü |
| C3 | Sebep: **Tahmin gerçekçi değil** | `❌ ... REDDEDİLDİ`; `red_sebebi` dolu |
| C4 | Farklı kararı reddet, **Miktar/oran yanlış** | Aynı akış, farklı sebep kodu |
| C5 | Aynı butona **ikinci kez** bas | Telegram hata vermez |

Sonra yerelde `python ogrenme_dongusu.py` → eşik önerileri.

> **Bilinen sınır:** Onaylanmış bir kararı (aynı mağaza + gün) tekrar
> sorarsan `appendOrUpdate` satırı `onay_bekliyor`'a döndürür. Demo'da
> onaylanan kararı tekrar sorma.

---

## D. SINIR VE HATA DAVRANIŞI

| # | Telegram mesajı | Beklenen |
|---|---|---|
| D1 | `Bayat mal kimde birikmiş? 2015-06-30` | "Veri yok", geçerli aralık + örnek, araç yok |
| D2 | `Mağaza 5000 için 10 Temmuz'da karar ver` | Geçersiz mağaza hatası |
| D3 | `Mağaza 200 için 2015-13-45'te karar ver` | Bozuk tarih hatası |
| D4 | `Mağaza 200 için 2016-01-15'te karar ver` | Pencere dışı tarih hatası |
| D5 | `Hava nasıl?` | Kibarca reddeder, araç yok |
| D6 | `Bana bir şiir yaz` | Reddeder |
| D7 | `Python'da liste nasıl sıralanır?` | Reddeder |
| D8 | `Türkiye'nin başkenti neresi?` | Reddeder |

---

## E. ÇOK ADIMLI VE BELİRSİZ SORULAR

| # | Telegram mesajı | Beklenen |
|---|---|---|
| E1 | `Stok tükenen mağazalara promosyon yapılmalı mı?` | Araç zinciri (tukenme_riski + promo_senaryosu) |
| E2 | `Hangisi daha riskli, 250 mi 800 mü?` | magaza_karsilastir |
| E3 | `Kimde raf ömrü doluyor?` (tarihsiz) | Tarih sorar veya varsayılanla cevaplar |
| E4 | `Mağaza 292'nin tahmini güvenilir mi?` | model_performansi; **zayıf** |
| E5 | `Mağaza 1'in tahmini güvenilir mi?` | model_performansi; **güçlü** |
| E6 | `Dün ne konuştuk?` | Hatırlamadığını söyler (bellek oturum bazlı) |

---

## F. WEBHOOK YOLU (Telegram dışı giriş)

```powershell
# F1 — geçerli payload
curl.exe -X POST "http://localhost:5678/webhook/stok-karar" `
  -H "Content-Type: application/json" `
  -d '{{\\"tarih\\":\\"2015-07-15\\",\\"karar\\":\\"siparis_ver\\",\\"aciliyet\\":\\"yuksek\\",\\"magaza\\":269,\\"miktar\\":120,\\"tedarikci\\":\\"Schnellware\\",\\"onay_gerekli_mi\\":false,\\"gerekce\\":\\"test\\"}}'
```
Beklenen: JSON yanıtta `tarih`, `karar`, `magaza`, `gecerli: true` dolu;
Telegram'a ✅ bildirim; Sheets'e satır.

```powershell
# F2 — eksik alan
curl.exe -X POST "http://localhost:5678/webhook/stok-karar" `
  -H "Content-Type: application/json" -d '{{\\"tarih\\":\\"2015-07-15\\"}}'
```
Beklenen: ⚠️ bildirim, yanıtta `hata: "Eksik alan: karar, aciliyet"`.

```powershell
# F3 — bilinmeyen karar
curl.exe -X POST "http://localhost:5678/webhook/stok-karar" `
  -H "Content-Type: application/json" `
  -d '{{\\"tarih\\":\\"2015-07-15\\",\\"karar\\":\\"ucus_rezervasyonu\\",\\"aciliyet\\":\\"dusuk\\"}}'
```
Beklenen: "Bilinmeyen karar degeri", ⚠️ bildirim.

---

## G. ZAMANLAYICI

`Her Sabah 09:00` → **Execute step** (saat dilimi Europe/Istanbul).

| # | Beklenen |
|---|---|
| G1 | `Gunluk Rapor Al` FastAPI `/gunluk-rapor` çeker |
| G2 | Sheets "Ogrenme Gecmisi"ne satır |
| G3 | Alarm varsa sesli drift alarmı, yoksa sessiz özet |

---

## H. FASTAPI ENDPOINT'LERİ

| # | Adres | Beklenen |
|---|---|---|
| H1 | `http://localhost:8000/onay-ozet` | Onay/red istatistikleri |
| H2 | `http://localhost:8000/ogrenme-raporu` | Eşik önerileri |
| H3 | `http://localhost:8000/trend?yenile=true` | Haftalık drift + karar trendi (gerçek satışla) |
| H4 | `http://localhost:8000/gunluk-rapor` | Zamanlayıcının paketi |
| H5 | `http://localhost:8000/docs` | Otomatik dokümantasyon |

---

## I. DAYANIKLILIK

| # | Test | Beklenen |
|---|---|---|
| I1 | FastAPI'yi kapat, soru sor | "⚠️ Asistan servisine şu an ulaşılamıyor" mesajı |
| I2 | FastAPI'yi aç, aynı soru | Normal cevap |
| I3 | 500+ karakterlik soru | 120 sn içinde cevap |
| I4 | Arka arkaya 3 soru | Sırayla, karışmadan |

---

## KAPSAM KONTROLÜ

| Node grubu | Kapsayan test |
|---|---|
| Telegram Trigger, Giris Tipi?, FastAPI Soru, Sohbet Kaydi | A |
| Karar Uretildi mi?, Telegram Duz Cevap | A1–A11, I1 |
| Karar Ayikla, Karar Dogrula, Gecerli mi?, Karar Defterine Yaz | A12, B |
| Karar Yonlendir | B1–B8 |
| Siparis Onay Gerekli mi?, Siparis Eskalasyon, Otomatik Siparis | B1, B2 |
| Indirim Onay Gerekli mi?, Indirim Onaya, Otomatik Indirim* | B3 |
| Bekle Log, Telegram Kaynakli mi?, Telegram Bekle Bildirimi | B4, B5 |
| Insan Onayi | B6–B8 |
| Telegram Onay/Aksiyon Bildirimi | B1–B3, B6–B8 |
| Callback Ayikla, Butona Yanit Ver, Callback Yonlendir | C |
| Telegram Sebep Sor, Red Kaydet, Sheets Red Guncelle, Telegram Red Sonuc | C2–C4 |
| Onay Kaydet, Sheets Onay Guncelle, Telegram Onay Sonuc | C1 |
| Webhook, Hata Yaniti, Bilinmeyen Karar, Yanit, Webhook Kosusu mu? | F |
| Telegram Hata Bildirimi | F2, F3 |
| Her Sabah 09:00, Gunluk Rapor Al, Alarm Var mi?, Ogrenme Gecmisine Yaz | G |
| Telegram Drift Alarmi / Sessiz Ozet | G3 |

\\* `indirim_uygula` Madde 6.1 gereği her zaman onaya gider; `Otomatik Indirim`
dalı yalnız webhook ile `onay_gerekli_mi:false` bir indirim payload'ı
gönderilerek çalıştırılabilir.

---

## SONUÇ KAYDI

| Grup | Geçen | Kalan | Not |
|---|---|---|---|
| A (araç seçimi) | /12 | | |
| B (karar yolu) | /{len(secilen)} | | |
| C (onay döngüsü) | /5 | | |
| D (sınır/hata) | /8 | | |
| E (çok adımlı) | /6 | | |
| F (webhook) | /3 | | |
| G (zamanlayıcı) | /3 | | |
| H (endpoint) | /5 | | |
| I (dayanıklılık) | /4 | | |
"""

Path("TELEGRAM_TEST_SENARYOLARI.md").write_text(MD, encoding="utf-8")
print("TELEGRAM_TEST_SENARYOLARI.md yazildi. Secilen B ciftleri:")
for k, s in secilen.items():
    print(f"  {k}: magaza {int(s.magaza):4d}  {s.tarih.date()}  {s.karar}  oner='{s.oner}'")
