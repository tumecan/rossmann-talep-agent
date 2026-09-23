# Rossmann Taze Kategori — Talep Tahmini → LLM Agent → Otomasyon

Bozulabilir üründe stok kararı iki yönlü bir hatadır: az sipariş verirsen satış
kaçar, çok verirsen mal çöpe gider. Bu proje 1115 Rossmann mağazası için günlük
talebi **olasılıksal** olarak tahmin eder, tahmini stok ve raf ömrüyle
birleştirerek **yapısal bir karar** üretir ve kararı insan onaylı, kendi
performansını izleyen bir otomasyon hattına bağlar. Sistem, **gerçek satışla
geriye dönük test** edilmiştir.

> Miuul *Deep Learning to AI Agent* bootcamp bitirme projesi — tema: Talep
> Tahmini (Forecasting → Agent → Tedarik).

| Katman | Ne yapar | Ana sonuç |
|---|---|---|
| **DL modeli** | Global GRU seq2seq, p10/p50/p90, 3 seed | test sMAPE **8.34** (XGBoost 9.07), kalibre kapsama **%77.4** |
| **LLM agent** | 12 araç, deterministik kural motoru, sabit JSON, RAG | routing **59/59**, yasaklı araç ihlali **0**, **51** birim testi |
| **n8n** | Doğrulama → routing → insan onayı → kayıt → günlük drift raporu | 45 node, Telegram + Google Sheets, uçtan uca canlı |
| **Gerçek satışla backtest** | Siparişler tahminle, tüketim gerçek talepten | Ulaşılabilir değerin **%92.3**'ü; model hatasının bedeli **628.793 EUR** |

---

## İçindekiler

1. [Problem ve veri](#1-problem-ve-veri)
2. [Mimari](#2-mimari)
3. [Katman 1 — Tahmin modeli](#3-katman-1--tahmin-modeli)
4. [Stok projeksiyonu ve sipariş politikası](#4-stok-projeksiyonu-ve-sipariş-politikası)
5. [Katman 2 — LLM agent](#5-katman-2--llm-agent)
6. [Katman 3 — n8n otomasyonu](#6-katman-3--n8n-otomasyonu)
7. [Değerlendirme — gerçek satışla geriye dönük test](#7-değerlendirme--gerçek-satışla-geriye-dönük-test)
8. [Karar kuralının evrimi — bulunan hatalar](#8-karar-kuralının-evrimi--bulunan-hatalar)
9. [Öğrenme döngüsü ve trend analizi](#9-öğrenme-döngüsü-ve-trend-analizi)
10. [Testler](#10-testler)
11. [Kurulum ve çalıştırma](#11-kurulum-ve-çalıştırma)
12. [Varsayımlar ve bilinen sınırlar](#12-varsayımlar-ve-bilinen-sınırlar)
13. [Bir şirkette ilk neyi değiştirirdim](#13-bir-şirkette-ilk-neyi-değiştirirdim)
14. [Repo yapısı](#14-repo-yapısı)

---

## 1. Problem ve veri

Taze kategori raf ömrü **3 gün**. Her mağaza-gün için üç soru var:

| Soru | Yanlış cevabın bedeli |
|---|---|
| Sipariş verilsin mi, ne kadar? | Az → kaçan kâr. Çok → fire |
| Elde eriyemeyecek stok var mı? | İndirim geciktikçe mal değersizleşir |
| Bu karar otomatik uygulanabilir mi? | Yanlış otomatik karar, denetimsiz zarar |

Üçüncü soru projenin merkezinde. Bir tahmin modeli sayı üretir; bir **karar
sistemi** o sayının ne zaman güvenilir olduğunu da bilmek zorundadır.

**Veri:** [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
— 1115 mağaza, 2013-01-01 – 2015-07-31, günlük ciro (1.017.209 satır). Veri
repoda yoktur; Kaggle'dan indirilip `Rossman/data/` altına konur.

- **Bölme tarihe göre:** validasyon hedefi 05-25 – 07-03, test hedefi
  **07-04 – 07-31** (28 gün). Zaman serisinde rastgele bölme geleceği geçmişe
  sızdırır.
- **Karar günü 2015-07-03.** Sistem "bugün"ü bu tarih kabul eder; pencere
  dışındaki tarihler için "veri yok" der, uydurmaz.
- `Customers` kolonu kullanılmaz: tahmin anında bilinmez (sızıntı).
- Ciro → taze kategori adedi: `adet = ciro × 0.20 / 8.0 EUR`.
- Birim fiyat 8 EUR, marj %25 → birim kâr (Cu) 2 EUR, birim maliyet (Co) 6 EUR.

---

## 2. Mimari

```
                      ┌──────────────────────────────────────────┐
  Kaggle train.csv ──►│ KATMAN 1 — Global GRU seq2seq (Colab)    │
                      │ 1115 mağaza × 28 gün × {p10, p50, p90}    │
                      └───────────────────┬──────────────────────┘
                                          ▼
                      ┌──────────────────────────────────────────┐
                      │ Stok projeksiyonu (FIFO, raf ömrü 3 gün)  │
                      │ newsvendor + 4 tedarikçi sözleşmesi       │
                      └───────────────────┬──────────────────────┘
                                          ▼
  Telegram ──► n8n ──► FastAPI ──► ┌──────────────────────────────┐
                                   │ KATMAN 2 — LangChain agent    │
                                   │ LLM: araç seçimi + yorum      │
                                   │ karar_tool: tüm sayılar/kurallar│
                                   │ RAG: sözleşme + politika      │
                                   └──────────────┬───────────────┘
                                                  ▼
                      ┌──────────────────────────────────────────┐
                      │ KATMAN 3 — n8n                            │
                      │ doğrula → yönlendir → onay → Sheets       │
                      │ her sabah 09:00 drift raporu              │
                      └──────────────────────────────────────────┘
```

### Temel ilke — LLM sayı üretmez

Deterministik olarak hesaplanabilen hiçbir değer LLM'e bırakılmaz. LLM yalnız
**hangi aracı hangi parametreyle çağıracağını** seçer ve sonucu yorumlar; tüm
sayılar ve karar kuralları `karar_tool.py` / `tools_toplu.py` içinde hesaplanır.
Agent parametreyi yanlış verebilir (görülür, reddedilir), ama **sayı uyduramaz**.

---

## 3. Katman 1 — Tahmin modeli

### Model

| | |
|---|---|
| Mimari | Seq2seq: GRU kodlayıcı (128) + GRU çözücü (128) + mağaza embedding (24) |
| Düzenlileştirme | Dropout 0.3, L2 1e-5 |
| Parametre | 310.155 |
| Girdi penceresi | 56 gün (5 geçmiş kanalı: satış, açık, promo, okul, tatil) |
| Gelecek kanalları | 14 (promo, tatil, takvim sin/cos, geçen yıl aynı gün — 364 gün gecikme) |
| Ufuk | 28 gün, tek geçişte (recursive değil) |
| Kayıp | Pinball (kuantil) — p10 / p50 / p90 |
| Eğitim | Adam lr 3e-4, batch 256, EarlyStopping (patience 20, en iyi ağırlık), ReduceLROnPlateau |
| Topluluk | 3 seed (42, 1337, 2024); tekil seed gürültü bandı **0.23** sMAPE puanı |
| Kalibrasyon | Validasyonda seçilen tek katsayı, k = 1.051 |

![Eğitim kaybı](figs/global_model_loss.png)

**Neden global:** 1115 mağaza için tek model, ortak deseni (hafta günü,
promosyon, tatil) paylaşır; mağaza farkını embedding taşır.
**Neden kuantil:** "Kötümser talepte stok tükenir mi" sorusu p90 olmadan
cevaplanamaz. **Neden seq2seq:** recursive çok adımlı tahminde hata birikir
(aşağıda tek mağaza deneyi).

### Baseline karşılaştırması

**A) Global modeller — 1115 mağaza, aynı test penceresi**

| Model | sMAPE | wMAPE | Bias | Kapsama p10–p90 |
|---|---|---|---|---|
| XGBoost global (kuantil) | 9.07 | 8.88 | +0.58 | 75.9 |
| GRU — tekil seed | 8.37 – 8.60 | — | — | 72.4 – 74.9 |
| **GRU — 3 seed topluluğu** | **8.34** | **8.12** | **−0.76** | 75.0 → **77.4** (kalibre) |

Mağaza bazlı hata korelasyonu XGBoost ile GRU arasında **0.87**: iki farklı
model aynı mağazalarda zorlanıyor, zorluk modelden çok veriden geliyor.

**B) Keşif aşaması — tek mağaza (mağaza 1)**

| Model | sMAPE | R² | Not |
|---|---|---|---|
| Naive-1 (dünkü satış) | 10.34 | 0.15 | |
| Naive-7 (geçen hafta aynı gün) | 26.73 | −2.31 | |
| Prophet | 7.22 | 0.63 | |
| SimpleRNN / LSTM / GRU | 8.71 / 9.15 / 10.06 | 0.49 / 0.40 / 0.28 | |
| XGBoost (42 gün, recursive) | 6.14 | 0.76 | |
| LSTM (42 gün, recursive) | 20.01 | −1.33 | Hata birikimi |

Tek mağazada Prophet ve XGBoost tekil RNN'lerden iyi. İki ders nihai tasarımı
belirledi: tek mağaza verisi derin model için yetmez → **global model**;
recursive tahminde hata birikir → **ufku tek geçişte üreten seq2seq**.

### Validasyon ve test

| Metrik | Val | Test |
|---|---|---|
| sMAPE | 7.59 | **8.34** |
| wMAPE | 7.50 | 8.12 |
| Kapsama p10–p90 | 77.85 | 75.0 (kalibre **77.4**) |
| Bias | −1.26 | −0.76 |

Val–test farkı küçük (0.75 puan): belirgin aşırı öğrenme yok. **sMAPE** ana
metrik (kapalı günde gerçek satış 0, MAPE sıfıra böler); **wMAPE** hacim
ağırlıklı; **bias** yönü ölçer (sistematik yüksek tahmin = fire, düşük =
stoksuzluk); **kapsama + Kupiec** aralığın kalibrasyonunu sınar.

**Kupiec POF testi:** LR 109.98 → reddedildi, bant sistematik olarak dar.
Ancak 26.845 gözlemde test çok güçlüdür (2.6 puanlık açık bile "anlamlı"
çıkar) ve gözlemler bağımsız değildir (aynı gün mağazalar arası korelasyon).
Farkın büyüklüğü LR'den değil kapsama açığından okunmalı.

### Hata analizi — model nerede zorlanıyor

| Segment | n | sMAPE | Bias | Kapsama | Yorum |
|---|---|---|---|---|---|
| **Pazar** | 129 | **18.41** | **−10.47** | 73.6 | En zayıf nokta: az gözlem, sistematik düşük tahmin |
| **Cumartesi** | 4.453 | 10.27 | **+1.95** | 72.6 | Fazla tahmin, p10 altı %17.4 → fire (bkz. bölüm 7) |
| Salı / Çarşamba | 4.453 | 7.29 / 7.44 | ~0 | 81 | En iyi günler |
| Promosyon günü | 11.128 | 7.93 | +0.12 | 77.0 | |
| Mağaza tipi b | 476 | 8.53 | −2.37 | 85.5 | Bant geniş |
| Talep Q1 (küçük) | 5.369 | 9.57 | −0.59 | 76.1 | Küçük mağazada hata yüksek |
| Talep Q5 (büyük) | 5.368 | 7.69 | −0.66 | 78.3 | |

**Hipotez testi — kapalı günler.** İlk hipotez, kapalı günlerin metrikleri
bozduğuydu. Test edildi, büyük ölçüde çürütüldü: model kapalı günde 0 tahmin
ediyor, sMAPE değişmiyor (8.34). Yalnız 3 mağaza (292, 876, 909) %35'ten fazla
kapalı; sMAPE ile kapalı gün oranı korelasyonu (0.478) bu 3 uç değerden geliyor.

### Belirsizlik kapısı — sızıntı düzeltmesi ve binom testi

Kapı, model bir mağazada zayıfsa kararı otomatik uygulamaz, insana devreder.

**Sızıntı.** İlk sürümde kapı test dönemi metrikleriyle besleniyordu; bunlar
07-04 – 07-31'in gerçek satışlarından hesaplanır ve karar günü 07-03'te
bilinemez. Kapı artık yalnız **validasyon dönemi** metrikleriyle çalışır.
Regresyon testi: test metrikleri bozulduğunda hiçbir karar değişmemeli (eski
kod 40/40 kararı değiştiriyordu, yenisi 0).

**Gürültü.** Mağaza kapsaması ~28 gözleme dayanır; kusursuz kalibre bir mağaza
bile %9 olasılıkla %70'in altına düşer. Sabit eşik yerine tek yönlü **binom
testi**: kapsama nominal %80'in altında **ve** fark şansla açıklanamıyorsa
(p < 0.05) kapı kapanır. Ek koşul: val sMAPE > %20.

**Val ile kur, test ile sına:**

| | Kapı kapalı | Kapı açık |
|---|---|---|
| Mağaza | 62 | 1.053 |
| Test sMAPE | %10.74 | %8.28 |
| Test kapsama | %65.9 | %77.9 |

Testte en kötü %10'luk dilimden (112 mağaza) 17'si yakalanıyor; şansla
beklenen ~6. 1115 mağazaya p < 0.05 uygulandığında şansla ~56 yanlış
pozitif beklenir; kapı bir çıkarım değil **triyaj** aracıdır ve örneklem dışı
kanıt ayrışmayı doğrular.

### Kalibrasyon denemesi (CQR) — işe yaramadı

`kuantil_kalibrasyon.py`: ilk 14 gün kalibrasyon, son 14 gün ölçüm.

| | Kapsama | p10 altı | p90 üstü | Kupiec LR |
|---|---|---|---|---|
| Ham bant | %75.28 | %8.85 | %15.87 | 177.39 |
| Kalibre bant | %75.80 | %8.61 | %15.59 | 141.15 |

Tek global katsayı yetmedi: bant **asimetrik** ve **zamanla kayıyor**
(bölüm 9). Bu sonuç raporlanıyor, gizlenmiyor.

---

## 4. Stok projeksiyonu ve sipariş politikası

`projeksiyon_uret.py` — 1115 mağaza, 28 gün, FIFO:
teslimat gelir → satış (en eski partiden) → gün sonu 3 günlük mal fire olur.

- **İki mod:** `siparis_yok` (hiç sipariş verilmezse) ve `politika`
  (newsvendor hedefi + tedarikçi kısıtları).
- **Üç talep senaryosu:** p10 / p50 / p90.
- **Newsvendor:** kritik oran Cu / (Cu + Co) = 0.25. Fire, kaçan satıştan üç
  kat pahalı olduğu için politika bilerek medyanın altında sipariş verir.
- Sipariş, **varış gününün** penceresi için hesaplanır: yoldaki sipariş ve
  varışa kadarki ara talep düşülür.

| Tedarikçi | Teslim | Min. sipariş (zincir, günlük) | Teslimat günleri |
|---|---|---|---|
| ExpressLog | 1 gün | 200 | Pzt–Cmt |
| Schnellware | 2 gün | 500 | Pzt, Çar, Cum |
| Rheinland | 3 gün | 1000 | Sal, Per |
| Nordmann | 5 gün | 1500 | Pzt–Cum |

Minimum sipariş **zincir düzeyinde** uygulanır: mağaza başına uygulamak,
Nordmann'ın 1500 birimini ortalama mağazanın ~9 günlük talebine eşitler; 3 gün
raf ömrüyle yapısal olarak imkânsızdır. Tedarikçi seçim kuralları RAG
belgelerindedir.

---

## 5. Katman 2 — LLM agent

LangChain agent (`gpt-5.5`), FastAPI ile sarmalanmış. Sistem promptu dosyada
ve versiyonlu: `prompts/asistan_promptu_v2.txt` (v1 geçmiş için tutulur).

### Araçlar (12)

| Araç | Ne döndürür |
|---|---|
| `tukenme_riski` | Sipariş verilmezse stoksuz kalacak mağazalar (en kötü durum) |
| `sevkiyat_plani_getir` | Belirli günün sevkiyatı |
| `en_cok_ihtiyac_duyan` | Öncelik sıralaması |
| `segment_ozeti` | Mağaza tipi / ürün yelpazesi kırılımı |
| `model_performansi` | Tek mağaza için sMAPE, kapsama, zincir kıyası |
| `raf_omru_durumu` | Fire riski, kalan raf ömrü |
| `dagitim_plani` | Kısıtlı stoğun ihtiyaç oranında dağıtımı |
| `magaza_karsilastir` | İki mağaza yan yana |
| `promo_senaryosu` | Kampanya etkisi |
| `tedarikci_bilgisi_ara` | **RAG** — sözleşme ve politika belgeleri (BM25 + vektör, RRF) |
| `trend_ozeti` | Zaman içi model ve karar trendi, drift alarmları |
| `karar_uret` | **Yapısal karar** — tek mağaza-gün |

On bir araç bilgi döndürür; yalnız `karar_uret` bağlayıcı karar üretir ve
karar defterine yazar. Bu ayrım routing değerlendirmesinde **yasaklı araç**
kontrolüyle sınanır.

### Kural motoru (`karar_tool.py`)

```
1. KAPALI GÜN      → insana_sor                                (her şeyden önce)
2. ÇELİŞKİ         → beklenen fire eşiği aşıldı VE politika bugün sipariş
                     veriyor → insana_sor (iki öneri de korunur)
3. FİRE            → beklenen fire / günlük talep > 0.40 → indirim_uygula
4. SİPARİŞ         → politika motoru bugün sipariş veriyor → siparis_ver
                     (miktar, tedarikçi, varış tarihi motorun hesabı)
5. Aksi halde      → bekle  (p90'da bugün eksik kalacaksa UYARI + aciliyet)
6. BELİRSİZLİK     → model zayıf ve karar sipariş/indirim → insana_sor
7. ONAY            → net etki > 3000 EUR / Madde 6.1 fire / kapalı / çelişki /
                     model / mağaza kaydı  (bekle asla onay istemez)
```

**Beklenen fire (Swanson kuralı):** `0.3 × fire(p10) + 0.4 × fire(p50) +
0.3 × fire(p90)`. Plan (p50) tahmin hatasından doğacak fireyi göremez; üç
kuantilden beklenen değer modelin belirsizlik bandını karara taşır.
Ağırlıklar standarttır, veriye bakılarak seçilmedi.

| Eşik | Değer | Kaynak |
|---|---|---|
| Fire oranı | 0.40 | Satınalma politikası Madde 6.1 |
| Onay tutarı | 3000 EUR | Politika Madde 6 |
| Belirsizlik — val sMAPE | %20 | Zincir ortalamasının belirgin üstü |
| Belirsizlik — val kapsama | binom p < 0.05 | Nominal %80'in anlamlı altı |
| İndirim elastikiyeti | 1.5 | **Varsayım** (bkz. sınırlar) |

### Çıktı şeması

`prompts/karar_semasi.json`. Karar seti dört değerle sınırlı. Her çıktı
**kararı üreten eşikleri** ve **varış tarihini** taşır; model zayıf olduğu
için devredilen bir kararda önerilen aksiyon `onerilen_aksiyon` alanında
korunur (defterde "insana_sor + %50 indirim" gibi çelişkili satır oluşmaz).

### "Bilmiyorum" diyebilmek

| Durum | Davranış |
|---|---|
| Pencere dışı tarih | Araç çağrılmadan reddedilir, geçerli aralık söylenir |
| Olmayan mağaza | Araç yapısal hata döndürür, uydurmaz |
| Model zayıf | Karar `insana_sor`'a devredilir |
| Kapsam dışı soru ("hava nasıl") | Kibarca reddedilir, araç çağrılmaz |
| Araç çıktısında talimat (prompt injection) | Veri sayılır, uygulanmaz |

### Routing değerlendirmesi

`routing_eval.py` — 59 soru: 12 araç için 4–6 soru, 4 kapsam dışı red,
yasaklı-araç kontrolleri. Son koşu **59/59**, yasaklı araç ihlali **0**.
Önceki koşular 58/59; LLM olasılıksal olduğu için sonuç aralık olarak izlenir.

---

## 6. Katman 3 — n8n otomasyonu

Workflow: [`n8n/Rossmann_Stok_Karar_Akisi.json`](n8n/Rossmann_Stok_Karar_Akisi.json)
— **45 node**, üç tetikleyici: Telegram, webhook (toplu), zamanlayıcı (09:00).

![n8n workflow](figs/n8n_workflow.png)

```
Telegram Trigger ─► Giriş Tipi? ─┬─ mesaj ─► FastAPI Soru ─┬─► Sohbet Kaydı (Sheets)
                                 │                          └─► Karar üretildi mi?
                                 │                                 │ hayır ─► Telegram düz cevap
                                 │                                 │ evet
Webhook (toplu) ─────────────────┼──────────────► Karar Doğrula ◄─┘
                                 │                     │
                                 │                Geçerli mi? ── hayır ─► Hata ─► Telegram ⚠️
                                 │                     │ evet
                                 │        ┌────────────┴───────────┐
                                 │   Karar Defteri          Karar Yönlendir (5 dal)
                                 │   (Sheets, yan dal)   sipariş · indirim · bekle · insana · bilinmeyen
                                 │                              │
                                 │                  onay gerekli? → Telegram 🔴 + Onayla/Reddet
                                 │                  otomatik      → Telegram ✅
                                 │                  bekle         → Telegram ℹ️ (yalnız Telegram sorgusunda)
                                 └─ buton ─► Callback Ayıkla ─► onay / sebep sor / red
                                                  └─► FastAPI /onay ─► Sheets güncelle ─► Telegram sonuç
Her Sabah 09:00 ─► /gunluk-rapor ─► Alarm var mı? ─► sesli alarm / sessiz özet
                                  └─► Sheets "Ogrenme Gecmisi"
```

### Tasarım kararları

- **Onay kuralı tek yerde.** `onay_gerekli_mi` yalnız `karar_tool.py`'de
  üretilir; n8n yalnız doğrular ve yönlendirir.
- **Sheets yan dalda.** Google Sheets node'u yazdığı satırı döndürür, girdiyi
  değil; ana hatta olsaydı sonraki node'lar karar nesnesini kaybederdi.
- **İki ayrı kayıt.** `Sayfa1` (karar defteri, yalnız kararlar) ve
  `Sohbet Kaydi` (%100 trafik). İş kaydı ile gözlemlenebilirlik kaydı ayrıdır.
- **Tekilleştirme.** Karar defteri `karar_id` (`<mağaza>_<tarih>`) üzerinden
  `appendOrUpdate` ile yazılır; aynı soru iki kez sorulursa satır çoğalmaz.
- **Red sebebi sabit küme:** `model`, `parametre`, `zamanlama`, `diger`.
  Serbest metin olsaydı öğrenme döngüsü sebepleri gruplayamazdı.
- **`bekle` bilgi mesajı** yalnız Telegram'dan sorulduğunda gider; toplu
  koşuda bildirim yorgunluğu yaratmaz.
- **Hata yakalama.** Geçersiz payload hata dalına gider (sessizce geçmez);
  Switch'te `bilinmeyen` fallback'i; FastAPI kapalıysa kullanıcıya
  "servise ulaşılamıyor" mesajı; zamanlayıcı HTTP çağrısı 3 kez dener.
- **Kimlik doğrulama.** Sheets için OAuth yerine service account (test
  modundaki OAuth refresh token'ı 7 günde düşer).

| Endpoint | Ne yapar |
|---|---|
| `POST /soru` | Soru sor, gerekirse yapısal karar üret |
| `POST /onay` | Onay / red kaydı (n8n çağırır) |
| `GET /onay-ozet` | Onay istatistikleri |
| `GET /ogrenme-raporu` | Eşik önerileri |
| `GET /trend` | Haftalık drift ve karar trendi |
| `GET /gunluk-rapor` | Zamanlayıcının çağırdığı paket |

---

## 7. Değerlendirme — gerçek satışla geriye dönük test

Routing "doğru aracı seçti mi" sorusunu ölçer. Bu bölüm başka bir soruyu
ölçer: **verilen kararlar gerçekte ne kazandırdı, ne kaybettirdi?**

### Yöntem (`gercek_backtest.py`)

`train.csv` 2015-07-31'e kadar gerçek satış içerir; test penceresi bunun
içindedir. Politika zinciri aynı motorla yeniden koşturulur:
**siparişler model tahminiyle** (karar anında bilinen), **tüketim gerçek
talepten** (sonradan gerçekleşen). Referans ne agent kurallarına ne modele
dayanır. Kayıp = stoksuz × Cu + fire × Co.

Tahmin / gerçek toplam talep: 4.623.389 / 4.658.907 adet (bias −%0.76).

### Politika düzeyi — 1115 mağaza × 28 gün

| Senaryo | Doluluk | Kaçan kâr (EUR) | Fire maliyeti (EUR) | Toplam kayıp (EUR) |
|---|---|---|---|---|
| Hiç sipariş yok | %4.5 | 8.899.604 | 367.692 | 9.267.296 |
| **Model (sistem)** | **%87.2** | 1.188.996 | 503.677 | **1.692.674** |
| Kusursuz bilgi (üst sınır) | %93.0 | 653.660 | 410.221 | 1.063.881 |
| Plan (sistemin beklentisi) | %89.1 | 1.008.520 | 371.588 | 1.380.108 |

- Sistem **ulaşılabilir değerin %92.3'ünü** yakalıyor.
- **Model hatasının bedeli 628.793 EUR** (model − kusursuz bilgi).
- **Plan %23 iyimser:** 1,38M EUR kayıp bekleniyordu, gerçekleşen 1,69M —
  dar bandın (Kupiec reddi) parasal karşılığı.
- Kusursuz bilgide bile doluluk %93: kritik oran 0.25, teslimat takvimi ve
  zincir minimumu bağlayıcı.

**Model hatası nerede?**

| Hafta | Model hatası (EUR) | 100 adet talep başına |
|---|---|---|
| H1 (07-04 – 07-10) | 79.474 | 8 |
| H2 (07-11 – 07-17) | 128.397 | 10 |
| H3 (07-18 – 07-24) | 193.970 | 19 |
| H4 (07-25 – 07-31) | 226.952 | 17 |

| Belirsizlik kapısı | Model hatası (EUR) | 100 adet başına |
|---|---|---|
| Açık | 592.529 | 13 |
| Kapalı (zayıf model) | 36.264 | 17 |

Birim hata maliyeti ikinci yarıda iki katına çıkıyor: bölüm 9'daki kapsama
drift'inin **euro karşılığı**. Kapı doğru yönde ayrıştırıyor (1,3 kat) ama
fark mütevazı.

### Gün bazlı kararlar — 26.845 açık mağaza-gün

| Metrik | Değer |
|---|---|
| Karar dağılımı | bekle %49.2 · sipariş %46.7 · insana_sor %3.9 · indirim %0.2 |
| Onay yükü | **%33.1** |
| Gerçekleşen fire (açık günler) | 654 gün / 341.616 EUR |
| Agent fire sinyali verdi | 325 gün / **267.032 EUR (%78)** |
| Kaçırılan fire | 329 gün / 74.584 EUR |
| Gereksiz indirim sinyali | 3 gün |
| Stok uyarısı — yakalama / isabet | %99.9 / %76.5 (taban oran %47) |

**%78 başlığı dikkatli okunmalı:**

| Hafta | Fire yakalama | Kaçırılan fire (EUR) |
|---|---|---|
| H1 | %66.7 | 35.477 |
| H2 | %7.7 | 13.338 |
| H3 | %5.0 | 3.546 |
| H4 | %22.5 | 22.223 |

Yakalanan firenin büyük kısmı ilk haftada, **başlangıç stoğundan** gelen ve
planda görünen firedir. Sonraki haftalarda fire çoğunlukla **tahmin
hatasından** doğar: model talebi fazla tahmin eder, politika fazla sipariş
verir, mal bozulur. En pahalı 10 kaçırmanın 8'i **Cumartesi** — model
Cumartesiyi fazla tahmin ediyor (bias +1.95, p10 altı %17.4). **Model
hatasının karar katmanına sızdığı somut nokta budur.** Beklenen fire sinyali
bunu kısmen yakalar; kalanı bandın dar olmasından gelir ve karar
katmanıyla kapatılamaz (bkz. bölüm 13).

**Kapalı gün firesi.** Politika düzeyindeki fire (503.677 EUR) ile açık gün
firesi (341.616 EUR) arasındaki fark kapalı günlerden gelir (çoğunlukla
Pazar): mağaza kapalıyken raf ömrü dolan mal.

### Fire eşiği duyarlılığı

| Fire eşiği | İndirim sinyali | Yakalanan EUR | Kaçırılan EUR | Gereksiz gün |
|---|---|---|---|---|
| 0.20 | 521 | 313.054 | 28.562 | 40 |
| 0.30 | 395 | 286.272 | 55.344 | 7 |
| **0.40** (politika 6.1) | 328 | 267.032 | 74.584 | 3 |
| 0.60 | 233 | 217.413 | 124.204 | 1 |
| 0.80 | 152 | 156.858 | 184.758 | 1 |

**Eşik 0.40'ta bırakıldı.** Tablo test döneminden üretildi; eşiği ona bakarak
seçmek, belirsizlik kapısında düzeltilen test sızıntısının aynısı olurdu.
Doğru adım, eşiği bağımsız bir dönemde doğrulayıp Madde 6.1'i ona göre
revize etmektir.

---

## 8. Karar kuralının evrimi — bulunan hatalar

Gerçek hatalardan doğan düzeltmeler, hayal edilmiş senaryolardan değerlidir.

| Konu | Bulgu | Düzeltme |
|---|---|---|
| Belirsizlik kapısı | Kapı **test** metrikleriyle besleniyordu (sızıntı); 28 gözlemde sabit %70 eşiği gürültüye tetikleniyordu | Val metrikleri + binom testi; sızıntı regresyon testi |
| Sipariş kuralı | Kural `siparis_yok` modunun p90 stoksuzluğuna bağlıydı. Bu mod hiç sipariş vermez; stok 3. günde biter | Sipariş kararı politika motorunun o günkü siparişine bağlandı |
| Fire sinyali | Sinyal yalnız plana (p50) bakıyordu; tahmin hatasından doğan fire görünmüyordu | Beklenen fire (Swanson) |

**Sipariş kuralı dejenerasyonu (`teshis_siparis.py`):**

| Ufuk günü | 1 | 2 | 3 | 4–28 |
|---|---|---|---|---|
| Kuralın tetiklenme oranı | %45.3 | %78.8 | %70.8 | **%100** |
| Agent `siparis_ver` oranı | %49.3 | %78.8 | %68.5 | **~%94.5** |

4. günden sonra kural, kapının devrettikleri dışında her açık mağazaya
"sipariş ver" diyordu. Düzeltmeden sonra sipariş oranı haftalar boyunca %41–50'de stabil,
onay yükü %65.1 → %33.1. Eski kural döneminde raporlanan simülasyon tabanlı isabet
oranları (modeli yine modelle sınıyordu) geçersiz sayıldı; bölüm 7 onların
yerini alır.

| | Plan (p50) sinyali | Beklenen fire sinyali |
|---|---|---|
| Yakalanan fire | 263.054 EUR | 267.032 EUR |
| Kaçırılan fire | 78.563 EUR | 74.584 EUR |
| Gereksiz indirim | 2 gün | 3 gün |
| Haftalık yakalama H2 / H3 / H4 | %2.6 / %0.0 / %15.8 | %7.7 / %5.0 / %22.5 |

---

## 9. Öğrenme döngüsü ve trend analizi

### İnsan onayı → öğrenme

Her onay ve red, n8n'den FastAPI `/onay` üzerinden `onay_kayitlari.jsonl`
dosyasına kararın üretim parametreleriyle birlikte yazılır.
`ogrenme_dongusu.py` karar türüne göre onay oranı, red sebebi dağılımı ve
**eşik önerisi** üretir. Minimum 5 gözlemin altında öneri "veri yetersiz"
diye işaretlenir.

**Sistem eşikleri kendisi değiştirmez.** Öneri üretilir, insan uygular,
etkisi `gercek_backtest.py` ile ölçülür. Az veriyle eşik değiştirmek gürültüyü
sinyal saymaktır.

### Trend ve drift analizi (`trend_analizi.py`)

Tek sayıya indirgenmiş metrikler "model zamanla kötüleşiyor mu" sorusunu
cevaplayamaz. Hafta bazlı model, karar ve insan trendi; son hafta önceki
haftaların ortalamasından eşik kadar saparsa alarm.

| Hafta | sMAPE | Bias | Kapsama | p10 altı | p90 üstü | Kupiec LR |
|---|---|---|---|---|---|---|
| H1 (07-04 – 07-10) | 8.60 | +0.10 | **80.3** | 12.0 | 7.7 | **0.39** ✓ |
| H2 (07-11 – 07-17) | 7.42 | +2.38 | 78.7 | 16.7 | 4.6 | 7.33 |
| H3 (07-18 – 07-24) | 8.31 | −4.31 | 76.8 | 5.2 | 18.0 | 42.61 |
| H4 (07-25 – 07-31) | 9.03 | −1.83 | **73.8** | 12.5 | 13.7 | **150.48** |

| Hafta | Sipariş | Bekle | İnsana | Onay yükü | Fire yakalama | Model hatası (EUR/100 adet) |
|---|---|---|---|---|---|---|
| H1 | %45.5 | %47.4 | %6.4 | %35.7 | %66.7 | 7.72 |
| H2 | %50.1 | %46.7 | %3.2 | %34.1 | %7.7 | 9.91 |
| H3 | %50.1 | %46.7 | %3.2 | %34.1 | %5.0 | 19.47 |
| H4 | %41.2 | %56.0 | %2.8 | %28.6 | %22.5 | 16.96 |

![Trend ve drift](figs/trend_grafik.png)

**Bulgu: nokta tahmin sağlam, belirsizlik bandı eskiyor.**

- sMAPE'in günlük eğimi ≈ 0 (r = −0.10); nokta tahmin dört hafta stabil.
- Kapsama her hafta düşüyor: **80.3 → 78.7 → 76.8 → 73.8**. Kupiec H1'de
  geçiyor (LR 0.39), H4'te LR 150: model ilk hafta iyi kalibre, sonra kayıyor.
- Kaymanın yönü değişiyor (H2 yüksek tahmin, H3 düşük tahmin): tek global
  kalibrasyon katsayısının bunu düzeltememesinin sebebi bu.
- Model hatasının bedeli H1'den H3'e 2,5 katına çıkıyor — aynı drift'in
  euro karşılığı.
- Günlük sMAPE tepeleri (5, 19, 26 Temmuz) **Pazar** günleri.

sMAPE'e bakan biri modelin sağlıklı olduğunu söylerdi; trend analizi bandın
dört haftada %80'den %74'e aşındığını gösterdi ve sistem bunu kendi
alarmıyla yakaladı. n8n her sabah 09:00'da `/gunluk-rapor`'u çağırır: alarm
varsa bildirimli Telegram mesajı, yoksa sessiz özet; her durumda Sheets
"Ogrenme Gecmisi"ne bir satır. Agent'a da açık: *"Model zamanla kötüleşiyor
mu?"* → `trend_ozeti`.

---

## 10. Testler

| Test | Kapsam | Sonuç |
|---|---|---|
| `test_karar_tool.py` | Kural motoru: şema, kapalı gün, fire, kapı, sakin gün, hatalı girdi, tutarlılık, **sızıntı regresyonu**, **sipariş dejenerasyonu regresyonu**, çelişki, beklenen fire | **35/35** |
| `test_agent.py` | Toplu karar motoru (agent_v6): fire, politika kısıtı, sipariş mantığı, onay, aciliyet, şema savunması | **16/16** |
| `routing_eval.py` | 59 soru, araç seçimi + yasaklı araç | **59/59**, ihlal 0 |
| Telegram / n8n uçtan uca | `TELEGRAM_TEST_SENARYOLARI.md` — 45 node, 8 karar tipi, onay döngüsü, webhook, zamanlayıcı | geçti |

Birim testleri LLM çağırmaz: ücretsiz, saniyeler sürer, tekrarlanabilir.
Olasılıksal katmanı değil, ondan gelen her çıktıyı doğru karara çevirmesi
gereken **deterministik katman** test edilir. `test_karar_tool.py` ek kurulum
istemez; `test_agent.py` ise import zinciri nedeniyle RAG indeksini arar, yani
önce `python rag_index.py` çalıştırılmalıdır (bkz. bölüm 11). Test günleri koda gömülü
değildir; projeksiyon tablosundan taranarak bulunur. `senaryo_uret.py`
Telegram test çiftlerini gerçek karar çıktısından seçer.

---

## 11. Kurulum ve çalıştırma

### Ortam

```bash
conda create -n miuul python=3.11
conda activate miuul
pip install -r requirements.txt
cp .env.example .env        # anahtarları doldur
```

### Veri

Veri repoda yoktur (Kaggle yarışma verisi, yeniden dağıtılmaz). Kaggle
hesabınızın `kaggle.json` anahtarı `~/.kaggle/` altındayken:

```bash
pip install kaggle
kaggle competitions download -c rossmann-store-sales -p Rossman/data
unzip Rossman/data/rossmann-store-sales.zip -d Rossman/data
```

Alternatif olarak [yarışma sayfasından](https://www.kaggle.com/competitions/rossmann-store-sales/data)
elle indirip `train.csv` ve `store.csv` dosyalarını `Rossman/data/` altına koyun.

### Model (Google Colab, GPU)

| Adım | Notebook | Çıktı |
|---|---|---|
| 1 | `notebooks/colab_01_veri_hazirlik.ipynb` | Pencereler, kanallar, bölme |
| 2 | `notebooks/colab_02_global_model.ipynb` | 3 seed GRU |
| 3 | `notebooks/colab_02b_kalibrasyon.ipynb` | Kalibrasyon katsayısı |
| 4 | `notebooks/colab_03_tahmin_uret.ipynb` | `tahminler_tum_magazalar.parquet` |
| 5 | `notebooks/colab_04_xgboost_kiyas.ipynb` | Baseline kıyası |
| 6 | `notebooks/colab_05_embedding_gorsel.ipynb` | Mağaza embedding görseli |

Tahmin dosyası `Rossman/model/` altına konur.

### Yerel hazırlık

```bash
python projeksiyon_uret.py                 # 1115 mağaza FIFO projeksiyonu
python tedarikci_dokumanlari_olustur.py    # RAG kaynak belgeleri
python politika_dokumanlari_olustur.py
python rag_index.py                        # vektör indeksi
```

### Canlı sistem

```bash
ngrok http --domain=<NGROK_DOMAIN> 5678

docker run -d --name n8n -p 5678:5678 -v n8n_data:/home/node/.n8n \
  -e WEBHOOK_URL=https://<NGROK_DOMAIN> \
  -e N8N_HOST=<NGROK_DOMAIN> \
  -e GENERIC_TIMEZONE=Europe/Istanbul n8nio/n8n

python fastapi_v7.py                       # http://localhost:8000/docs
```

n8n'de `n8n/Rossmann_Stok_Karar_Akisi.json` import edilir,
`<TELEGRAM_CHAT_ID>` ve `<GOOGLE_SHEET_ID>` kendi değerlerinizle
değiştirilir, Telegram ve Google service account credential'ları seçilir,
workflow **Active** yapılır.

**Tuzaklar:** `WEBHOOK_URL` olmadan başlayan n8n Telegram'a `localhost`
bildirir, mesajlar sessizce düşer. Workflow değişince Active kapat → aç
(Telegram `allowed_updates` tazelenir).

### Değerlendirme

```bash
python test_karar_tool.py
python test_agent.py
python routing_eval.py                     # LLM çağırır, ücretli
python model_validasyon.py
python kuantil_kalibrasyon.py
python karar_degerlendirme.py --kapi       # belirsizlik kapısı doğrulaması
python gercek_backtest.py                  # gerçek satışla backtest (~15 dk)
python trend_analizi.py --grafik
python ogrenme_dongusu.py
python teshis_siparis.py                   # sipariş kuralı dejenerasyon teşhisi
python senaryo_uret.py                     # Telegram test senaryoları
```

---

## 12. Varsayımlar ve bilinen sınırlar

1. **Tek kategori, türetilmiş adet.** Rossmann'da ürün kırılımı yok; taze
   kategori ciro × 0.20 / 8 EUR ile türetildi.
2. **Başlangıç stoku ve tedarikçi sözleşmeleri sentetiktir.** RAG belgeleri
   proje için üretildi.
3. **İndirim elastikiyeti varsayımdır (1.5).** Doğrusal erime modelinde
   getiri s·e·o·(1−o)·fiyat'tır ve maksimumu e ≤ 2 için her zaman o = 0.5'tedir;
   %30'u yalnız "peş peşe iki gün %50" kısıtı seçtirir.
4. **Backtest indirimin etkisini simüle etmez;** indirim sinyali
   gerçekleşen fireyle kıyaslanır.
5. **Sipariş kararı gün bazında değil politika düzeyinde değerlendirilir;**
   bugünkü sipariş varış gününü kurtarır.
6. **Fire sinyali bandın genişliğiyle sınırlı.** Beklenen fire tahmin
   hatasını kısmen görür; bant dar olduğu için ilk haftadan sonra yakalama
   düşüktür.
7. **Karşı-olgusal zincir yok.** Agent'ın gün bazlı indirim kararlarının
   sonraki günlere etkisi simüle edilmez.
8. **`tipik_gunluk_talep_adet` ile günlük tahmin arasında ölçek farkı var.**
   Fire oranı politika Madde 6.1 ile tutarlı olmak için ilkini kullanır.
9. **Onaylanmış karar tekrar sorulursa** karar defteri `appendOrUpdate` ile
   satırı `onay_bekliyor`'a döndürür.
10. **İndirim geçmişi interaktif modda yazılmaz;** ardışık gün kısıtı yalnız
    toplu modda gerçek geçmişe dayanır.
11. **Sohbet geçmişi süreç belleğinde,** uç noktalarda kimlik doğrulama yok
    (yalnız localhost). Üretimde kalıcı oturum deposu ve API anahtarı gerekir.
12. **Fire sinyali test backtest'indeki bir bulguya dayanarak
    değiştirildi.** Eşik ayarlanmadı, tasarım gerekçesiyle değişti; iki
    sürümün sonucu da raporlandı (bölüm 8).

---

## 13. Bir şirkette ilk neyi değiştirirdim

1. **Ufka göre ve asimetrik bant kalibrasyonu.** Drift, backtest ve fire
   bulgusu aynı noktayı gösteriyor: nokta tahmin sağlam, bant zamanla daralıyor
   ve yön değiştiriyor. Tek global katsayı (CQR) yetmedi; her ufuk haftası ve
   her kuyruk için ayrı katsayı val origin'lerinden öğrenilmeli. Bu, model
   hatasının bedelini (628.793 EUR) doğrudan hedefler.
2. **Fire eşiğini bağımsız bir dönemde doğrulamak.** Duyarlılık tablosu 0.20–0.30
   aralığını işaret ediyor, ama test üzerinden seçilemez.
3. **Gerçek fiyat testi** ile indirim elastikiyetini ölçmek.
4. **Karşı-olgusal stok motoru** ile gün bazlı kararların zincir etkisini
   ölçmek.

---

## 14. Repo yapısı

```
├── README.md
├── TELEGRAM_TEST_SENARYOLARI.md
├── requirements.txt · .env.example · .gitignore
├── prompts/
│   ├── asistan_promptu_v1.txt · asistan_promptu_v2.txt
│   └── karar_semasi.json               # çıktı şeması + eşikler
├── notebooks/colab_0*.ipynb            # veri, model, kalibrasyon, kıyas, embedding
├── n8n/Rossmann_Stok_Karar_Akisi.json
├── figs/                               # README görselleri
├── Rossman/                            # üretilen çıktılar (ham veri ve indeks hariç)
│
├── tools_toplu.py                      # 10 analiz aracının gövdesi
├── karar_tool.py                       # kural motoru — karar_uret
├── agent_v7.py · fastapi_v7.py         # canlı asistan + HTTP servisi
├── agent_v2 … agent_v6.py              # sürüm zinciri (v7 import eder)
├── rag_index.py · rag_hybrid.py        # RAG (BM25 + vektör, RRF)
├── tedarikci_dokumanlari_olustur.py · politika_dokumanlari_olustur.py
├── projeksiyon_uret.py · stok_simulasyon.py
├── model_validasyon.py · kuantil_kalibrasyon.py
├── gercek_backtest.py · teshis_siparis.py · karar_degerlendirme.py
├── trend_analizi.py · ogrenme_dongusu.py
├── test_karar_tool.py · test_agent.py · routing_eval.py · senaryo_uret.py
├── n8n_gonder.py · n8n_json_temizle.py
└── (keşif) prophet_*.py · xgboost_*.py · lstm_*.py · rnn_family.py · model_karsilastirma.py
```

**Sürüm zinciri:** `agent_v2` (deterministik hesap) → `v3` (tahmin yükleme) →
`v4_langchain` (ihtiyaç hesabı) → `v5_rag` (tedarikçi RAG aracı) → `v6`
(toplu karar motoru, tek mağaza, Prophet) → **`v7` (canlı çok araçlı asistan)**.
Ara sürümler ölü kod değildir; `v7` onları import eder.
