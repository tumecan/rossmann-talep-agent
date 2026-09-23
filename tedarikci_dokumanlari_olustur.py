"""
tedarikci_dokumanlari_olustur.py — RAG icin belge seti uretir.

Rossmann veri setinde tedarikci bilgisi YOKTUR. Bu belgeler, gercek bir
perakende operasyonundaki tedarikci sozlesmelerini temsil eden SENTETIK
belgelerdir. Amac: agent'in yapilandirilmamis metinden bilgi cekmesini
(RAG) gostermek.

Calistirma:
    python tedarikci_dokumanlari_olustur.py
Cikti:
    Rossman/tedarikci_dokumanlari/*.md  (5 belge)
"""

from pathlib import Path

BURASI = Path(__file__).resolve().parent
HEDEF = (BURASI / "Rossman" if (BURASI / "Rossman").is_dir() else BURASI) / "tedarikci_dokumanlari"

BELGELER = {}

# ---------------------------------------------------------------------------
BELGELER["tedarikci_A_schnellware.md"] = """# Tedarikçi Sözleşmesi — Schnellware GmbH (Tedarikçi A)

**Sözleşme No:** TS-2015-001
**Geçerlilik:** 01.01.2015 – 31.12.2015
**Kategori:** Standart / birincil tedarikçi

## Teslimat Koşulları
Schnellware GmbH, onaylanan siparişleri **2 iş günü** içinde teslim eder.
Teslimat günleri Pazartesi, Çarşamba ve Cuma'dır. Hafta sonu teslimatı yapılmaz.
Sipariş kesim saati 16:00'dır; bu saatten sonra girilen siparişler bir sonraki
iş gününde işleme alınır.

## Sipariş Limitleri
- Minimum sipariş miktarı: **500 birim**
- Maksimum günlük sipariş: 8.000 birim
- 500 birimin altındaki siparişler sistem tarafından reddedilir.

## Fiyatlandırma ve İskonto
Baz fiyat üzerinden kademeli iskonto uygulanır:

| Sipariş miktarı | İskonto |
|---|---|
| 500 – 1.999 birim | iskonto yok |
| 2.000 – 4.999 birim | %5 |
| 5.000 birim ve üzeri | %8 |

İskontolar kümülatif değildir, yalnızca ilgili kademe uygulanır.

## Notlar
Schnellware, rutin ve planlı siparişler için varsayılan tedarikçidir.
Stok devamlılığı yüksektir, son 12 ayda teslimat gecikmesi oranı %1,8'dir.
Acil ihtiyaçlarda lead-time yetersiz kalabilir; bu durumda ExpressLog
sözleşmesine bakınız.
"""

# ---------------------------------------------------------------------------
BELGELER["tedarikci_B_nordmann.md"] = """# Tedarikçi Sözleşmesi — Nordmann Handel AG (Tedarikçi B)

**Sözleşme No:** TS-2015-002
**Geçerlilik:** 01.01.2015 – 31.12.2015
**Kategori:** Maliyet odaklı / toplu alım

## Teslimat Koşulları
Nordmann Handel AG teslimat süresi **5 iş günüdür**. Tüm hafta içi günlerde
teslimat yapılabilir, gün kısıtı yoktur. Uzun lead-time nedeniyle yalnızca
önceden planlanabilen siparişlerde kullanılır.

## Sipariş Limitleri
- Minimum sipariş miktarı: **1.500 birim**
- Maksimum sipariş sınırı yoktur.
- 1.500 birimin altındaki talepler için Nordmann kullanılamaz.

## Fiyatlandırma ve İskonto
Nordmann'ın baz fiyatı Schnellware'den **%7 daha düşüktür**. Ek olarak:

| Sipariş miktarı | İskonto |
|---|---|
| 1.500 – 2.999 birim | iskonto yok |
| 3.000 birim ve üzeri | %10 |

Toplam maliyet avantajı büyük hacimli siparişlerde belirgindir.

## Notlar
Nordmann, maliyet optimizasyonu hedeflendiğinde ve teslimat aciliyeti
düşük olduğunda tercih edilir. Aciliyeti "yuksek" olan siparişlerde
kullanılması önerilmez. Teslimat gecikmesi oranı %4,3'tür.
"""

# ---------------------------------------------------------------------------
BELGELER["tedarikci_C_expresslog.md"] = """# Tedarikçi Sözleşmesi — ExpressLog Logistik (Tedarikçi C)

**Sözleşme No:** TS-2015-003
**Geçerlilik:** 01.01.2015 – 31.12.2015
**Kategori:** Acil / kısa vadeli tedarik

## Teslimat Koşulları
ExpressLog **aynı gün veya 1 iş günü** içinde teslimat yapar.
Aynı gün teslimat için sipariş saat **14:00'ten önce** girilmelidir.
14:00 sonrası siparişler ertesi gün teslim edilir. Cumartesi teslimatı
ek ücretle mümkündür, Pazar teslimatı yoktur.

## Sipariş Limitleri
- Minimum sipariş miktarı: **200 birim**
- Maksimum günlük sipariş: 3.000 birim
- 3.000 birimi aşan acil ihtiyaçlarda sipariş bölünmelidir.

## Fiyatlandırma
ExpressLog baz fiyatı Schnellware'den **%12 yüksektir**. İskonto kademesi
yoktur; hacim ne olursa olsun aynı fiyat uygulanır.

## Notlar
ExpressLog yalnızca aciliyeti "yuksek" olan ve stok tükenme riski bulunan
durumlarda kullanılır. Yüksek maliyeti nedeniyle rutin siparişlerde
kullanımı satınalma politikasınca kısıtlanmıştır.
Teslimat gecikmesi oranı %0,6 ile en düşük orandır.
"""

# ---------------------------------------------------------------------------
BELGELER["tedarikci_D_rheinland.md"] = """# Tedarikçi Sözleşmesi — Rheinland Distribution (Tedarikçi D)

**Sözleşme No:** TS-2015-004
**Geçerlilik:** 01.01.2015 – 31.12.2015
**Kategori:** Promosyon dönemi / kampanya tedariki

## Teslimat Koşulları
Rheinland Distribution teslimat süresi **3 iş günüdür**. Teslimat Salı ve
Perşembe günleri yapılır.

## Sipariş Limitleri
- Minimum sipariş miktarı: **1.000 birim**
- Promosyon dönemlerinde minimum miktar **500 birime** düşürülür.

## Promosyon Önceliği
Rheinland ile yapılan sözleşme, **promosyon dönemlerinde öncelikli tahsis**
hakkı içerir. Mağazada aktif promosyon varsa, Rheinland siparişleri diğer
müşterilerin önüne alınır ve lead-time 3 günden **2 güne** düşer.
Bu ayrıcalık yalnızca promosyonlu günler için geçerlidir.

## Fiyatlandırma
Baz fiyat Schnellware ile aynıdır. Promosyon dönemi siparişlerinde
sabit **%6 kampanya iskontosu** uygulanır, miktar kademesi aranmaz.

## Notlar
Promosyonlu günlerde Rheinland ilk tercih olmalıdır. Promosyon olmayan
günlerde belirgin bir avantajı yoktur; standart siparişler için
Schnellware tercih edilir.
"""

# ---------------------------------------------------------------------------
BELGELER["satinalma_politikasi.md"] = """# İç Satınalma Politikası — Perakende Operasyonları

**Doküman No:** POL-2015-011
**Yürürlük:** 01.03.2015
**Kapsam:** Tüm mağaza bazlı stok siparişleri

## 1. Tedarikçi Seçim Kuralları
Sipariş kararı verildikten sonra tedarikçi şu önceliğe göre belirlenir:

1. Mağazada **aktif promosyon** varsa → Rheinland Distribution (öncelikli
   tahsis ve kampanya iskontosu nedeniyle).
2. Aciliyet **yuksek** ve stok tükenme riski varsa → ExpressLog Logistik.
3. Sipariş miktarı **3.000 birim ve üzeriyse** ve aciliyet düşükse →
   Nordmann Handel (maliyet avantajı).
4. Diğer tüm durumlarda → Schnellware GmbH (varsayılan tedarikçi).

Seçilen tedarikçinin minimum sipariş miktarı sağlanamıyorsa, bir sonraki
uygun tedarikçiye geçilir.

## 2. Onay Eşikleri
- **3.000 birime kadar:** mağaza müdürü onayı yeterlidir, sistem otomatik
  onaylayabilir.
- **3.000 birim üzeri:** bölge müdürü onayı zorunludur, otomatik onay
  yapılamaz.
- Modelin belirsiz kaldığı (insan onayı istenen) siparişlerde miktar ne
  olursa olsun otomatik onay yapılmaz.

## 3. Tatil ve Kapalı Gün Kuralı
Mağazanın kapalı olduğu günler için sipariş oluşturulmaz. Resmi tatil
öncesindeki son iş gününde sipariş miktarı, tatil sonrası ilk günün
talebini de karşılayacak şekilde artırılabilir.

## 4. Emniyet Stoğu
Standart emniyet stoğu tahmini talebin **%15'idir**. Promosyon dönemlerinde
bu oran **%25'e** çıkarılır. Emniyet stoğu hesaplaması otomatik sistem
tarafından yapılır, manuel değiştirilemez.

## 5. İstisna Yönetimi
Politikaya aykırı sipariş verilmesi gerekiyorsa gerekçe yazılı olarak
kaydedilir ve satınalma birimine bildirilir.

## 6. Çok Mağazalı İşletim — Revize Onay Kuralları (Rev. 3, 03.07.2015)

Bu madde, zincir genelinde (1115 mağaza) merkezi satınalma ile yürütülen
işletim için geçerlidir. Tek mağazalı ve pilot işletimde Madde 1 ve 2
uygulanmaya devam eder.

### 6.1 Mağaza Düzeyi Onay (risk temelli — mağaza müdürü)

Sipariş miktarına bakılmaksızın, aşağıdaki durumların herhangi biri varsa
sipariş mağaza müdürü onayına gider:

- **Kritik stok:** yüksek talep senaryosunda (p90) mevcut stok 1 gün içinde
  tükeniyorsa,
- **Yüksek fire riski:** raf ömrünün son gününde bulunan stok, günlük talebin
  **%40'ını** aşıyorsa,
- **Belirsiz durum:** mevcut stok, ilk günün p10–p90 tahmin bandının içinde
  kalıyorsa (talep bandı stok seviyesini kapsıyorsa karar belirsizdir),
- **Ardışık indirim:** aynı ürün grubunda son iki gün içinde indirim
  uygulanmışsa.

### 6.2 Zincir Düzeyi Onay (tutar temelli — bölge müdürü)

- Tek bir mağazaya ait tekil sipariş tutarı **4.000 EUR**'yu aşarsa,
- Bir tedarikçiye aynı gün verilen **toplam** sipariş tutarı **500.000 EUR**'yu
  aşarsa.

**Rev. 3 notu — eşik neden değişti:** Rev. 2'de günlük toplam eşiği 150.000 EUR
olarak belirlenmişti. Bu değer, dar kapsamlı pilot işletim için tanımlanmıştı.
1115 mağazalı zincir işletiminde bir tedarikçiye verilen günlük toplam
siparişin medyanı 185.000 EUR düzeyindedir; dolayısıyla 150.000 EUR eşiği
rutin günlerin neredeyse tamamını onaya taşımakta ve ayırt edici olmaktan
çıkmaktadır. Eşik, rutin olmayan günleri yakalayacak şekilde 500.000 EUR'ya
yükseltilmiştir. Rev. 2'deki 150.000 EUR eşiği, tek mağazalı senaryolarda ve
pilot raporlamada geçerliliğini korur.

### 6.3 Minimum Sipariş Miktarının Uygulanması

Tedarikçi sözleşmelerindeki minimum sipariş miktarı, tedarikçiye verilen
**günlük toplam** siparişe uygulanır; mağaza başına değil. Merkezi satınalma
toplu sipariş verir, dağıtım mağazalara yapılır.

Madde 2 ve Madde 1'deki miktar temelli kurallar, tek mağazalı senaryolarda ve
pilot raporlamada geçerliliğini korur.
"""


if __name__ == "__main__":
    HEDEF.mkdir(parents=True, exist_ok=True)
    for ad, icerik in BELGELER.items():
        (HEDEF / ad).write_text(icerik, encoding="utf-8")
        print(f"  yazildi: {ad}  ({len(icerik)} karakter)")
    print(f"\nToplam {len(BELGELER)} belge -> {HEDEF}")