# Fire Yönetimi ve Tedarikçi Seçim Rehberi

**Doküman No:** POL-2015-015
**Yürürlük:** 01.05.2015
**Dayanak:** 2015 Haziran–Temmuz dönemi stok simülasyon analizi (36 iş günü,
Mağaza 1, taze kategori)

## 1. Fire Yazma Prosedürü

Raf ömrünü dolduran partiler gün sonunda fire olarak kaydedilir. Fire maliyeti,
malın **alış maliyeti** üzerinden hesaplanır; satış fiyatı üzerinden değil.
Fire, satın alınıp satılamayan mal olduğu için ayrıca cezalandırılmaz —
maliyeti zaten satın alma tutarında gerçekleşmiştir.

Stoksuz kalma ise **kaybedilen brüt marj** olarak fırsat maliyeti kaydedilir.

## 2. Geçmiş Analiz Sonuçları — Tedarikçi Performansı

3 günlük raf ömründe, 36 iş günlük simülasyon sonuçları:

| Tedarikçi | Fire (birim) | Stoksuz (birim) | Net kâr (€) |
|---|---|---|---|
| **ExpressLog** | 0 | 0 | **+5.174** |
| Schnellware | 537 | 126 | +4.435 |
| Rheinland | 1.806 | 1.270 | −4.929 |
| Nordmann | 4.210 | 1.930 | −18.693 |

**Ana bulgu:** Birim fiyatı en yüksek olan ExpressLog (%12 pahalı), toplam
sahip olma maliyetinde en kârlı tedarikçidir. Günlük teslimat yapması ve
minimum sipariş miktarının düşük olması (200 birim) sayesinde fire ve stoksuz
kalma sıfıra iner. Birim fiyat avantajı, fire maliyeti karşısında anlamını
yitirir.

**Nordmann uyarısı:** %7 daha ucuz olmasına rağmen 5 günlük lead-time ve 1.500
birimlik minimum sipariş, 3 günlük raf ömrüyle yapısal olarak uyumsuzdur. Mal
satılmadan bozulur. Taze kategoride kullanılmamalıdır.

## 3. Raf Ömrüne Göre Tedarikçi Seçimi

Doğru tedarikçi, ürünün raf ömrüne göre değişir; sabit bir "en iyi tedarikçi"
yoktur.

| Raf ömrü | Önerilen tedarikçi | Gerekçe |
|---|---|---|
| 2–3 gün | **ExpressLog** | tek uygun seçenek, diğerleri fire veriyor |
| 4–7 gün | **Schnellware** | fire baskısı azalır, birim fiyat avantajı öne çıkar |
| 7 gün üzeri | **Nordmann** | raf ömrü bağlayıcı değil, en düşük birim fiyat kazanır |

Promosyon dönemlerinde Rheinland'ın öncelikli tahsis hakkı ve kampanya
iskontosu devreye girer; ancak raf ömrü 5 günün altındaysa Rheinland'ın 3
günlük lead-time'ı ve Salı/Perşembe teslimat kısıtı fire riski yaratır. Bu
durumda promosyon avantajı yerine ExpressLog tercih edilir.

## 4. Sipariş Miktarı Politikası Üzerine Not

Simülasyonda iki sipariş politikası karşılaştırılmıştır: beklenen talebe göre
sipariş ("ortalama") ve maliyet asimetrisini dikkate alan kuantil siparişi
("newsvendor"). Kısıtlı senaryoda iki politika arasında anlamlı fark
bulunamamıştır.

Sebep: tedarikçinin minimum sipariş miktarı, günlük talebin birkaç katına denk
geldiğinde sipariş miktarını politika değil sözleşme belirler. Bu koşulda
sipariş miktarı ince ayarı yerine **tedarikçi seçimi** üzerinde durulmalıdır;
kâr farkının tamamı oradan gelmektedir.

## 5. Aksiyon Önceliği

Fazla stok tespit edildiğinde sırasıyla değerlendirilir:

1. Raf ömrü dolmadan satılabilir mi → aksiyon gerekmez
2. İndirimle eritilebilir mi → raf ömrü ve indirim politikasına bakılır
3. Eritilemeyecekse → fire kaydı açılır ve tedarikçi/sipariş miktarı gözden
   geçirilir
