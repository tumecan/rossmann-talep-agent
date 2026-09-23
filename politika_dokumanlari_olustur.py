"""
politika_dokumanlari_olustur.py — RAG belge setine 2 belge daha ekler.

1) raf_omru_ve_indirim_politikasi.md
   Kategori bazinda raf omru, indirim kademeleri, erime oranlari, onay yetkisi.
   Agent "israf mi indirim mi" karsilastirmasini bu belgedeki katsayilarla yapar.

2) fire_yonetimi_ve_tedarikci_rehberi.md
   Fire yazma proseduru + stok simulasyonundan cikan GECMIS ANALIZ sonuclari.
   Bu belge sayesinde RAG yalnizca sozlesme metni degil, kurumsal hafiza olur:
   agent tedarikci secimini gecmis performans verisine dayandirabilir.

Calistirma:
    python politika_dokumanlari_olustur.py
    python rag_index.py          # indeksi yeniden kur (7 belge olacak)
"""

from pathlib import Path

BURASI = Path(__file__).resolve().parent
HEDEF = (BURASI / "Rossman" if (BURASI / "Rossman").is_dir() else BURASI) / "tedarikci_dokumanlari"

BELGELER = {}

# ---------------------------------------------------------------------------
BELGELER["raf_omru_ve_indirim_politikasi.md"] = """# Raf Ömrü ve İndirim Politikası

**Doküman No:** POL-2015-014
**Yürürlük:** 01.04.2015
**Kapsam:** Kısa raf ömürlü (taze) kategori stok yönetimi

## 1. Kategori Tanımları ve Raf Ömrü

| Kategori | Ciro payı | Raf ömrü | Brüt marj |
|---|---|---|---|
| Taze (süt, ekmek, hazır gıda) | %20 | **3 gün** | %25 |
| Kozmetik | %50 | 180 gün | %35 |
| Temizlik | %30 | 365 gün | %22 |

Bu politika yalnızca **taze** kategoriyi kapsar. Diğer kategorilerde raf ömrü
pratikte bağlayıcı olmadığından israf yönetimi uygulanmaz.

Raf ömrü, malın mağazaya teslim edildiği günden itibaren sayılır. Süresi dolan
partiler satılamaz ve fire yazılır.

## 2. İndirim Kademeleri

Raf ömrünün dolmasına kalan süreye göre indirim uygulanır:

| Kalan süre | İndirim oranı | Beklenen erime oranı |
|---|---|---|
| 2 gün | **%30** | fazla stoğun **%60'ı** satılır |
| 1 gün | **%50** | fazla stoğun **%85'i** satılır |

"Erime oranı", indirim uygulandığında fazla stoğun ne kadarının satılabildiğini
gösterir. Erimeyen kısım yine fire yazılır.

## 3. İndirim Kârlılık Hesabı

Bir indirim kararı, indirim yapılmaması durumuyla karşılaştırılarak verilir.

- **İndirim yok:** fazla stoğun tamamı fire olur → zarar = fazla × birim maliyet
- **İndirim var:** eriyen kısım indirimli fiyattan satılır, kalanı fire olur

İndirimli satış geliri, birim maliyetin altına düşse bile fire yazmaktan daha
iyi olabilir; çünkü fire durumunda gelir sıfırdır. Karar her zaman bu iki
seçeneğin karşılaştırılmasıyla verilir.

## 4. Onay Yetkisi

- **%30 indirim:** mağaza müdürü onayı yeterlidir, sistem otomatik uygulayabilir.
- **%50 indirim:** bölge müdürü onayı zorunludur, otomatik uygulanamaz.
- Aynı ürün grubunda **peş peşe iki gün %50 indirim uygulanamaz**. Bu durumda
  ikinci gün en fazla %30 indirim uygulanır ve fark fire olarak kabul edilir.

## 5. Promosyon Dönemi İstisnası

Aktif promosyon döneminde emniyet stoğu oranı %15'ten %25'e çıkarıldığı için
fire riski artar. Promosyon biten günün ertesinde stok gözden geçirilir ve
gerekiyorsa doğrudan %30 indirim uygulanır.
"""

# ---------------------------------------------------------------------------
BELGELER["fire_yonetimi_ve_tedarikci_rehberi.md"] = """# Fire Yönetimi ve Tedarikçi Seçim Rehberi

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
"""


if __name__ == "__main__":
    if not HEDEF.is_dir():
        raise SystemExit(
            f"HATA: {HEDEF} yok.\n"
            "Once 'python tedarikci_dokumanlari_olustur.py' calistir."
        )
    for ad, icerik in BELGELER.items():
        (HEDEF / ad).write_text(icerik, encoding="utf-8")
        print(f"  yazildi: {ad}  ({len(icerik)} karakter)")

    toplam = len(list(HEDEF.glob("*.md")))
    print(f"\nBelge setinde toplam {toplam} belge var -> {HEDEF}")
    print("Simdi indeksi yeniden kur:  python rag_index.py")