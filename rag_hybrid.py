"""
rag_hybrid.py — Hibrit arama: BM25 (kelime) + Chroma (anlam), RRF ile birlestirme.

NEDEN: Sadece dense (embedding) arama, "3.000 birim" gibi sayisal ve "onay
esigi" gibi spesifik terimlerde zayif kaliyordu; tum tedarikci belgeleri
anlamsal olarak birbirine cok benzedigi icin siralama bozuluyordu.
BM25 kelime eslesmesine bakar, dense anlama bakar.

BIRLESTIRME YONTEMI: Reciprocal Rank Fusion (RRF)
    skor(belge) = toplam( 1 / (K + sira) )   her iki listede de
Skorlari degil SIRALARI birlestirir; olcekleri farkli iki skoru normalize
etme derdi kalmaz.

ONBELLEK: chunk listesi ve BM25 indeksi modul seviyesinde onbellege alinir.

SESSIZ HATA VE DUZELTMESI — TEK PARCALAMA:
Parcalama iki yerde ayri tanimliydi: rag_index.py Chroma'yi kurarken, bu dosya
BM25 icin kendi parcalarini uretiyordu. rag_index'e chunk basina
"[kaynak > baslik]" etiketi eklenince iki liste kaydi; dense_sirala() Chroma
sonucunu METIN ESLESTIREREK buldugu icin hicbir eslesme olmadi, dense listesi
BOS dondu, RRF tek listeyle calisti. Cikti "hibrit" gorunuyordu ama saf
BM25'ti — kod hata vermedi.
Duzeltme: parcalama TEK yerde (rag_index.parcala) + eslesme orani kontrolu.

DEGERLENDIRME:
"1. sonuc dogru mu" yanlis metrikti; agent ilk K parcayi aliyor. --test her
sorgunun BEKLENEN kaynagini tanimlayip uc yontemi ilk-K isabeti ve MRR ile
kiyasliyor. Beklenen parca alt baslikta (###) olabilecegi icin eslesme hem
baslik metadata'sinda hem parca ICERIGINDE aranir.

Onkosul:
    python rag_index.py          (Chroma indeksi kurulmus olmali)
    pip install rank_bm25

Calistirma:
    python rag_hybrid.py --sorgu "3000 birim uzeri onay"
    python rag_hybrid.py --karsilastir "3000 birim uzeri onay"
    python rag_hybrid.py --test
"""

import re
import sys

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_index import BELGE_DIR, indeks_yukle, parcala

RRF_K = 60          # RRF sabiti; standart deger 60
ADAY_SAYISI = 8     # her yontemden alinacak aday sayisi (fuzyon oncesi)
GETIRILEN_K = 3     # agent'a verilen parca sayisi — degerlendirme bu K'ya gore

# Degerlendirme seti: (sorgu, beklenen kaynak, baslikta/icerikte gecmesi gereken metin)
TEST_SETI = [
    ("onay esigi kac birim",
     "satinalma_politikasi.md", "Onay Eşikleri"),
    ("minimum siparis miktari nasil uygulanir",
     "satinalma_politikasi.md", "Minimum sipariş"),
    ("cok magazali isletimde onay kurallari",
     "satinalma_politikasi.md", "Çok Mağazalı"),
    ("pes pese iki gun yuzde 50 indirim yapilabilir mi",
     "raf_omru_ve_indirim_politikasi.md", ""),
    ("en hizli teslimat yapan tedarikci hangisi",
     "tedarikci_C_expresslog.md", ""),
    ("nordmann ne zaman secilir",
     "satinalma_politikasi.md", "Tedarikçi Seçim"),
    ("promosyon doneminde emniyet stogu orani",
     "satinalma_politikasi.md", "Emniyet"),
]

_PARCALAR = None
_BM25 = None


# ---------------------------------------------------------------------------
# Tokenizasyon — Turkce ve sayi duyarli
# ---------------------------------------------------------------------------
def tokenize(metin: str) -> list[str]:
    """Turkce kucultme + binlik ayraci temizligi + noktalama atma."""
    metin = metin.replace("I", "ı").replace("İ", "i").lower()
    metin = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", metin)     # "3.000" -> "3000"
    metin = re.sub(r"[^\wçğıöşü]+", " ", metin, flags=re.UNICODE)
    return [t for t in metin.split() if len(t) > 1]


def parcalari_getir() -> list[Document]:
    """rag_index ile AYNI parcalama fonksiyonu — BM25 ve Chroma birebir ayni
    parcalar uzerinde calismak zorunda (bkz. SESSIZ HATA notu)."""
    global _PARCALAR
    if _PARCALAR is None:
        dosyalar = sorted(BELGE_DIR.glob("*.md"))
        if not dosyalar:
            sys.exit(f"HATA: {BELGE_DIR} icinde belge yok.")
        belgeler = [Document(page_content=d.read_text(encoding="utf-8"),
                             metadata={"kaynak": d.name}) for d in dosyalar]
        _PARCALAR = parcala(belgeler)
    return _PARCALAR


def _bm25_indeksi(parcalar: list[Document]) -> BM25Okapi:
    global _BM25
    if _BM25 is None:
        _BM25 = BM25Okapi([tokenize(p.page_content) for p in parcalar])
    return _BM25


# ---------------------------------------------------------------------------
# Arama
# ---------------------------------------------------------------------------
def bm25_sirala(soru: str, parcalar: list[Document], n: int) -> list[int]:
    """BM25'e gore en iyi n parcanin indekslerini dondurur."""
    skorlar = _bm25_indeksi(parcalar).get_scores(tokenize(soru))
    return sorted(range(len(skorlar)), key=lambda i: skorlar[i], reverse=True)[:n]


def dense_sirala(soru: str, parcalar: list[Document], n: int,
                 sessiz: bool = False) -> list[int]:
    """Chroma'ya gore en iyi n parcanin indekslerini dondurur.

    Eslesme kopmasi sessiz hataya yol actigindan burada kontrol ediliyor."""
    bulunan = indeks_yukle().similarity_search(soru, k=n)
    metin_to_idx = {p.page_content.strip(): i for i, p in enumerate(parcalar)}
    sonuc, kayip = [], 0
    for b in bulunan:
        idx = metin_to_idx.get(b.page_content.strip())
        if idx is None:
            kayip += 1
        elif idx not in sonuc:
            sonuc.append(idx)
    if kayip and not sessiz:
        print(f"  UYARI: Chroma sonucunun {kayip}/{len(bulunan)} parcasi yerel "
              f"listeyle eslesmedi. Indeks eski -> python rag_index.py")
    return sonuc


def rrf_birlestir(*siralamalar: list[int]) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: skorlari degil siralari birlestirir."""
    skor: dict[int, float] = {}
    for siralama in siralamalar:
        for sira, idx in enumerate(siralama, start=1):
            skor[idx] = skor.get(idx, 0.0) + 1.0 / (RRF_K + sira)
    return sorted(skor.items(), key=lambda x: x[1], reverse=True)


def hibrit_ara(soru: str, k: int = GETIRILEN_K) -> list[Document]:
    """Agent'in ve CLI'nin kullandigi ana fonksiyon."""
    parcalar = parcalari_getir()
    bm = bm25_sirala(soru, parcalar, ADAY_SAYISI)
    ds = dense_sirala(soru, parcalar, ADAY_SAYISI, sessiz=True)
    return [parcalar[idx] for idx, _ in rrf_birlestir(bm, ds)[:k]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _etiket(p: Document, genislik: int = 34) -> str:
    baslik = (p.metadata.get("baslik") or
              p.page_content.strip().split("\n")[0]).strip()
    kaynak = p.metadata.get("kaynak", "").replace(".md", "")
    return f"{kaynak[:26]:<26} {baslik[:genislik]}"


def yazdir(soru: str, k: int = GETIRILEN_K) -> None:
    print(f"\n>>> SORU: {soru}\n")
    for i, belge in enumerate(hibrit_ara(soru, k), 1):
        print(f"--- {i}. sonuc | {belge.metadata.get('kaynak')} "
              f"| {belge.metadata.get('baslik') or '-'} ---")
        print(belge.page_content.strip()[:400])
        print()


def karsilastir(soru: str, k: int = GETIRILEN_K) -> None:
    """Dense, BM25 ve hibrit siralamayi yan yana gosterir (sunum icin)."""
    parcalar = parcalari_getir()
    bm = bm25_sirala(soru, parcalar, ADAY_SAYISI)
    ds = dense_sirala(soru, parcalar, ADAY_SAYISI)
    hb = [idx for idx, _ in rrf_birlestir(bm, ds)]

    print(f"\n>>> SORU: {soru}")
    print(f"\n{'#':<3}{'SADECE DENSE':<62}{'SADECE BM25':<62}{'HIBRIT (RRF)'}")
    print("-" * 186)
    for i in range(k):
        d = _etiket(parcalar[ds[i]]) if i < len(ds) else ""
        b = _etiket(parcalar[bm[i]]) if i < len(bm) else ""
        h = _etiket(parcalar[hb[i]]) if i < len(hb) else ""
        print(f"{i+1:<3}{d:<62}{b:<62}{h}")
    print()


def _sira_bul(idxler: list[int], parcalar: list[Document],
              kaynak: str, baslik: str) -> int:
    """Beklenen parcanin kacinci sirada oldugunu doner (1 tabanli, 0 = yok).

    Beklenen metin alt baslikta (###) olabilecegi icin hem baslik
    metadata'sinda hem parca iceriginde aranir."""
    for sira, idx in enumerate(idxler, 1):
        p = parcalar[idx]
        if p.metadata.get("kaynak") != kaynak:
            continue
        if baslik:
            b = baslik.lower()
            if b not in (p.metadata.get("baslik") or "").lower() \
               and b not in p.page_content.lower():
                continue
        return sira
    return 0


def test_seti() -> None:
    """Uc yontemi ilk-K isabeti ve MRR ile kiyaslar (regresyon + sunum tablosu)."""
    parcalar = parcalari_getir()
    yontemler = ["dense", "bm25", "hibrit"]
    siralar = {y: [] for y in yontemler}

    print(f"\nDEGERLENDIRME — beklenen parca kacinci sirada? "
          f"(0 = ilk {ADAY_SAYISI} icinde yok)\n")
    print(f"{'sorgu':<48}{'dense':>7}{'bm25':>7}{'hibrit':>8}   beklenen")
    print("-" * 112)
    for soru, kaynak, baslik in TEST_SETI:
        bm = bm25_sirala(soru, parcalar, ADAY_SAYISI)
        ds = dense_sirala(soru, parcalar, ADAY_SAYISI, sessiz=True)
        hb = [idx for idx, _ in rrf_birlestir(bm, ds)]
        s = {"dense": _sira_bul(ds, parcalar, kaynak, baslik),
             "bm25": _sira_bul(bm, parcalar, kaynak, baslik),
             "hibrit": _sira_bul(hb, parcalar, kaynak, baslik)}
        for y in yontemler:
            siralar[y].append(s[y])
        bek = kaynak.replace(".md", "")[:24] + (f" > {baslik}" if baslik else "")
        print(f"{soru[:47]:<48}{s['dense']:>7}{s['bm25']:>7}{s['hibrit']:>8}   {bek}")

    print("-" * 112)
    n = len(TEST_SETI)

    def satir(ad, fn):
        print(f"{ad:<48}" + "".join(
            fn(siralar[y]).rjust(7 if y != "hibrit" else 8) for y in yontemler))

    satir("ilk-1 isabet", lambda s: f"{sum(1 for x in s if x == 1)}/{n}")
    satir(f"ilk-{GETIRILEN_K} isabet (agent bunu goruyor)",
          lambda s: f"{sum(1 for x in s if 1 <= x <= GETIRILEN_K)}/{n}")
    satir("MRR", lambda s: f"{sum(1/x for x in s if x)/n:.3f}")
    satir("hic bulunamadi", lambda s: str(sum(1 for x in s if x == 0)))
    print()


if __name__ == "__main__":
    bayraklar = {"--karsilastir", "--sorgu", "--test"}
    metin = " ".join(a for a in sys.argv[1:] if a not in bayraklar).strip()

    if "--test" in sys.argv:
        test_seti()
    elif "--karsilastir" in sys.argv:
        if not metin:
            sys.exit("Kullanim: python rag_hybrid.py --karsilastir \"sorgu metni\"")
        karsilastir(metin)
    elif "--sorgu" in sys.argv:
        if not metin:
            sys.exit("Kullanim: python rag_hybrid.py --sorgu \"sorgu metni\"")
        yazdir(metin)
    else:
        sys.exit(__doc__.split("Calistirma:")[-1].strip())