"""
rag_index.py — Tedarikci belgelerini vektor veritabanina indeksler.

Akis: belgeleri yukle -> parcala (chunk) -> baslik baglami ekle -> embedding
      -> Chroma'ya yaz. Indeks diske kalici yazilir, agent her calistiginda
      yeniden kurulmaz.

ONBELLEK: Chroma istemcisi ve embedding nesnesi modul seviyesinde onbellege
alinir. Agent tool'lari LangGraph tarafindan thread havuzunda calistirildigi
icin, her cagrida yeni PersistentClient acmak Chroma'nin paylasimli istemci
yonetimini bozuyordu (RustBindingsAPI hatasi).

BASLIK BAGLAMI: Her parcanin basina "[kaynak > ust baslik]" satiri eklenir.
Belgeler alt basliklar icerdiginden (or. satinalma_politikasi.md Madde
6.1/6.2/6.3), 600 karakterlik parcalar ust baglamdan kopuyordu.

OLCUM HATASI VE DUZELTMESI — BASLIK ATAMASI:
_ust_baslik() onceki surumde metin[:start_index] dilimi icinde geriye dogru
"## " ariyordu. Parca tam olarak kendi basligiyla BASLIYORSA o baslik dilime
girmiyor, fonksiyon bir ONCEKI basligi donduruyordu: "## 2. Onay Esikleri"
parcasi "1. Tedarikci Secim Kurallari" olarak etiketleniyordu. Getirme dogru
calisirken degerlendirme yanlis parcayi gormus gibi rapor ediyordu.
Duzeltme: parca kendi basligiyla basliyorsa dogrudan o kullanilir.

Onkosul:
    python tedarikci_dokumanlari_olustur.py

Calistirma:
    python rag_index.py                          # indeksi kur
    python rag_index.py --sorgu "acil teslimat"  # indeksi test et
"""

import os
import re
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BURASI = Path(__file__).resolve().parent
KOK = BURASI / "Rossman" if (BURASI / "Rossman").is_dir() else BURASI
BELGE_DIR = KOK / "tedarikci_dokumanlari"
INDEKS_DIR = KOK / "chroma_tedarikci"

EMBEDDING_MODELI = "text-embedding-3-small"
CHUNK_BOYUT = 600      # belgeler kisa; kucuk chunk = daha isabetli getirme
CHUNK_ORTUSME = 100    # bolum sinirlarinda baglam kaybini onler

# indeks kurulduktan sonra otomatik calisan kendi kendini test sorgulari
KONTROL_SORGULARI = [
    "onay esigi kac birim",
    "minimum siparis miktari nasil uygulanir",
    "cok magazali isletimde onay kurallari",
]

_EMB = None            # onbellek
_DB = None             # onbellek


def embedding_olustur() -> OpenAIEmbeddings:
    global _EMB
    if _EMB is None:
        load_dotenv()
        anahtar = os.getenv("openai_apikey")
        if not anahtar:
            sys.exit("HATA: .env icinde openai_apikey yok.")
        _EMB = OpenAIEmbeddings(
            model=EMBEDDING_MODELI,
            api_key=anahtar.strip().strip('"'),
        )
    return _EMB


def _ust_baslik(metin: str, konum: int, parca: str) -> str:
    """Parcanin ait oldugu '## ' basligini bulur.

    Parca kendi basligiyla BASLIYORSA dogrudan onu kullanir; aksi halde
    belgede geriye dogru en yakin ust basligi arar (bkz. OLCUM HATASI notu)."""
    ilk = parca.lstrip().split("\n")[0].strip()
    if ilk.startswith("## ") and not ilk.startswith("### "):
        return ilk.lstrip("#").strip()
    basliklar = [m.group(1).strip()
                 for m in re.finditer(r"^##\s+(.+)$", metin[:konum], re.MULTILINE)]
    return basliklar[-1] if basliklar else ""


def parcala(belgeler: list[Document]) -> list[Document]:
    """Belgeleri parcalar ve her parcanin basina kaynak + ust baslik ekler.

    rag_hybrid.py da BU fonksiyonu cagirir — BM25 ve Chroma birebir ayni
    parcalar uzerinde calismak zorunda (bkz. rag_hybrid SESSIZ HATA notu)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_BOYUT,
        chunk_overlap=CHUNK_ORTUSME,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],   # basliklardan bol
        add_start_index=True,
    )
    ham = splitter.split_documents(belgeler)

    kaynak_metin = {d.metadata["kaynak"]: d.page_content for d in belgeler}
    sonuc = []
    for p in ham:
        kaynak = p.metadata.get("kaynak", "")
        bas = p.metadata.get("start_index", 0)
        baslik = _ust_baslik(kaynak_metin.get(kaynak, ""), bas, p.page_content)
        p.metadata["baslik"] = baslik

        # baslik parcanin icinde zaten varsa etikette tekrar etme
        ilk_satir = p.page_content.lstrip().split("\n")[0]
        if baslik and baslik in ilk_satir:
            p.page_content = f"[{kaynak}]\n{p.page_content}"
        else:
            etiket = f"[{kaynak}" + (f" > {baslik}]" if baslik else "]")
            p.page_content = f"{etiket}\n{p.page_content}"
        sonuc.append(p)
    return sonuc


def indeks_kur() -> None:
    global _DB
    if not BELGE_DIR.is_dir():
        sys.exit(f"HATA: {BELGE_DIR} yok.\n"
                 "Once 'python tedarikci_dokumanlari_olustur.py' calistir.")

    dosyalar = sorted(BELGE_DIR.glob("*.md"))
    if not dosyalar:
        sys.exit(f"HATA: {BELGE_DIR} icinde .md belge yok.")

    print(f"{len(dosyalar)} belge bulundu.")

    belgeler = [
        Document(page_content=d.read_text(encoding="utf-8"),
                 metadata={"kaynak": d.name})
        for d in dosyalar
    ]

    parcalar = parcala(belgeler)
    print(f"{len(parcalar)} parcaya (chunk) bolundu.\n")

    print(f"{'belge':<44}{'parca':>7}{'karakter':>10}")
    for d in belgeler:
        ait = [p for p in parcalar if p.metadata["kaynak"] == d.metadata["kaynak"]]
        print(f"{d.metadata['kaynak']:<44}{len(ait):>7}{len(d.page_content):>10}")

    # baslik atamasi dogru mu — ilk birkac parcayi goster (olcum hatasi tekrarlamasin)
    print(f"\n{'parca':<6}{'kaynak':<40}baslik")
    print("-" * 100)
    for i, p in enumerate(parcalar[:12], 1):
        print(f"{i:<6}{p.metadata['kaynak']:<40}{p.metadata.get('baslik') or '-'}")

    if INDEKS_DIR.exists():
        shutil.rmtree(INDEKS_DIR)
        print("\nEski indeks silindi.")

    _DB = Chroma.from_documents(
        documents=parcalar,
        embedding=embedding_olustur(),
        persist_directory=str(INDEKS_DIR),
        collection_name="tedarikci",
    )
    print(f"Indeks kuruldu: {INDEKS_DIR}")

    print("\n" + "=" * 62)
    print("KONTROL SORGULARI (ilk sonucun kaynagi ve basligi)")
    print("=" * 62)
    for soru in KONTROL_SORGULARI:
        sonuc = _DB.similarity_search(soru, k=1)
        if sonuc:
            m = sonuc[0].metadata
            print(f"  {soru:<42} -> {m.get('kaynak')} > {m.get('baslik') or '-'}")
        else:
            print(f"  {soru:<42} -> SONUC YOK")
    print("\nDetayli degerlendirme: python rag_hybrid.py --test")


def indeks_yukle() -> Chroma:
    """Chroma istemcisini bir kez acar, sonraki cagrilarda ayni nesneyi doner."""
    global _DB
    if _DB is None:
        if not INDEKS_DIR.exists():
            sys.exit(f"HATA: {INDEKS_DIR} yok. Once 'python rag_index.py' calistir.")
        _DB = Chroma(
            persist_directory=str(INDEKS_DIR),
            embedding_function=embedding_olustur(),
            collection_name="tedarikci",
        )
    return _DB


def sorgula(soru: str, k: int = 3) -> None:
    db = indeks_yukle()
    sonuclar = db.similarity_search_with_score(soru, k=k)

    print(f"\n>>> SORU: {soru}\n")
    for i, (belge, skor) in enumerate(sonuclar, 1):
        print(f"--- {i}. sonuc | kaynak: {belge.metadata.get('kaynak')} "
              f"| baslik: {belge.metadata.get('baslik') or '-'} "
              f"| uzaklik: {skor:.4f} ---")
        print(belge.page_content.strip()[:400])
        print()


if __name__ == "__main__":
    if "--sorgu" in sys.argv:
        sorgula(" ".join(a for a in sys.argv[1:] if a != "--sorgu").strip())
    else:
        indeks_kur()