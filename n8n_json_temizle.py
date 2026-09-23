"""
n8n_json_temizle.py — n8n workflow export'undaki KISISEL bilgileri temizler.

NE YAPAR
    - Telegram node'larindaki SABIT (dinamik expression olmayan) chatId
      degerlerini {{TELEGRAM_CHAT_ID}} placeholder'iyla degistirir.
    - Google Sheets node'larindaki documentId.value (tam URL) alanini
      {{GOOGLE_SHEET_URL}} placeholder'iyla degistirir.
    - Credential id/name alanlarina DOKUNMAZ (zaten disaridan anlamsizdir,
      gercek token/anahtar export'ta hic bulunmaz — n8n boyle calisir).

NE YAPMAZ
    - webhookId, node id gibi n8n'in kendi urettigi rastgele degerleri
      degistirmez; bunlar kisisel bilgi degildir.

KULLANIM
    python n8n_json_temizle.py n8n/Rossmann_Birlesik_Akis_v11.json

    Ciktiyi ayni klasore "<ad>_temiz.json" olarak yazar, orijinali
    BOZMAZ. Kac deger degistirildigini ekrana yazar.

REPO'YA YUKLERKEN
    "_temiz.json" dosyasini kullan. README'ye bir not eklemeyi unutma:
    "chatId ve Sheets URL placeholder'dir; kendi degerlerinizle
    degistirin" gibi.
"""

import json
import sys
from pathlib import Path

PLACEHOLDER_CHAT = "{{TELEGRAM_CHAT_ID}}"
PLACEHOLDER_SHEET = "{{GOOGLE_SHEET_URL}}"


def temizle(yol: Path) -> None:
    veri = json.loads(yol.read_text(encoding="utf-8"))
    chat_degisen = 0
    sheet_degisen = 0

    for n in veri.get("nodes", []):
        tip = n.get("type", "")
        par = n.get("parameters", {})

        # Telegram: sabit chatId'yi degistir, expression'a (={{ ... }}) DOKUNMA
        if "telegram" in tip.lower():
            cid = par.get("chatId")
            if isinstance(cid, str) and cid and not cid.startswith("={{"):
                par["chatId"] = PLACEHOLDER_CHAT
                chat_degisen += 1

        # Google Sheets: documentId.value icindeki URL'i degistir
        if tip == "n8n-nodes-base.googleSheets":
            doc = par.get("documentId")
            if isinstance(doc, dict) and doc.get("value"):
                if "docs.google.com" in str(doc["value"]):
                    doc["value"] = PLACEHOLDER_SHEET
                    doc["cachedResultUrl"] = PLACEHOLDER_SHEET
                    sheet_degisen += 1

    cikti = yol.with_name(yol.stem + "_temiz.json")
    cikti.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"okundu   : {yol}")
    print(f"chatId degisen   : {chat_degisen}")
    print(f"sheet url degisen: {sheet_degisen}")
    print(f"yazildi  : {cikti}")
    if chat_degisen == 0 and sheet_degisen == 0:
        print("\nUYARI: hicbir sey degismedi. Node adlari/tipleri bu dosyada "
              "farkli olabilir; workflow'u guncellediysen elle kontrol et.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("kullanim: python n8n_json_temizle.py <workflow.json>")
    temizle(Path(sys.argv[1]))