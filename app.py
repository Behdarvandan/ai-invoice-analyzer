import argparse
import json
import os
import sys

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

MODEL = "llama-3.3-70b-versatile"


class InvoiceData(BaseModel):
    company_name: str = Field(description="Faturayı düzenleyen şirketin adı")
    date: str = Field(description="Fatura tarihi (YYYY-MM-DD formatında, mümkünse)")
    total_amount: float = Field(description="Faturanın toplam tutarı (KDV dahil)")
    vat: float = Field(description="Faturadaki KDV/vergi tutarı")
    currency: str = Field(description="Para birimi (örn. TRY, USD, EUR) ISO 4217 kodu olarak")


def read_invoice_text(invoice_path: str) -> str:
    with open(invoice_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_invoice_data(invoice_text: str) -> InvoiceData:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY ortam değişkeni tanımlı değil.")

    client = Groq(api_key=api_key)
    schema = InvoiceData.model_json_schema()

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen bir fatura analiz asistanısın. Sana verilen fatura metninden "
                    "şirket adı, tarih, toplam tutar, KDV ve para birimi bilgilerini "
                    "doğru ve eksiksiz şekilde ayıkla. Yanıtını, aşağıdaki JSON şemasına "
                    "tam olarak uyan ve başka hiçbir açıklama içermeyen bir JSON nesnesi "
                    f"olarak ver:\n{json.dumps(schema, ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": f"Bu faturadaki bilgileri ayıkla:\n\n{invoice_text}",
            },
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    try:
        return InvoiceData.model_validate_json(content)
    except ValidationError as exc:
        raise RuntimeError(f"Model çıktısı beklenen şemaya uymuyor: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bir fatura metninden yapılandırılmış veri ayıklar (Groq llama-3.3-70b-versatile)."
    )
    parser.add_argument("invoice_path", help="Fatura metni dosya yolu (.txt)")
    args = parser.parse_args()

    if not os.path.isfile(args.invoice_path):
        print(f"Hata: '{args.invoice_path}' bulunamadı.", file=sys.stderr)
        sys.exit(1)

    try:
        invoice_text = read_invoice_text(args.invoice_path)
        result = extract_invoice_data(invoice_text)
    except Exception as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
