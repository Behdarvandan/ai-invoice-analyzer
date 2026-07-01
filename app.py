import argparse
import json
import os
import sys

from groq import Groq
from pydantic import BaseModel, Field, ValidationError

MODEL = "llama-3.3-70b-versatile"


class InvoiceData(BaseModel):
    company_name: str = Field(description="Name of the company that issued the invoice")
    date: str = Field(description="Invoice date (in YYYY-MM-DD format, if possible)")
    total_amount: float = Field(description="Total amount of the invoice (VAT included)")
    vat: float = Field(description="VAT/tax amount on the invoice")
    currency: str = Field(description="Currency (e.g. TRY, USD, EUR) as an ISO 4217 code")


def read_invoice_text(invoice_path: str) -> str:
    with open(invoice_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_invoice_data(invoice_text: str) -> InvoiceData:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
    schema = InvoiceData.model_json_schema()

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an invoice analysis assistant. From the invoice text given to you, "
                    "extract the company name, date, total amount, VAT, and currency "
                    "accurately and completely. Return your answer as a JSON object that "
                    "conforms exactly to the following JSON schema and contains no other "
                    f"explanation:\n{json.dumps(schema, ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": f"Extract the information from this invoice:\n\n{invoice_text}",
            },
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    try:
        return InvoiceData.model_validate_json(content)
    except ValidationError as exc:
        raise RuntimeError(f"Model output does not match the expected schema: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extracts structured data from invoice text (Groq llama-3.3-70b-versatile)."
    )
    parser.add_argument("invoice_path", help="Path to the invoice text file (.txt)")
    args = parser.parse_args()

    if not os.path.isfile(args.invoice_path):
        print(f"Error: '{args.invoice_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        invoice_text = read_invoice_text(args.invoice_path)
        result = extract_invoice_data(invoice_text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
