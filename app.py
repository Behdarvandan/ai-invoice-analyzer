import json
import os

import boto3
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

MODEL = "llama-3.3-70b-versatile"

s3_client = boto3.client("s3")


class InvoiceData(BaseModel):
    company_name: str = Field(description="Name of the company that issued the invoice")
    date: str = Field(description="Invoice date (in YYYY-MM-DD format, if possible)")
    total_amount: float = Field(description="Total amount of the invoice (VAT included)")
    vat: float = Field(description="VAT/tax amount on the invoice")
    currency: str = Field(description="Currency (e.g. TRY, USD, EUR) as an ISO 4217 code")


def read_invoice_text_from_s3(bucket_name: str, object_key: str) -> str:
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    raw_bytes = response["Body"].read()
    return raw_bytes.decode("utf-8")


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


def lambda_handler(event, context):
    record = event["Records"][0]
    bucket_name = record["s3"]["bucket"]["name"]
    object_key = record["s3"]["object"]["key"]

    try:
        invoice_text = read_invoice_text_from_s3(bucket_name, object_key)
        result = extract_invoice_data(invoice_text)
    except Exception as exc:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(exc)}, ensure_ascii=False),
        }

    return {
        "statusCode": 200,
        "body": json.dumps(result.model_dump(), ensure_ascii=False),
    }
