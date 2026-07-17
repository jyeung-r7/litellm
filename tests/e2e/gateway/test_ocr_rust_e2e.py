"""
Gateway E2E smoke for Rust-backed OCR.

Start the proxy with:

litellm --config tests/e2e/gateway/litellm-config.yml --port 4000
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pypdf
import pytest
import yaml
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

AZURE_DI_MODEL = "rust-ocr-azure-document-intelligence"
INVOICE_NUMBER = "INV-123"

TEST_PDF_URL = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm"
    "@d769e81c90d453240c61fc572cdb27fae06a89d0"
    "/tests/llm_translation/fixtures/dummy.pdf"
)
TEST_IMAGE_URL = (
    "https://cdn.jsdelivr.net/gh/BerriAI/litellm"
    "@d769e81c90d453240c61fc572cdb27fae06a89d0"
    "/tests/image_gen_tests/test_image.png"
)

RUST_OCR_GATEWAY_CASES = [
    pytest.param(
        "rust-ocr-mistral",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="mistral",
    ),
    pytest.param(
        "rust-ocr-azure-ai",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="azure_ai",
    ),
    pytest.param(
        "rust-ocr-azure-document-intelligence",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="azure_document_intelligence",
    ),
    pytest.param(
        "rust-ocr-vertex-mistral",
        {"type": "document_url", "document_url": TEST_PDF_URL},
        id="vertex_mistral",
    ),
    pytest.param(
        "rust-ocr-vertex-deepseek",
        {
            "type": "image_url",
            "image_url": os.getenv("RUST_OCR_IMAGE_URL", TEST_IMAGE_URL),
        },
        id="vertex_deepseek",
    ),
]

CONFIG_PATH = Path(__file__).with_name("litellm-config.yml")


@dataclass(frozen=True)
class OcrGateway:
    base_url: str
    master_key: str

    def model_names(self) -> set[str]:
        with httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))
        ) as client:
            response = client.get(
                f"{self.base_url.rstrip('/')}/model/info",
                headers={"Authorization": f"Bearer {self.master_key}"},
            )
        assert response.status_code == 200, response.text
        return {
            model["model_name"]
            for model in response.json().get("data", [])
            if "model_name" in model
        }

    def ocr(
        self,
        model: str,
        document: dict[str, str],
        pages: list[int] | str | None = None,
        features: list[str] | None = None,
    ) -> httpx.Response:
        body: dict[str, object] = {"model": model, "document": document}
        if pages is not None:
            body = {**body, "pages": pages}
        if features is not None:
            body = {**body, "features": features}
        with httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))
        ) as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/v1/ocr",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json=body,
            )


@dataclass(frozen=True)
class OcrResources:
    gateway: OcrGateway


@pytest.fixture
def resources() -> OcrResources:
    proxy_url = os.getenv("LITELLM_PROXY_URL")
    if not proxy_url:
        pytest.skip(
            "Start a Rust OCR proxy and set LITELLM_PROXY_URL, e.g. http://localhost:4000"
        )
    return OcrResources(
        gateway=OcrGateway(
            base_url=proxy_url,
            master_key=os.getenv("LITELLM_MASTER_KEY", "sk-1234"),
        )
    )


def _assert_ocr_response_shape(response_json: dict[str, Any]) -> None:
    assert response_json["object"] == "ocr"
    assert response_json["model"]
    assert isinstance(response_json["pages"], list)
    assert len(response_json["pages"]) > 0
    assert "index" in response_json["pages"][0]
    assert "markdown" in response_json["pages"][0]


def _invoice_page_content_stream() -> bytes:
    header_pairs = (
        (f"Invoice Number: {INVOICE_NUMBER}", 690),
        ("Invoice Date: 2026-01-15", 672),
        ("Bill To: Globex LLC", 654),
        ("Total Due: $1,250.00", 636),
    )
    table_rows_y = (560.0, 530.0, 500.0, 470.0, 440.0)
    table_cols_x = (72.0, 300.0, 420.0, 540.0)
    table_cells = (
        ("Description", 80, 566),
        ("Qty", 306, 566),
        ("Unit Price", 426, 566),
        ("Widget A", 80, 536),
        ("2", 306, 536),
        ("$100.00", 426, 536),
        ("Widget B", 80, 506),
        ("5", 306, 506),
        ("$150.00", 426, 506),
        ("Service Fee", 80, 476),
        ("1", 306, 476),
        ("$50.00", 426, 476),
    )

    title = (
        "BT",
        "/F1 18 Tf",
        "1 0 0 1 72 720 Tm",
        "(ACME Corporation Invoice) Tj",
        "ET",
    )
    headers = tuple(
        op
        for text, y in header_pairs
        for op in ("BT", "/F1 12 Tf", f"1 0 0 1 72 {y} Tm", f"({text}) Tj", "ET")
    )
    grid = (
        "0.5 w",
        *(f"72 {y} m 540 {y} l S" for y in table_rows_y),
        *(f"{x} {table_rows_y[0]} m {x} {table_rows_y[-1]} l S" for x in table_cols_x),
    )
    cells = tuple(
        op
        for text, x, y in table_cells
        for op in ("BT", "/F1 10 Tf", f"1 0 0 1 {x} {y} Tm", f"({text}) Tj", "ET")
    )
    return "\n".join((*title, *headers, *grid, *cells)).encode("latin-1")


def _add_invoice_page(writer: pypdf.PdfWriter) -> None:
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(_invoice_page_content_stream())
    page[NameObject("/Contents")] = writer._add_object(stream)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    page[NameObject("/MediaBox")] = ArrayObject(
        [NumberObject(0), NumberObject(0), NumberObject(612), NumberObject(792)]
    )


def _three_page_pdf_data_uri() -> str:
    writer = pypdf.PdfWriter()
    for _ in range(3):
        _add_invoice_page(writer)
    buffer = io.BytesIO()
    writer.write(buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


@pytest.fixture
def three_page_document() -> dict[str, str]:
    return {"type": "document_url", "document_url": _three_page_pdf_data_uri()}


def _require_azure_di(resources: OcrResources) -> None:
    if AZURE_DI_MODEL not in resources.gateway.model_names():
        pytest.skip(f"{AZURE_DI_MODEL} is not loaded on the gateway")


@pytest.mark.e2e
class TestAzureDocumentIntelligencePagesParity:
    def test_zero_based_pages_are_deduped_and_normalized(
        self, resources: OcrResources, three_page_document: dict[str, str]
    ) -> None:
        _require_azure_di(resources)
        response = resources.gateway.ocr(
            AZURE_DI_MODEL, three_page_document, pages=[2, 0, 2]
        )
        assert response.status_code == 200, response.text
        indices = [page["index"] for page in response.json()["pages"]]
        assert indices == [0, 2]

    def test_single_zero_based_page(
        self, resources: OcrResources, three_page_document: dict[str, str]
    ) -> None:
        _require_azure_di(resources)
        response = resources.gateway.ocr(AZURE_DI_MODEL, three_page_document, pages=[0])
        assert response.status_code == 200, response.text
        indices = [page["index"] for page in response.json()["pages"]]
        assert indices == [0]

    def test_out_of_range_page_returns_provider_bad_request(
        self, resources: OcrResources, three_page_document: dict[str, str]
    ) -> None:
        _require_azure_di(resources)
        response = resources.gateway.ocr(
            AZURE_DI_MODEL, three_page_document, pages=[99]
        )
        assert response.status_code == 400, response.text

    def test_full_document_preserves_content_and_extra_fields(
        self, resources: OcrResources, three_page_document: dict[str, str]
    ) -> None:
        _require_azure_di(resources)
        response = resources.gateway.ocr(
            AZURE_DI_MODEL, three_page_document, features=["keyValuePairs"]
        )
        assert response.status_code == 200, response.text
        body = response.json()
        _assert_ocr_response_shape(body)
        assert len(body["pages"]) == 3
        assert isinstance(body["content"], str) and body["content"]
        assert isinstance(body["tables"], list) and body["tables"]
        assert isinstance(body["keyValuePairs"], list) and body["keyValuePairs"]
        assert any(
            INVOICE_NUMBER in ((pair.get("value") or {}).get("content") or "")
            for pair in body["keyValuePairs"]
        )


class TestRustOcrGateway:
    def test_rust_ocr_models_are_on_gateway_config(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text())
        configured_models = {
            model_config["model_name"] for model_config in config["model_list"]
        }

        expected_models = {case.values[0] for case in RUST_OCR_GATEWAY_CASES}
        assert expected_models.issubset(configured_models)

    def test_running_gateway_loaded_rust_ocr_models(
        self, resources: OcrResources
    ) -> None:
        expected_models = {case.values[0] for case in RUST_OCR_GATEWAY_CASES}
        assert expected_models.issubset(resources.gateway.model_names())

    @pytest.mark.parametrize(("model", "document"), RUST_OCR_GATEWAY_CASES)
    def test_rust_ocr_model_gateway_response(
        self, resources: OcrResources, model: str, document: dict[str, str]
    ) -> None:
        response = resources.gateway.ocr(model, document)

        assert response.status_code == 200, response.text
        _assert_ocr_response_shape(response.json())
