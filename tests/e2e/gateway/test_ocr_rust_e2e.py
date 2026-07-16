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

REPO_ROOT = Path(__file__).resolve().parents[3]
ONE_PAGE_FIXTURE = REPO_ROOT / "tests" / "llm_translation" / "fixtures" / "dummy.pdf"
AZURE_DI_MODEL = "rust-ocr-azure-document-intelligence"

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


def _three_page_pdf_data_uri() -> str:
    assert ONE_PAGE_FIXTURE.exists(), f"missing one-page fixture at {ONE_PAGE_FIXTURE}"
    reader = pypdf.PdfReader(str(ONE_PAGE_FIXTURE))
    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_page(reader.pages[0])
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
        assert isinstance(body["tables"], list)
        assert isinstance(body["keyValuePairs"], list)


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
