"""
Gateway E2E smoke for Rust-backed OCR.

Start the proxy with:

litellm --config tests/e2e/gateway/litellm-config.yml --port 4000
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import BaseModel, Field

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


class ModelInfoDetail(BaseModel):
    id: str | None = None


class ModelInfoEntry(BaseModel):
    model_name: str | None = None
    model_info: ModelInfoDetail = Field(default_factory=ModelInfoDetail)


class ModelInfoResponse(BaseModel):
    data: list[ModelInfoEntry] = Field(default_factory=list)


def _parse_model_info(response: httpx.Response) -> ModelInfoResponse:
    assert response.status_code == 200, response.text
    return ModelInfoResponse.model_validate(response.json())


@dataclass(frozen=True)
class OcrGateway:
    base_url: str
    master_key: str

    def model_names(self) -> set[str]:
        with self._client() as client:
            response = client.get(f"{self.base_url.rstrip('/')}/model/info")
        parsed = _parse_model_info(response)
        return {
            entry.model_name for entry in parsed.data if entry.model_name is not None
        }

    def ocr(self, model: str, document: dict[str, str]) -> httpx.Response:
        with httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120"))
        ) as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/v1/ocr",
                headers={"Authorization": f"Bearer {self.master_key}"},
                json={"model": model, "document": document},
            )

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=float(os.getenv("E2E_REQUEST_TIMEOUT", "120")),
            headers={"Authorization": f"Bearer {self.master_key}"},
        )

    def create_model(
        self, model_name: str, litellm_params: dict[str, str]
    ) -> httpx.Response:
        with self._client() as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/model/new",
                json={"model_name": model_name, "litellm_params": litellm_params},
            )

    def delete_model(self, model_id: str) -> httpx.Response:
        with self._client() as client:
            return client.post(
                f"{self.base_url.rstrip('/')}/model/delete",
                json={"id": model_id},
            )

    def model_id(self, model_name: str) -> str | None:
        with self._client() as client:
            response = client.get(f"{self.base_url.rstrip('/')}/model/info")
        parsed = _parse_model_info(response)
        for entry in parsed.data:
            if entry.model_name == model_name:
                return entry.model_info.id
        return None

    def wait_for_model(self, model_name: str, attempts: int = 20) -> None:
        for _ in range(attempts):
            if model_name in self.model_names():
                return
            time.sleep(1)
        raise AssertionError(
            f"{model_name} did not appear on /model/info within {attempts}s"
        )

    def wait_for_model_absent(self, model_name: str, attempts: int = 20) -> None:
        for _ in range(attempts):
            if model_name not in self.model_names():
                return
            time.sleep(1)
        raise AssertionError(
            f"{model_name} still present on /model/info after {attempts}s"
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


class TestRustOcrDynamicDeployment:
    def test_os_environ_api_key_deployment_lifecycle(
        self, resources: OcrResources
    ) -> None:
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("Set MISTRAL_API_KEY on the proxy for the live OCR lifecycle")

        gateway = resources.gateway
        model_name = f"rust-ocr-env-e2e-{uuid.uuid4().hex[:8]}"

        create = gateway.create_model(
            model_name=model_name,
            litellm_params={
                "model": "mistral/mistral-ocr-latest",
                "api_key": "os.environ/MISTRAL_API_KEY",
            },
        )
        assert create.status_code == 200, create.text

        try:
            gateway.wait_for_model(model_name)

            response = gateway.ocr(
                model_name,
                {"type": "document_url", "document_url": TEST_PDF_URL},
            )
            assert response.status_code == 200, response.text
            _assert_ocr_response_shape(response.json())
            assert "os.environ/MISTRAL_API_KEY" not in response.text
        finally:
            deployed_id = gateway.model_id(model_name)
            if deployed_id is not None:
                delete = gateway.delete_model(deployed_id)
                assert delete.status_code == 200, delete.text
                gateway.wait_for_model_absent(model_name)
