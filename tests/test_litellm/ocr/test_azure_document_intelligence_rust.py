from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.loader import native_bridge_available

pytestmark = pytest.mark.skipif(
    not native_bridge_available(),
    reason="native Rust OCR bridge is not built",
)

DATA_URI_PDF = "data:application/pdf;base64,JVBERi0xLjQK"

SUCCEEDED_BODY: dict[str, object] = {
    "status": "succeeded",
    "analyzeResult": {
        "content": "full document text",
        "pages": [
            {"pageNumber": 1, "lines": [{"content": "first"}]},
            {"pageNumber": 3, "lines": [{"content": "third"}]},
        ],
        "tables": [{"rowCount": 2, "columnCount": 3}],
        "keyValuePairs": [{"key": {"content": "Total"}, "value": {"content": "42"}}],
    },
}


class _DocIntelHandler(BaseHTTPRequestHandler):
    analyze_query: str = ""
    analyze_status: int = 202
    error_body: dict[str, object] = {}

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        type(self).analyze_query = self.path.split("?", 1)[-1] if "?" in self.path else ""
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        if self.analyze_status != 202:
            body = json.dumps(self.error_body).encode()
            self.send_response(self.analyze_status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        host = self.headers.get("host")
        self.send_response(202)
        self.send_header("operation-location", f"http://{host}/operations/1")
        self.send_header("content-length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        body = json.dumps(SUCCEEDED_BODY).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server:
    def __init__(self, analyze_status: int = 202, error_body: dict[str, object] | None = None):
        _DocIntelHandler.analyze_status = analyze_status
        _DocIntelHandler.error_body = error_body or {}
        _DocIntelHandler.analyze_query = ""
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DocIntelHandler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"


MODEL = "azure_ai/doc-intelligence/prebuilt-layout"
DOCUMENT: dict[str, str] = {"type": "document_url", "document_url": DATA_URI_PDF}
_UNSET = object()


def _ocr(api_base: str, pages: object = _UNSET) -> OCRResponse:
    extra = {} if pages is _UNSET else {"pages": pages}
    result = litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="test-key",
        api_base=api_base,
        **extra,
    )
    assert isinstance(result, OCRResponse)
    return result


class TestAzureDocumentIntelligenceRustParity:
    def test_zero_based_pages_are_normalized_to_provider_pages(self) -> None:
        with _Server() as server:
            _ocr(server.base_url, pages=[2, 0, 2])
        assert "pages=1,3" in _DocIntelHandler.analyze_query

    def test_single_page_selection_is_supported(self) -> None:
        with _Server() as server:
            _ocr(server.base_url, pages=[0])
        assert "pages=1" in _DocIntelHandler.analyze_query

    def test_response_preserves_content_tables_and_key_value_pairs(self) -> None:
        with _Server() as server:
            response = _ocr(server.base_url)
        dumped = response.model_dump()
        assert dumped["content"] == "full document text"
        assert dumped["tables"] == [{"rowCount": 2, "columnCount": 3}]
        assert dumped["keyValuePairs"] == [{"key": {"content": "Total"}, "value": {"content": "42"}}]

    def test_negative_pages_raise_bad_request(self) -> None:
        with pytest.raises(litellm.BadRequestError):
            _ocr("https://acct.cognitiveservices.azure.com", pages=[0, -1])

    def test_malformed_pages_raise_bad_request(self) -> None:
        with pytest.raises(litellm.BadRequestError):
            _ocr("https://acct.cognitiveservices.azure.com", pages="1-,foo")

    def test_provider_not_found_status_is_preserved(self) -> None:
        with _Server(analyze_status=404, error_body={"error": {"code": "NotFound"}}) as server:
            with pytest.raises(litellm.NotFoundError):
                _ocr(server.base_url)

    def test_provider_bad_request_status_is_preserved(self) -> None:
        with _Server(analyze_status=400, error_body={"error": {"code": "InvalidArgument"}}) as server:
            with pytest.raises(litellm.BadRequestError):
                _ocr(server.base_url)
