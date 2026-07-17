"""Regenerate the deterministic three-page invoice fixture used by the Azure
Document Intelligence gateway E2E.

The output is a minimal, self-contained PDF built from raw bytes so it depends
on no third-party writer and produces byte-identical output on every run. Each
page draws a visible table grid plus form text including a key-value pair
(``Invoice Number: INV-123``) so a live layout model has real tables and
key-value pairs to extract.

Run from the repo root:

    python tests/e2e/gateway/fixtures/generate_invoice_fixture.py
"""

from __future__ import annotations

from pathlib import Path

INVOICE_NUMBER = "INV-123"
OUTPUT = Path(__file__).with_name("invoice_three_page.pdf")

_HEADER_LINES = (
    (f"Invoice Number: {INVOICE_NUMBER}", 690),
    ("Invoice Date: 2026-01-15", 672),
    ("Bill To: Test Buyer", 654),
    ("Total Due: $1,250.00", 636),
)
_TABLE_ROWS_Y = (560.0, 530.0, 500.0, 470.0, 440.0)
_TABLE_COLS_X = (72.0, 300.0, 420.0, 540.0)
_TABLE_CELLS = (
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


def _content_stream() -> bytes:
    title = (
        "BT",
        "/F1 18 Tf",
        "1 0 0 1 72 720 Tm",
        "(Sample Invoice) Tj",
        "ET",
    )
    headers = tuple(
        op
        for text, y in _HEADER_LINES
        for op in ("BT", "/F1 12 Tf", f"1 0 0 1 72 {y} Tm", f"({text}) Tj", "ET")
    )
    grid = (
        "0.5 w",
        *(f"72 {y} m 540 {y} l S" for y in _TABLE_ROWS_Y),
        *(
            f"{x} {_TABLE_ROWS_Y[0]} m {x} {_TABLE_ROWS_Y[-1]} l S"
            for x in _TABLE_COLS_X
        ),
    )
    cells = tuple(
        op
        for text, x, y in _TABLE_CELLS
        for op in ("BT", "/F1 10 Tf", f"1 0 0 1 {x} {y} Tm", f"({text}) Tj", "ET")
    )
    return "\n".join((*title, *headers, *grid, *cells)).encode("latin-1")


def _build_pdf() -> bytes:
    content = _content_stream()
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(index).encode("ascii") + b" 0 obj\n" + body + b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    return bytes(out)


if __name__ == "__main__":
    OUTPUT.write_bytes(_build_pdf())
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")  # noqa: T201
