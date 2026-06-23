"""
Tests for the global upload size ceiling (MAX_CONTENT_LENGTH).

Guards against memory exhaustion from oversized uploads: Werkzeug rejects a
request whose body exceeds MAX_CONTENT_LENGTH with 413 before reading it.
"""
import pytest

flask = pytest.importorskip("flask")

import config
from flask import Flask, request


def test_ceiling_is_above_largest_legitimate_upload():
    # Must never reject a legitimate maximum-size PDF statement.
    pdf_limit = pytest.importorskip("ocr.service").MAX_PDF_BYTES
    assert config.MAX_UPLOAD_BYTES >= pdf_limit
    # And it is a sane, finite positive value.
    assert config.MAX_UPLOAD_BYTES > 0


def test_max_content_length_rejects_oversized_body():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100  # tiny ceiling for the test

    @app.route("/echo", methods=["POST"])
    def echo():
        request.get_data()  # accessing the body triggers the ceiling check
        return "ok"

    client = app.test_client()
    # Under the ceiling -> accepted.
    assert client.post("/echo", data=b"x" * 50).status_code == 200
    # Over the ceiling -> 413 Request Entity Too Large (body not processed).
    assert client.post("/echo", data=b"x" * 500).status_code == 413


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
