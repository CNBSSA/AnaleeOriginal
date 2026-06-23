"""OCR blueprint: extract transactions from receipt images via Claude vision."""
from flask import Blueprint

ocr = Blueprint('ocr', __name__, url_prefix='/ocr')

from . import routes  # noqa: E402,F401
