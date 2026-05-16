from flask import Blueprint

predictions = Blueprint('predictions', __name__, url_prefix='/predictions')

from . import routes
