from flask import Flask

from train_model import get_model

cls, model_info = get_model()
from api import api, ns, api_blueprint

from app import routes

flask_app = Flask(__name__)

api.init_app(api_blueprint)
api.add_namespace(ns)
flask_app.register_blueprint(api_blueprint)
