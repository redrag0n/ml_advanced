from flask import Blueprint
from flask_restx import Api, fields

from train_model import get_model

api = Api(version='1.0', title='ML API Example', validate=False)
ns = api.namespace('heart', description='Heart disease')
_, model_info = get_model()
feature_row = api.model('featues', {feat: fields.Float(required=True) for feat in model_info['required_cols']})

information = api.model('model information', {'model_name': fields.String(required=True),
                                              'f1-score': fields.Float(required=True),
                        'required_cols': fields.List(fields.String())})

prediction = api.model('prediction', {'prediction': fields.Integer()})

api_blueprint = Blueprint('api', __name__)