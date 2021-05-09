from flask import request
from flask_restx import Resource


from train_model import get_model
from api import api, ns, feature_row, information, prediction


@ns.route('/predict')
class ModelPredict(Resource):
    @api.expect(feature_row, validate=True)
    @api.marshal_with(prediction)
    def post(self):
        """Returns model prediction"""
        feature_dict = request.get_json()
        cls, model_info = get_model()
        return {'prediction': int(cls.predict([[feature_dict[col] for col in model_info['required_cols']]])[0])}


@ns.route('/model')
class ModelInfo(Resource):
    @api.marshal_with(information)
    def get(self):
        """Returns model information"""
        _, model_info = get_model()
        return model_info
