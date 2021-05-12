import json
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

from config import DATA_PATH, MODEL_PATH, MODEL_INFO_PATH


model, model_info = None, None


def get_model():
    global model, model_info
    if model is not None:
        return model, model_info
    with open(MODEL_PATH, mode='rb') as in_:
        model = pkl.load(in_)
    with open(MODEL_INFO_PATH, encoding='utf8') as in_:
        model_info = json.load(in_)

    return model, model_info


def train_model():
    data_set = pd.read_csv(DATA_PATH)
    target_col = 'target'
    feature_cols = [col for col in data_set.columns if col != target_col]
    X_train, X_test, y_train, y_test = train_test_split(data_set[feature_cols], data_set[target_col],
                                                        stratify=data_set[target_col], test_size=0.2)
    cls = GradientBoostingClassifier()
    cls.fit(X_train, y_train)
    with open(MODEL_PATH, mode='wb') as out:
        pkl.dump(cls, out)
    prediction = cls.predict(X_test)
    score = float(f1_score(y_test, prediction))
    model_info = {'model_name': 'Gradient Boosting', 'f1-score': score, 'required_cols': feature_cols}
    print(score, flush=True)
    with open(MODEL_INFO_PATH, encoding='utf8', mode='w') as out:
        json.dump(model_info, out)


if __name__ == '__main__':
    train_model()
