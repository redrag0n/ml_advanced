import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

from config import DATA_PATH

cls = None
model_info = None


def get_model():
    global cls, model_info
    if cls is not None:
        return cls, model_info
    data_set = pd.read_csv(DATA_PATH)
    target_col = 'target'
    feature_cols = [col for col in data_set.columns if col != target_col]
    X_train, X_test, y_train, y_test = train_test_split(data_set[feature_cols], data_set[target_col],
                                                        stratify=data_set[target_col], test_size=0.2)
    cls = GradientBoostingClassifier()
    cls.fit(X_train, y_train)
    prediction = cls.predict(X_test)
    score = float(f1_score(y_test, prediction))
    model_info = {'model_name': 'Gradient Boosting', 'f1-score': score, 'required_cols': feature_cols}
    print(score, flush=True)

    return cls, model_info

