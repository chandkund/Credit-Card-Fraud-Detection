# src/hyperparameter.py

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def tune_model(X_train, y_train):

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1]
    }

    grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss'),
        param_grid,
        scoring='roc_auc',
        cv=3
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_