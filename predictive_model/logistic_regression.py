import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def get_fit_lr_model(x_train: pd.DataFrame, y_train: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression model using the provided features and targets"""

    model = LogisticRegression(solver="lbfgs", max_iter=250)
    model.fit(x_train, y_train)
    return model


def show_lr_results(trained_model: LogisticRegression, x_test: pd.DataFrame, y_test: pd.DataFrame):
    """Display a variety of results describing the performance of the model against the provided testing data"""

    predicted_outcomes = trained_model.predict(x_test)
    predicted_probabilities = trained_model.predict_proba(x_test)
    score = trained_model.score(x_test, y_test)

    print(predicted_outcomes)
    print(predicted_probabilities)
    print(score)
    print(confusion_matrix(y_test, predicted_outcomes))
    print(classification_report(y_test, predicted_outcomes))
