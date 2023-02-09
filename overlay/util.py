# read the data into a pandas DataFrame
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random
from loguru import logger
import pickle

randoms = []
# with open('overlay/data/gis_joins.json') as in_file:
#     gis_joins = json.load(in_file)
# print(f'gis_joins: {len(gis_joins)}')

path = 'overlay/data/temp_data.csv'
col_names = ['f1', 'f2', 'f3', 'f4', 'f5',
             'f6', 'f7', 'f8', 'label']
pima = pd.read_csv(path, header=None, names=col_names)

# define X and y
feature_cols = ['f1', 'f5', 'f6', 'f8']
X = pima[feature_cols]
y = pima.label


class Result:
    def __init__(self, gis_join_: str):
        self.gis_join = gis_join_
        self.precisions = []
        self.recalls = []
        self.auc_of_roc = None
        self.x_coordinates = []
        self.y_coordinates = []

    def to_json_string(self):
        json_result = {"auc_of_roc": self.auc_of_roc}
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        json_result["roc_graph"] = \
            {
                "x_coordinates": list(self.x_coordinates),
                "y_coordinates": list(self.y_coordinates)
            }
        for t, pos in zip(thresholds, range(len(thresholds))):
            json_result[str(t)] = \
                {
                    "precision": self.precisions[pos],
                    "recall": self.recalls[pos]
                }
        return json_result


# limit = len(gis_joins)
limit = 3088


def generate(gis_join: str):
    gis_join_result = Result(gis_join)
    random_state = random.randint(0, limit)
    while random_state in randoms:
        random_state = random.randint(0, limit)
    randoms.append(random_state)

    logger.debug(f'Random state: {random_state}')

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        # split X and y into training and testing sets
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, random_state=random_state)

        # train a logistic regression model on the training set

        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)

        # make class predictions for the testing set
        y_pred_class = model.predict(X_test)
        y_pred = (model.predict_proba(X_test)[:, 1] >= t).astype(int)

        precision = metrics.precision_score(y_test, y_pred, zero_division=0)
        precision = round(precision, 4)
        recall = metrics.recall_score(y_test, y_pred, zero_division=0)
        recall = round(recall, 4)
        # print(f'Threshold: {t}')
        # print(f'Precision: {precision}')
        # print(f'Recall: {recall}')
        gis_join_result.precisions.append(precision)
        gis_join_result.recalls.append(recall)

        # confusion matrix
        # confusion = metrics.confusion_matrix(y_test, y_pred_class)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        # print(f'fpr: {fpr}')
        # print(f'tpr: {tpr}')
        roc_auc_score = metrics.roc_auc_score(y_test, y_pred)
        roc_auc_score = round(roc_auc_score, 4)
        gis_join_result.auc_of_roc = roc_auc_score
        gis_join_result.x_coordinates = fpr
        gis_join_result.y_coordinates = tpr
        # print(f'auc_of_roc: {gis_join_result.auc_of_roc}')

    return json.dumps(gis_join_result.to_json_string())
