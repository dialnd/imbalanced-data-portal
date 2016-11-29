import pymysql
import sys
import json
import ast
import pandas as pd
import numpy as np
import timeit
import re
import random
import patsy
import os
import argparse
import glob

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import cross_validation
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import *


MODELS = {
    "decision_tree" : DecisionTreeClassifier(),
    "naive_bayes" : GaussianNB(),
    "knn" : KNeighborsClassifier(),
    "logistic_regression" : LogisticRegression(),
    "svc" : SVC(),
    "sgd" : SGDClassifier(),
    "random_forest" : RandomForestClassifier(),
    "extra_trees" : ExtraTreesClassifier(),
    "gradient_boost" : GradientBoostingClassifier()
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Perform preprocessing on a dataset")


    parser.add_argument("--workflow",
                        help="workflow id",
                        type=str,
                        required=True
                        )

    parser.add_argument("-c", "--current-dir",
                        help="the directory of this script"
                        type=str,
                        required=True
                        )

    parser.add_argument("--model",
                        help="name of the model to run",
                        type=str,
                        choices=[
                            "decision_tree",
                            "naive_bayes",
                            "knn",
                            "logistic_regression",
                            "svc",
                            "sgd",
                            "random_forest",
                            "extra_trees",
                            "gradient_boost"
                        ]
                        )

    parser.add_argument("--model-params",
                        help="number of principal components",
                        type=str,
                        )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    workflow_path = os.path.join(args.current_dir, "..", "workflows")
    skf = glob.glob(os.path.join(workflow_path, "sampling*"))

    model = None
    model_params = json.loads(args.model_params)

    if model == "sgd" and "shuffle" in model_params:
        model_params["shuffle"] = bool(model_params["shuffle"])

    if model_params is not None and len(model_params) > 0:
        model = MODELS[args.model].set_params(**model_params)
    else:
        model = MODELS[args.model]

    # Run 10-fold cross validation and compute AUROC
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    y_prob = []
    y_pred = []
    indexes = []
    y_original_values = []
    skf = cross_validation.StratifiedKFold(y, n_folds=10)

    for i in skf:
        path = os.path.join(workflow_path, i)
        train_feats = pd.read_csv(os.path.join(path, "train_feats.csv"))
        train_labels = pd.read_csv(os.path.join(path, "train_labels.csv"))
        test_feats = pd.read_csv(os.path.join(path, "test_feats.csv"))
        test_labels = pd.read_csv(os.path.join(path, "test_labels.csv"))

        clf.fit(X_train, y_train)
        probas_ = clf.predict_proba(X[test_index])
        preds_ = clf.predict(X[test_index])

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:, 1])

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        # Keep track of original y-values and corresponding predicted probabilities and values
        y_original_values = np.concatenate((y_original_values,y[test_index]),axis=0)
        y_prob = np.concatenate((y_prob,probas_[:, 1]),axis=0)
        y_pred = np.concatenate((y_pred,preds_),axis=0)
        indexes = np.concatenate((indexes,test_index),axis=0)


    # Compute TPR and AUROC
    mean_tpr /= len(skf)
    mean_tpr[-1] = 1.0
    auroc = auc(mean_fpr, mean_tpr)

    # Compute precision,recall curve points and area under PR-curve
    precision, recall, thresholds = precision_recall_curve(y_original_values, y_prob)
    aupr = auc(recall, precision)

    # Store a flag for mispredictions
    errors = np.logical_xor(y_pred,y_original_values).astype(int)

    # Compute overall accuracy
    accuracy = 1- (np.sum(errors)/float(len(errors)))
