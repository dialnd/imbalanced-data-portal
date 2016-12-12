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


    parser.add_argument("-w", "--workflow",
                        help="workflow id",
                        type=str,
                        required=True
                        )

    parser.add_argument("-c", "--current-dir",
                        help="the directory of this script",
                        type=str,
                        required=True
                        )

    parser.add_argument("--analysis",
                        help="whether to run the full analysis",
                        action="store_true",
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
                        help="parameters for the model",
                        type=str,
                        default=None
                        )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    start = timeit.default_timer()

    workflow_path = os.path.join(args.current_dir, "..", "workflows")
    skf = glob.glob(os.path.join(workflow_path, args.workflow, "fold*"))
    dest = "parameterization" if not args.analysis else ""
    outpath = os.path.join(workflow_path, args.workflow, dest)

    model = args.model
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

    for i in skf:
        path = os.path.join(workflow_path, i)
        train_feats = np.loadtxt(os.path.join(path, "train_feats.csv"), delimiter=',')
        train_labels = np.loadtxt(os.path.join(path, "train_labels.csv"), delimiter=',')
        test_feats = np.loadtxt(os.path.join(path, "test_feats.csv"), delimiter=',')
        test_labels = np.loadtxt(os.path.join(path, "test_labels.csv"), delimiter=',')

        model.fit(train_feats, train_labels)
        probas_ = model.predict_proba(test_feats)
        preds_ = model.predict(test_feats)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(test_labels, probas_[:, 1])

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        # Keep track of original y-values and corresponding predicted probabilities and values
        #y_original_values = np.concatenate((y_original_values,test_labels),axis=0)
        y_original_values.extend(test_labels)
        #y_prob = np.concatenate((y_prob,probas_[:, 1]),axis=0)
        y_prob.extend(probas_[:, 1])
        #y_pred = np.concatenate((y_pred,preds_),axis=0)
        y_pred.extend(preds_)
        #indexes = np.concatenate((indexes,test_labels),axis=0)
        indexes.extend(test_labels)

        if args.analysis is None:
            break


    # Compute TPR and AUROC
    mean_tpr /= len(skf) if args.analysis else 1
    mean_tpr[-1] = 1.0
    auroc = auc(mean_fpr, mean_tpr)

    # Compute precision,recall curve points and area under PR-curve
    precision, recall, thresholds = precision_recall_curve(y_original_values, y_prob)
    aupr = auc(recall, precision)

    # Store a flag for mispredictions
    errors = np.logical_xor(y_pred,y_original_values).astype(int)

    # Compute overall accuracy
    accuracy = 1 - (np.sum(errors)/float(len(errors)))

    # Store xy coordinates of ROC curve points
    roc_points = ''
    for x in zip(mean_fpr,mean_tpr):
        roc_points += ('[' + str("%.3f" % x[0]) + ',' + str("%.3f" % x[1]) + '],')
    roc_points = roc_points[:-1]

    # Store xy coordinates of PR curve points
    prc_points = ''
    for x in zip(recall,precision):
        prc_points += ('[' + str("%.3f" % x[0]) + ',' + str("%.3f" % x[1]) + '],')
    prc_points = prc_points[:-1]

    # Store confusion matrix as a string with values separated by commas
    confusion_matrix = str(confusion_matrix(y_original_values, y_pred).tolist()).replace(']',"").replace('[',"")

    # Store a list of the numeric values returned by classification_report()
    clf_report = re.sub(r'[^\d.]+', ', ', classification_report(y_original_values, y_pred))[5:-2]

    # Compute precision, recall and f1-score
    precision, recall, f1_score, support = precision_recall_fscore_support(y_original_values, y_pred)

    # Limit the number of instances saved to DB so report won't crash
    LAST_INDEX = 1000 if (len(indexes)>1000) else len(indexes)

    # Sort everything by instance number
    sorted_ix = np.argsort(indexes)
    print(sorted_ix)
    indexes = ','.join(str(e) for e in np.array(indexes)[sorted_ix][:LAST_INDEX].astype(int))
    y_original_values = ','.join(str(e) for e in np.array(y_original_values)[sorted_ix][:LAST_INDEX].astype(int))
    y_pred = ','.join(str(e) for e in np.array(y_pred)[sorted_ix][:LAST_INDEX].astype(int))
    errors = ','.join(str(e) for e in np.array(errors)[sorted_ix][:LAST_INDEX]).replace('0'," ").replace('1',"&#x2717;")
    y_prob = ','.join(str(e) for e in np.around(np.array(y_prob)[sorted_ix][:LAST_INDEX], decimals=4))

    stop = timeit.default_timer()

    results = {
        'auroc': auroc,
        'aupr': aupr,
        'roc_points': roc_points,
        'prc_points': prc_points,
        'confusion_matrix': confusion_matrix,
        'classification_report': clf_report,
        'indexes': indexes,
        'y_original_values': y_original_values,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'errors': errors,
        'accuracy': accuracy,
        'runtime': stop - start
    }

    with open(os.path.join(outpath, 'results.json'), 'w') as f:
        json.dump(results, f)
