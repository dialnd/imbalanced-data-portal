import argparse
import glob
import json
import os
import re
import timeit

import numpy as np
import pandas as pd
import pymysql
from scipy import interp
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODELS = {
    "decision_tree" : DecisionTreeClassifier(),
    "extra_trees" : ExtraTreesClassifier(),
    "gradient_boost" : GradientBoostingClassifier(),
    "knn" : KNeighborsClassifier(),
    "logistic_regression" : LogisticRegression(),
    "naive_bayes" : GaussianNB(),
    "random_forest" : RandomForestClassifier(),
    "sgd" : SGDClassifier(),
    "svc" : SVC()
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Perform preprocessing on a dataset.")


    parser.add_argument("-w", "--workflow",
                        help="Workflow ID.",
                        type=str,
                        required=True
                        )

    parser.add_argument("-c", "--current-dir",
                        help="The directory of this script.",
                        type=str,
                        required=True
                        )

    parser.add_argument("--analysis",
                        help="Whether to run the full analysis.",
                        action="store_true",
                        )

    parser.add_argument("--model",
                        help="Name of the model to run.",
                        type=str,
                        choices=[
                            "decision_tree",
                            "extra_trees",
                            "gradient_boost",
                            "knn",
                            "logistic_regression",
                            "naive_bayes",
                            "random_forest",
                            "sgd",
                            "svc"
                        ]
                        )

    parser.add_argument("--model-params",
                        help="Parameters for the model.",
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
    model_params = {}
    with open(os.path.join(workflow_path, args.workflow, args.model_params), 'r') as f:
        model_params = json.load(f)

    if model == "decision_tree":
        model_params["max_features"] = float(model_params["max_features"])
        model_params["max_leaf_nodes"] = int(float(model_params["max_leaf_nodes"]))
        model_params["min_samples_split"] = float(model_params["min_samples_split"])
        model_params["min_samples_leaf"] = float(model_params["min_samples_leaf"])
        model_params["max_depth"] = float(model_params["max_depth"])

    if model == "sgd" and "shuffle" in model_params:
        model_params["shuffle"] = bool(model_params["shuffle"])

    if model_params is not None and len(model_params) > 0:
        model = MODELS[args.model].set_params(**model_params)
    else:
        model = MODELS[args.model]

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    y_test = []
    y_prob = []
    y_pred = []
    indexes = []

    # Run 10-fold cross-validation and compute AUROC.
    n_splits = 10
    skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        path = os.path.join(workflow_path, fold)
        X_train_i = np.loadtxt(os.path.join(path, "train_feats.csv"), delimiter=',')
        y_train_i = np.loadtxt(os.path.join(path, "train_labels.csv"), delimiter=',')

        X_test_i = np.loadtxt(os.path.join(path, "test_feats.csv"), delimiter=',')
        y_test_i = np.loadtxt(os.path.join(path, "test_labels.csv"), delimiter=',')

        model.fit(X_train_i, y_train_i)
        y_pred_i = model.predict(X_test_i)
        y_prob_i = np.array(model.predict_proba(X_test_i))

        # Compute ROC curve and ROC area for each class.
        fpr = dict()
        tpr = dict()
        pre = dict()
        rec = dict()
        roc_auc = dict()
        pr_auc = dict()
        if n_classes > 2:
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_i[:, i], y_pred_i[:, i])
                pre[i], rec[i], _ = precision_recall_curve(y_test_i[:, i], y_pred_i[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                pr_auc[i] = auc(pre[i], rec[i])

            # Compute micro-average ROC curve and ROC area.
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_i.ravel(), y_pred_i.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute micro-average P-R curve and P-R area.
            pre["micro"], rec["micro"], _ = \
                precision_recall_curve(y_test_i.ravel(), y_pred_i.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            pr_auc["micro"] = auc(pre["micro"], rec["micro"])

            # Compute macro-average ROC curve and ROC area.

            # First aggregate all false positive rates.
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points.
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC.
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Compute ROC curve and area under the curve.
            #fpr, tpr, thresholds = roc_curve(y_test_i, y_prob_i[:, 1])

            #mean_tpr += interp(mean_fpr, fpr, tpr)
            #mean_tpr[0] = 0.0

            # Define y-values and corresponding predicted probabilities and values.
            y_test.extend((y_test, y[test_idx]))
            y_prob.extend((y_prob, np.argmax(y_pred_i, axis=1)))
            y_pred.extend((y_pred, np.argmax(y_pred_i, axis=1)))
            indexes.extend((indexes, test_idx))
        else:
            # Compute ROC curve and area under the curve.
            fpr["micro"], tpr["micro"], _ = roc_curve(y[test_idx], y_prob_i[:, 1])

            mean_tpr += interp(mean_fpr, fpr["micro"], tpr["micro"])
            mean_tpr[0] = 0.0

            # Define y-values and corresponding predicted probabilities and values.
            y_test.extend(y[test_idx]))
            y_prob.extend((y_prob, y_prob_i[:, 1]))
            y_pred.extend((y_pred, y_pred_i))
            indexes.extend((indexes, test_idx))

        if args.analysis is None:
            break

    if n_classes == 2:
        # Compute TPR and AUROC
        mean_tpr /= n_splits
        mean_tpr[-1] = 1.0
        auroc = auc(mean_fpr, mean_tpr)

        # Compute precision-recall curve points and area under the PR-curve.
        pre["micro"], rec["micro"], _ = precision_recall_curve(y_test, y_prob)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        pr_auc["micro"] = auc(rec["micro"], pre["micro"])

    # Store a flag for mispredictions.
    errors = np.logical_xor(y_pred, y_test).astype(int)

    # Compute overall accuracy.
    accuracy = 1 - (np.sum(errors) / float(len(errors)))

    # Store x-y coordinates of ROC curve points.
    roc_points = ''
    for x in zip(fpr["micro"], tpr["micro"]):
        roc_points += ('[' + str('%.3f' % x[0]) + ',' + str('%.3f' % x[1]) + '],')
    roc_points = roc_points[:-1]

    # Store x-y coordinates of P-R curve points.
    prc_points = ''
    for x in zip(rec["micro"], pre["micro"]):
        prc_points += ('[' + str('%.3f' % x[0]) + ',' + str('%.3f' % x[1]) + '],')
    prc_points = prc_points[:-1]

    # Store confusion matrix as a string with comma-separated values.
    confusion_matrix = str(confusion_matrix(
        y_test, y_pred).tolist()).replace(']', '').replace('[', '')

    # Store a list of the numeric values returned by classification_report().
    clf_report = re.sub(
        r'[^\d.]+', ', ', classification_report(y_test, y_pred))[5:-2]

    # Compute precision, recall, and F1-score.
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_test, y_pred)

    # Limit number of instances saved to the database so report will finish.
    # TODO: Extend reporting to an arbitrary number of instances.
    LAST_INDEX = 1000 if (len(indexes) > 1000) else len(indexes)

    # Sort results by instance number.
    sorted_ix = np.argsort(indexes)
    indexes = ','.join(str(e) for e in indexes[sorted_ix][:LAST_INDEX].astype(int))
    y_test = ','.join(str(e) for e in y_test[sorted_ix][:LAST_INDEX].astype(int))
    y_pred = ','.join(str(e) for e in y_pred[sorted_ix][:LAST_INDEX].astype(int))
    errors = ','.join(str(e) for e in errors[sorted_ix][:LAST_INDEX]).replace(
        '0', ' ').replace('1', '&#x2717;')
    y_prob = ','.join(str(e) for e in np.around(y_prob[sorted_ix][:LAST_INDEX],
                                                decimals=4))

    stop = timeit.default_timer()

    results = {
        'auroc': roc_auc["micro"],
        'aupr': pr_auc["micro"],
        'roc_points': roc_points,
        'prc_points': prc_points,
        'confusion_matrix': confusion_matrix,
        'classification_report': clf_report,
        'indexes': indexes,
        'y_original_values': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'errors': errors,
        'accuracy': accuracy,
        'runtime': stop - start
    }

    if args.analysis:
        with open(os.path.join(outpath, 'results.json'), 'w') as f:
            json.dump(results, f)
    else:
        with open(os.path.join(outpath, 'parameterization', 'graph_data.json'), 'w') as f:
            json.dump(results, f)
