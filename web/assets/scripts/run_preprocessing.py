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

import matplotlib.pyplot as plt

from sklearn import preprocessing
from scipy import stats


class WorkflowExistsException(Exception):
    pass


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Perform preprocessing on a dataset")

    parser.add_argument("--dataset",
                        help="dataset filename",
                        type=str,
                        required=True
                        )

    parser.add_argument("--workflow",
                        help="workflow id",
                        type=str,
                        required=True
                        )

    parser.add_argument("-c", "--current-dir",
                        help="the directory of this script",
                        type=str,
                        required=True
                        )

    parser.add_argument("--missing-data",
                        help="pass to perform outlier detection",
                        type=str,
                        choices=["default", "average", "interpolation"],
                        required=True
                        )

    parser.add_argument("--outlier-detection",
                        help="pass to perform outlier detection",
                        action="store_true"
                        )

    parser.add_argument("--standardization",
                        help="pass to perform standardization",
                        action="store_true"
                        )

    parser.add_argument("--normalization",
                        help="pass to perform normalization",
                        action="store_true"
                        )

    parser.add_argument("--norm-method",
                        help="normalization method",
                        type=str,
                        choices=["l1", "l2"]
                        )

    parser.add_argument("--binarization",
                        help="pass to perform binarization",
                        action="store_true"
                        )

    parser.add_argument("--binarization-threshold",
                        help="pass to perform binarization",
                        type=float,
                        default=1.0
                        )

    return parser.parse_args(args)


def setup_workflow_dir(workflow_path, workflow_id):
    if not os.path.isdir(workflow_path):
        os.mkdir(workflow_path)
    if not os.path.isdir(os.path.join(workflow_path, workflow_id)):
        os.mkdir(os.path.join(workflow_path, workflow_id))


if __name__ == "__main__":
    args = parse_args()
    args.current_dir = os.path.abspath(args.current_dir)
    dataset_path = os.path.abspath(os.path.join(args.current_dir, "..", "datasets"))
    workflow_path = os.path.abspath(os.path.join(args.current_dir, "..", "workflows"))

    setup_workflow_dir(workflow_path, args.workflow)
    workflow_path = os.path.join(workflow_path, args.workflow)
    out_path = os.path.join(workflow_path, "preprocessing")

    start = timeit.default_timer()

    X = pd.read_csv(os.path.join(dataset_path, args.dataset))

    # Fill in missing values
    if args.missing_data == 'default':
        X = X.fillna(-1)
    elif args.missing_data == 'average':
        X = X.fillna(X.mean())
    elif args.missing_data == 'interpolation':
        X = X.interpolate()
    else:
        print("Invalid option for 'missing_data'")
        sys.exit(1)

    y = np.array([i for i in X.values[:, -1]]).astype(np.int8)

    np.savetxt(os.path.join(workflow_path, "classes.csv"), y, delimiter=',')
    del y

    s = ' + '.join(X.columns) + ' -1'
    cols = X.columns[:-1]
    # Note: The encoding below could create very large dataframes for datasets with many categorical features.
    #X = patsy.dmatrix(s, X, return_type='dataframe').values[:,:-1]
    X = X.values[:, : -1].astype(np.float64)
    outlier_idx = (np.abs(stats.zscore(X)) >= 3).all(axis=1)


    # Remove outliers
    # TODO: Move this to traning step of K-fold so as to keep test untouched?
    if args.outlier_detection:
        non_outlier_idx = (np.abs(stats.zscore(X)) < 3).all(axis=1)
        X = X[non_outlier_idx]

    # Perform standardization. TODO: Should this come before encoding?
    if args.standardization:
        X = preprocessing.scale(X)

    # Perform normalization. TODO: Check to make sure order doesn't matter
    if args.normalization:
        X = preprocessing.normalize(X, norm=args.norm_method)

    # Perform binarization. TODO: Check to make sure order doesn't matter
    if args.binarization:
        binarizer = preprocessing.Binarizer(threshold=args.binarization_threshold).fit(X)
        X = binarizer.transform(X)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    np.savetxt(os.path.join(out_path, "data.csv"), X, delimiter=',')

    graph_data = {
        "boxplot": [],
        "histogram": [],
        "frequency": [],
        "odds": []
    }

    print(X)

    for i in range(len(cols)):
        boxplot = {}
        histogram = {}
        frequency = {}
        odds = {}

        col = cols[i]
        data = X[:, i]

        outliers = [[1, j] for j in data[outlier_idx]]

        boxplot = {
            'chart': {
                'type': "boxplot"
            },
            'title': {
                'text': "Distribution of {}".format(col)
            },
            'legend': {
                'enabled': False
            },
            'credits': {
                'enabled': False
            },
            'exporting': {
                'enabled': False
            },
            'yAxis': {
                'title': col
            },
            'series': [
                {
                    'name': 'Observations',
                    'data': [
                        [np.min(data), np.percentile(data, 25), np.median(data), np.percentile(data, 75), np.max(data)]
                    ]
                },
                {
                    'name': 'Outliers',
                    'data': outliers,
                    'type': 'scatter',
                    'tooltip': {
                        'pointFormat': 'Observation: {{point.y}}'
                    }
                }
            ]
        }

        bin_data, bin_edges = np.histogram(data, bins=20)
        width = ((bin_edges[0] + bin_edges[1]) / 2) / 2
        series = [[bin_edges[j] + width, float(bin_data[j])] for j in range(len(bin_data))]

        histogram = {
            'chart': {
                'type': 'column'
            },
            'title': {
                'text': "Distribution of {}".format(col)
            },
            'legend': {
                'enabled': False
            },
            'credits': {
              'enabled': False
            },
            'exporting': {
              'enabled': False
            },
            'tooltip': {},
            'xAxis': {
                'title': col,
                'crosshair': True
            },
            'yAxis': {
                'min': 0,
                'title': "Count"
            },
            'plotOptions': {
                'column': {
                    'pointPadding': 0,
                    'borderWidth': 0,
                    'groupPadding': 0,
                    'shadow': False
                }
            },
            'series': [
                {
                    'type': 'column',
                    'name': "Distribution",
                    'data': series
                }
            ]
        }

        frequency = {

        }

        graph_data['boxplot'].append(boxplot)
        graph_data['histogram'].append(histogram)

    stop = timeit.default_timer()
    graph_data["runtime"] = stop - start
    with open(os.path.join(out_path, 'graph_data.json'), 'w') as f:
        json.dump(graph_data, f)
