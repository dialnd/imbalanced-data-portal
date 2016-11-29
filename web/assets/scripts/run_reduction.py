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

from sklearn import decomposition


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

    parser.add_argument("--reduce",
                        help="pass to perform dimensionality reduction",
                        action="store_true"
                        )

    parser.add_argument("-n", "--n-components",
                        help="number of principal components",
                        type=int,
                        default=5
                        )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    workflow_path = os.path.join(args.current_dir, "..", "workflows", args.workflow)

    X = pd.read_csv(os.path.join(workflow_path, "preprocessing", "data.csv")).values

    # Apply PCA
    if args.reduce:
        estimator = decomposition.PCA(n_components=args.n_components)
        X = estimator.fit_transform(X)

    out_path = os.path.join(workflow_path, "reduction")
    if not os.path.isdir(out_path)
        os.mkdir(out_path)
    df.to_csv(os.path.join(os.path.join(out_path, "data.csv")))

    graph_data = {
        "boxplot": {},
        "histogram": {},
        "frequency": {},
        "odds": {}
    }

    features = []
    for i in range(0, n_features):
        features.append("PC{}".format(i + 1))

    boxplot = {}
    for
