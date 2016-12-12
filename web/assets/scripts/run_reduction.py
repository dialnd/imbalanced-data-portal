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
                        help="the directory of this script",
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

    start = timeit.default_timer()

    workflow_path = os.path.join(args.current_dir, "..", "workflows", args.workflow)

    X = pd.read_csv(os.path.join(workflow_path, "preprocessing", "data.csv")).values

    var_ratio = None

    if not args.reduce or args.n_components > len(X[0]):
        args.n_components = len(X[0])

    # Apply PCA
    estimator = decomposition.PCA(n_components=args.n_components)
    X = estimator.fit_transform(X)
    var_ratio = list(estimator.explained_variance_ratio_)

    out_path = os.path.join(workflow_path, "reduction")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    np.savetxt(os.path.join(out_path, "data.csv"), X)

    graph_data = {
        "scree": {},
        "scatter": {}
    }

    scree = {
        'chart': {
            'type': 'column'
        },
        'title': {
            'text': "Scree Plot of Principal Components"
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
            'title': "Principal Components",
            'categories': ['Component {}'.format(i + 1) for i in range(len(var_ratio))]
        },
        'yAxis': {
            'title': "Explained Variance (%)"
        },
        'plotOptions': {
            'column': {
                'pointPadding': 0.2,
                'borderWidth': 0
            }
        },
        'series': [{'name': 'PCA', 'data': var_ratio}]
    }

    scatterdata = [[i[0], i[1]] for i in X]
    scatter = {
        'chart': {
            'type': 'scatter',
            'zoomType': 'xy'
        },
        'title': {
            'text': "Scatter Plot of First Two Principal Components"
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
            'title': "Component 1"
        },
        'yAxis': {
            'title': "Component 2"
        },
        'plotOptions': {
            'scatter': {
                'marker': {
                    'radius': 5,
                    'states': {
                        'hover': {
                            'enabled': True,
                            'lineColor': 'rgb(100,100,100)'
                        }
                    }
                },
                'states': {
                    'hover': {
                        'marker': {
                            'enabled': False
                        }
                    }
                }
            }
        },
        'series': [
            {
                'name': "Components",
                'data': scatterdata
            }
        ]
    }

    graph_data['scree'] = scree
    graph_data['scatter'] = scatter

    stop = timeit.default_timer()
    graph_data["runtime"] = stop - start

    with open(os.path.join(out_path, 'graph_data.json'), 'w') as f:
        json.dump(graph_data, f)
