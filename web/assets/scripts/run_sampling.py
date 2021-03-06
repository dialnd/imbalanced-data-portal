import sys
import pandas as pd
import numpy as np
import timeit
import re
import random
import argparse
import os
import json

from sklearn import preprocessing, decomposition, cross_validation
from sklearn.neighbors import NearestNeighbors
from scipy import stats

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Perform under- and oversampling on data.")

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

    parser.add_argument("--undersampling",
                        help="pass to perform undersampling",
                        action="store_true"
                        )

    parser.add_argument("--undersampling-rate",
                        help="undersampling rate",
                        type=float,
                        default=1.0
                        )

    parser.add_argument("--oversampling",
                        help="pass to perform oversampling",
                        action="store_true"
                        )

    parser.add_argument("--oversampling-percentage",
                        help="oversampling rate",
                        type=float,
                        default=1.0
                        )

    return parser.parse_args(args)


def undersample(x, y, ix, subsampling_rate=1.0):
    """ Data undersampling.

    This function takes in a list or array indexes that will be used for training
    and it performs subsampling in the majority class (c == 0) to enforce a certain ratio
    between the two classes
    Parameters
    ----------
    x : np.ndarray
        The entire dataset as a ndarray
    y : np.ndarray
        The labels
    ix : np.ndarray
        The array indexes for the instances that will be used for training
    subsampling_rate : float
        The desired percentage of majority instances in the subsample

    Returns
    --------
    np.ndarray
        The new list of array indexes to be used for training
    """

    # Get indexes of instances that belong to classes 0 and 1
    indexes_0 = [item for item in ix if y[item] == 0]
    indexes_1 = [item for item in ix if y[item] == 1]

    # Determine how large the new majority class set should be
    sample_length = int(len(indexes_0)*subsampling_rate)
    sample_indexes = random.sample(indexes_0, sample_length) + indexes_1

    return sample_indexes


def SMOTE(T, N, k, h = 1.0):
    """ Synthetic minority oversampling.
    Returns (N/100) * n_minority_samples synthetic minority samples.
    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples:
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours.
    Returns
    -------
    S : Synthetic samples. array,
        shape = [(N/100) * n_minority_samples, n_features].
    """
    n_minority_samples, n_features = T.shape

    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    N = int(N/100)
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)

    #Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in range(N):
            nn_index = random.choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it
            while nn_index == i:
                nn_index = random.choice(nn[0])

            dif = T[nn_index] - T[i]
            gap = np.random.uniform(low = 0.0, high = h)
            S[n + i * N, :] = T[i,:] + gap * dif[:]

    return S


if __name__ == "__main__":
    args = parse_args()

    start = timeit.default_timer()

    # Set up the path to the current workflow
    workflow_path = os.path.join(args.current_dir,
                                 "..",
                                 "workflows",
                                 args.workflow
                                 )

    # Load the dataset and split into features and labels
    X = np.loadtxt(os.path.join(workflow_path, "reduction", "data.csv"))
    y = np.loadtxt(os.path.join(workflow_path, "classes.csv")).astype(np.int32)

    # Set up cross-validation divisions
    skf = cross_validation.StratifiedKFold(y, n_folds=10)
    i = 0

    # Run under- and oversampling on each of the k folds

    graph_data = {}

    for train_index, test_index in skf:
        train_index = train_index[np.where(train_index < len(X))]
        test_index = test_index[np.where(test_index < len(X))]

        fold_data = {
            'boxplot': [],
            'histogram': [],
        }

        # Perform undersampling
        if args.undersampling:
            train_index = undersample(X,y,train_index,float(args.undersampling_rate)/100)

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        # Perform oversampling
        if args.oversampling:
            minority = X_train[np.where(y_train==1)]
            smotted = SMOTE(minority, args.oversampling_percentage, 5)
            X_train = np.vstack((X_train,smotted))
            y_train = np.append(y_train,np.ones(len(smotted),dtype=np.int32))

        # Save training and test sets for this fold in its own directory
        if os.path.isdir(workflow_path):
            foldpath = os.path.join(workflow_path, "fold{}".format(i))
            if not os.path.isdir(foldpath):
                os.mkdir(foldpath)
            np.savetxt(os.path.join(foldpath, "train_feats.csv"), X_train, delimiter=',')
            np.savetxt(os.path.join(foldpath, "train_labels.csv"), y_train, delimiter=',')
            np.savetxt(os.path.join(foldpath, "test_feats.csv"), X_test, delimiter=',')
            np.savetxt(os.path.join(foldpath, "test_labels.csv"), y_test, delimiter=',')

        for j in range(len(X_train[0])):
            data = X_train[:, j]

            boxplot = {}
            histogram = {}
            frequency = {}
            odd = {}

            boxplot = {
                'chart': {
                    'type': "boxplot"
                },
                'title': {
                    'text': "Distribution of Feature {}".format(j)
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
                    'title': "Feature {}".format(j)
                },
                'series': [
                    {
                        'name': 'Observations',
                        'data': [
                            [np.min(data), np.percentile(data, 25), np.median(data), np.percentile(data, 75), np.max(data)]
                        ]
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
                    'text': "Distribution of Feature {}".format(j)
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
                    'title': "Feature {}".format(j),
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

            fold_data['boxplot'].append(boxplot)
            fold_data['histogram'].append(histogram)

        graph_data["fold{}".format(i)] = fold_data
        i += 1

    stop = timeit.default_timer()
    graph_data["runtime"] = stop - start

    outpath = os.path.join(workflow_path, 'sampling')
    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    with open(os.path.join(outpath, 'graph_data.json'), 'w') as f:
        json.dump(graph_data, f)
