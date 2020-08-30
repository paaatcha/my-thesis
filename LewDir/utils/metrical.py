#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the metrics and the Dirichlet estimation for each set (hit, miss, and all).

If you find any bug, email me.

"""
import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
import dirichlet
import numpy as np
import pandas as pd
from raug.utils.common import insert_pred_col
from scipy.spatial.distance import cosine, euclidean, jensenshannon, mahalanobis, correlation
from scipy.stats import wasserstein_distance, energy_distance


def get_metric (x, y, name, alphas=None, ep=0.000001):
    """
    This function computes a metric according to the given name
    :param x: an array k-dim
    :param y: an array k-dim
    :param name: the metric name (see the if below)
    :param alphas: if you intend to use the Mahalanobis distance, set the Dirichlet distribution parameters (alphas)
    :param ep (float): an very small value to avoid log(0)
    :return: the distance
    """
    if name == "kld_div":
        return kld_div(y, x, ep)
    elif name == "euclidean":
        return euclidean(x,y)
    elif name == "wasserstein":
        return wasserstein_distance(x, y)
    elif name == "hellinger":
        return hellinger(x, y)
    elif name == "bhattcharyya":
        return bhattacharyya(x,y)
    elif name == "jensen-shannon":
        return jensenshannon(x, y)
    elif name == "correlation":
        return correlation(x, y)
    elif name == "cosine":
        return cosine(x, y)
    elif name == "energy":
        return energy_distance(x, y)
    elif name == "mahalanobis":
        if alphas is None:
            print ("You need to pass alpha to compute mahalanobis")
            raise ValueError
        else:
            return mahalanobis_dist(x, y, alphas)
    else:
        print("There is no distance called", type)
        raise ValueError


def entropy(x, ep=0.00001):
    """
    This function computes the Shannon entropy of a probability set
    :param x (numpy array): an array containing the probabilities
    :param ep (float): an very small value to avoid log(0)
    :return: the shannon entropy
    """
    ents = -np.sum(x * np.log2(x+ep))
    return ents


def kld_div(y, x, ep=0.000001):
    """
    This function computes the Kulback Leibler divergence between two set of probabilities
    :param x (numpy array): an array containing the probabilities
    :param y (numpy array): an array containing the probabilities
    :param ep (float): an very small value to avoid log(0)
    :return: the KL divergence between x and y
    """
    x = np.asarray(x, np.float)
    y = np.asarray(y, np.float)
    x = x + ep # numerical stability
    y = y + ep # numerical stability
    return (x * np.log(x / y)).sum()


def bhattacharyya (x, y, ep=0.000001):
    """
    This function computes the Bhattacharyya distance between two arrays
    :param x: an array k-dim
    :param y: an array k-dim
    :param ep (float): an very small value to avoid log(0)
    :return: the distance
    """
    x = np.asarray(x, np.float)
    y = np.asarray(y, np.float)
    x = x + ep # numerical stability
    y = y + ep # numerical stability
    bc = np.sqrt(x * y).sum()
    return -np.log(bc)


def hellinger(x, y):
    """
    This function computes the Hellinger distance between two arrays
    :param x: an array k-dim
    :param y: an array k-dim
    :return: the distance
    """
    x = np.asarray(x, np.float)
    y = np.asarray(y, np.float)
    d = np.sqrt(np.power(np.sqrt(x) - np.sqrt(y), 2).sum())
    return d/np.sqrt(2)


def mahalanobis_dist(x, y, alphas):
    """
    This function computes the Mahalanobis distance between two arrays from a Dirichlet distribution
    :param x: an array k-dim
    :param y: an array k-dim
    :param alphas: the Dirichlet distribution parameters
    :return: the distance
    """
    cov = dirichlet.cov_matrix(alphas)
    return mahalanobis(x, y, np.linalg.pinv(cov))


def dirichlet_sets_estimation(data, labels_name, col_pred="PRED", pred_pos=2, col_true="REAL", max_iter=5000, tol=1e-6):
    """
    This function implements the Dirichlet estimation for each label of a classification results based on a softmax.
    It computes alphas for the following sets for each label in labels_name:
    hit: the set of hit predictions in data for the label
    miss: the set of miss predictions in data for the label
    all: the set of all predictions for the label

    :param data: a pandas DataFrame with the softmax results of a classifier
    :param labels_name: a list with the labels names
    :param col_pred: the column in data representing the predicted label
    :param pred_pos: if the col_pred is not present in data, it will include this column in this position
    :param col_true: the column with the real results for the samples
    :param max_iter: the max interation to estimate the dirichlet (see dirichlet.py)
    :param tol: the tolerance to stop the estimation (see dirichlet.py)
    :return: a dictionary with all alphas for each set
    """

    print ("-"*50)
    print("- Starting the Dirichlet sets estimation for the DataFrame")

    # If the data is a path, we load it.
    if isinstance(data, str):
        print("- Loading DataFrame...")
        data = pd.read_csv(data)

    # Checking if we need to include the prediction column or the DataFrame already has it.
    data = insert_pred_col(data, labels_name, pred_pos=pred_pos, col_pred=col_pred)

    # Dict to save the stats
    estimations = dict()

    # Now we're going to compute the max, min and avg entropy for each label and considering the hits and misses:
    for lab in labels_name:
        print("- Dirichlet estimation for {}...".format(lab))
        d_lab = data[data[col_true] == lab]
        d_hit = d_lab[d_lab[col_true] == d_lab[col_pred]]
        d_miss = d_lab[d_lab[col_true] != d_lab[col_pred]]

        Dlab = d_lab[labels_name].values
        alphas_lab = dirichlet.estimate(Dlab, tol=tol, max_iter=max_iter)

        Dhit = d_hit[labels_name].values
        alphas_hit = dirichlet.estimate(Dhit, tol=tol, max_iter=max_iter)

        Dmiss = d_miss[labels_name].values
        if len(Dmiss) == 0:
            alphas_miss = None
        else:
            alphas_miss = dirichlet.estimate(Dmiss, tol=tol, max_iter=max_iter)

        estimations[lab] = {
            'hit': {
                'avg_prob': d_hit[labels_name].mean(),
                'std_prob': d_hit[labels_name].std(),
                'alphas': alphas_hit

            },
            'miss': {
                'avg_prob': d_miss[labels_name].mean(),
                'std_prob': d_miss[labels_name].std(),
                'alphas': alphas_miss
            },
            'all': {
                'avg_prob': d_lab[labels_name].mean(),
                'std_prob': d_lab[labels_name].std(),
                'alphas': alphas_lab
            }
        }

    return estimations
