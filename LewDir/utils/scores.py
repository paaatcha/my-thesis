# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

This file implements the function to compute the scores between samples and the dirichlet estimation sets.

If you find any bug, email me.
"""

import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
from raug.utils.common import insert_pred_col
import pandas as pd
import numpy as np
import scipy.stats as stats
import pickle
import os
from metrical import get_metric, dirichlet_sets_estimation


def compute_estimations (data_val, labels, path=None, save=True):
    """
    This function computes the Dirichlet estimation for a given DataFrame that contains the softmax probabilities
    for each class. In addition, to save time, the function saves computed estimations. Just pass the path and let
    save=True

    :param data_val: dataframe with the softamx probabilities for each labe
    :param labels: a list with the classification labels
    :param path: a path to a pikcle file to be loaded or saved
    :param save: if you want to save the computed values, set it True
    :return: a dictionary with the estimations (see metrical.dirichlet_sets_estimation)
    """

    if path is None:
        data_val = insert_pred_col(data_val, labels, pred_pos=2)
        estimations = dirichlet_sets_estimation(data_val, labels)
    else:
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                estimations = pickle.load(f)
        else:
            data_val = insert_pred_col(data_val, labels, pred_pos=2)
            estimations = dirichlet_sets_estimation(data_val, labels)
            if save:
                with open(path, 'wb') as f:
                    pickle.dump(estimations, f)

    return estimations


def _diver (x,y):
    """
    Auxiliary function to compute the difference between to vectors
    :param x: an array
    :param y: an array
    :return: the difference between the arrays
    """
    return abs(x-y)


def _get_n_score_ood (samp_probs, metrics_labels, labels, N=None, verbose=False):
    """
    This function computes the score for a given sample considering the contribution of each label.

    :param samp_probs: the softmax sample probabilities
    :param metrics_labels: the labels metrics got from _get_metrics_labels
    :param labels: the classification labels
    :param N: if want to compute the contribution of the Nth labels, just set it here
    :param verbose: weather you want to print the results
    :return: two dictionaries
    1) the scores considering the contribution of each label
    2) the scores considering only the max probability
    """

    if N is None:
        pred_lab = labels[samp_probs.argsort()[-1]]
    elif N <= len(labels):
        pos = samp_probs.argsort()[::-1][0:N]
        labels = [labels[i] for i in pos]
        samp_probs = [samp_probs[i] for i in pos]
        pred_lab = labels[0]
    else:
        print ("N is greater than the number of labels")
        raise ValueError

    if verbose:
        print ('- Pred label:', pred_lab)

    # Scores considering the probabilities
    score_all, score_hit, score_miss = 0, 0, 0
    # Score pred considers only metrics for the max prob
    score_all_pred, score_hit_pred, score_miss_pred = 0, 0, 0

    for lab, prob in zip(labels, samp_probs):
        m_all, m_hit, m_miss = metrics_labels[lab]['all'], metrics_labels[lab]['hit'], metrics_labels[lab]['miss']
        w_m_all, w_m_hit, w_m_miss = prob * m_all, prob * m_hit, prob * m_miss

        score_all += w_m_all
        score_hit += w_m_hit
        score_miss += w_m_miss

        if lab == pred_lab:
            score_all_pred, score_hit_pred, score_miss_pred, diff_pred = m_all, m_hit, m_miss, _diver(m_hit, m_miss)

        if verbose:
            print ('- Label: {} | Pred: {:.3f}'.format(lab, prob))
            print ('-- All: {:.3f} | Hit: {:.3f} | Miss: {:.3f} | Diff: {:.3f}'.format(m_all, m_hit, m_miss,
                                                                                       _diver(m_hit, m_miss)))
            print ("-- Score all: {:.3f} | Score hit: {:.3f} | Score miss {:.3f}".format(w_m_all, w_m_hit, w_m_miss))

    if verbose:
        print ("- Cumulative prob: {:.3f}".format(sum(samp_probs)))
        print ("- Cumulative Scores:")
        print ("-- All: {:.3f} | Hit: {:.3f} | Miss: {:.3f} | Diff: {:.3f}\n\n".format(score_all, score_hit, score_miss,
                                                                                   _diver(score_hit, score_miss)))

    scores = {'all': score_all, 'hit': score_hit, 'miss': score_miss, 'diff': _diver(score_hit, score_miss)}
    scores_pred = {'all': score_all_pred, 'hit': score_hit_pred, 'miss': score_miss_pred, 'diff': diff_pred}

    return scores, scores_pred


def _get_metrics_labels (estimations, labels, samp_probs, metric_name, method):
    """
    This function compute the metrics (distance or divergence) between a given sample and all labels estimations.

    :param estimations: a dictionary with the estimations. See metrics.dirichlet_sets_estimation functions
    :param labels: a list with the classification labels
    :param samp_probs: the softmax probabilities for each label in labels
    :param metric_name: the metric name, see metrics.get_metric function
    :param method: you can use 'dirichlet' or 'avg'
    :return: a dictionary containing the metrics for each label
    """

    metrics_labels = dict()

    # Computing metrics for the sample considering each label in labels
    for lab in labels:
        metrics_labels[lab] = dict()

        if method == 'dirichlet':
            alphas_hit = estimations[lab]['hit']['alphas']
            alphas_miss = estimations[lab]['miss']['alphas']
            alphas_all = estimations[lab]['all']['alphas']
        else:
            alphas_hit, alphas_miss, alphas_all = None, None, None

        if alphas_all is not None:
            m_all = get_metric(samp_probs, stats.dirichlet.mean(alphas_all), metric_name, alphas_all)
            metrics_labels[lab]['all'] = m_all
        else:
            m_all = get_metric(samp_probs, estimations[lab]['all']['avg_prob'].values,
                           metric_name, alphas_all)
            metrics_labels[lab]['all'] = m_all

        if alphas_hit is not None:
            m_hit = get_metric(samp_probs, stats.dirichlet.mean(alphas_hit), metric_name, alphas_hit)
            metrics_labels[lab]['hit'] = m_hit
        else:
            m_hit = get_metric(samp_probs, estimations[lab]['hit']['avg_prob'].values,
                           metric_name, alphas_hit)
            metrics_labels[lab]['hit'] = m_hit

        if alphas_miss is not None:
            m_miss = get_metric(samp_probs, stats.dirichlet.mean(alphas_miss), metric_name, alphas_miss)
            metrics_labels[lab]['miss'] = m_miss
        else:
            if np.isnan(estimations[lab]['miss']['avg_prob'].values).any():
                m_miss = 100
            else:
                m_miss = get_metric(samp_probs, estimations[lab]['miss']['avg_prob'].values,
                                    metric_name, alphas_miss)

            metrics_labels[lab]['miss'] = m_miss

    return metrics_labels


def get_scores(data_test, labels, estimations, method="dirichlet", metric_name="mahalanobis", col_pred='PRED',
               scores_per_label=True, path_scores=None, save=True):
    """
    This function computes the scores for each sample in the data_test according to to the estimations got from
    metrical.dirichlet_sets_estimation

    :param data_test: a DataFrame with the softmax probabilities for each label
    :param labels: a list with classification labels
    :param estimations: the estimations got from metrical.dirichlet_sets_estimation
    :param method: the method ('dirichlet' or 'avg')
    :param metric_name: the metric name (see metrical.get_metric)
    :param col_pred: the name of the prediction column in data_test
    :param scores_per_label: if you want to have a dictionary with the scores per label let it True. If it is False, all
    scores will be saved in a list
    :param path_scores: the path to a pickle file to load or save the scores
    :param save: let it true if you want to save the scores to save time in the next execution
    :return: a list or a dictionary (depends on scores_per_label)
    """

    # Check if we have a final scores saved in the disk. If yes, great! Let's load it and save some time
    if path_scores is not None and os.path.isfile(path_scores):
        with open(path_scores, 'rb') as f:
            final_scores = pickle.load(f)
            return final_scores

    # If data_test is a path to a .csv, we load it.
    if isinstance(data_test, str):
        data_test = pd.read_csv(data_test)

    print ("- Computing scores for the model...")

    # The function can return a list with all final metrics or a dictionary dividing the metrics by label
    if scores_per_label:
        final_scores = { p: list() for p in labels }
    else:
        final_scores = list()

    # Through this loop the metric is computed for each sample
    for idx, sample in data_test.iterrows():

        # These operations is just to ensure the probabilities sum up 1. Sometimes, because of numerical issues, the sum
        # is something around 1.00X.
        samp_probs = list(sample[labels].values)
        samp_probs = np.array(samp_probs)
        samp_probs = (samp_probs / samp_probs.sum())
        samp_max_prob = max(samp_probs)
        try:
            samp_pred = sample[col_pred]
        except KeyError:
            samp_pred = labels[np.argmax(samp_probs)]

        # Getting the metrics per label
        metrics_labels = _get_metrics_labels(estimations, labels, samp_probs, metric_name, method)

        # Getting the scores. The mac_prob is also included (it's useful for OOD detection algorithms)
        scores, scores_pred = _get_n_score_ood(samp_probs, metrics_labels, labels, N=None, verbose=False)
        scores['max_prob'] = samp_max_prob

        if scores_per_label:
            final_scores[samp_pred].append(scores)
        else:
            final_scores.append(scores)

    # If you set save=True and set a path, the final scores will be saved in the disk to save some time in the next
    # execution
    if save and path_scores is not None:
        with open(path_scores, 'wb') as f:
            pickle.dump(final_scores, f)

    return final_scores
