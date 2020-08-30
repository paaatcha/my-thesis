# -*- coding: utf-8 -*-
"""

Autor: André Pacheco
Email: pacheco.comp@gmail.com

Functions that implement the ensemble of models
"""

import sys
sys.path.insert(0,'../') # including the path to deep-tasks folder
sys.path.insert(0,'./utils') # including the path to deep-tasks folder
from constants import TOPSIS_PATH
sys.path.insert(0,TOPSIS_PATH)
import os
import pandas as pd
import pickle
import numpy as np
from scipy.stats import mode
from tqdm import tqdm
from TOPSIS import TOPSIS
from scores import compute_estimations, get_scores


def get_scores_test (data_val, data_test, labels, metric_name, test_metrics_path=None, save=True):

    test_scores = None
    if test_metrics_path is not None:
        if os.path.isfile(test_metrics_path):
            with open(test_metrics_path, 'rb') as f:
                test_scores = pickle.load(f)

    if test_scores is None:

        estimations = compute_estimations(data_val, labels)
        test_scores = get_scores(data_test, labels, estimations, metric_name=metric_name, scores_per_label=False)
        if save:
            with open(test_metrics_path, 'wb') as f:
                pickle.dump(test_scores, f)

    return test_scores


def _dynamic_weigths_dir (hits, misses):
    h = np.array(hits)
    m = np.array(misses)
    s = m/h
    return s / s.sum()


def _get_LewDir (ensemble, idx):

    hits, misses = list(), list()
    for net in ensemble.keys():
        hit = ensemble[net]['metrics'][idx]['hit']
        miss = ensemble[net]['metrics'][idx]['miss']
        hits.append(hit)
        misses.append(miss)
    return _dynamic_weigths_dir (hits, misses)


def _get_NP_weights (predictions, type, sig=1):
    n_models, n_labels = predictions.shape
    weights = np.zeros([n_labels, n_models])
    for i in range(n_labels):
        preds = predictions[:,i]
        if type == "NP-AVG":
            f_ = preds.mean()
        elif type == "NP-MAX":
            f_ = preds.max()
        weights[i,:] = (1/sig*np.sqrt(2*np.pi)) * np.exp(-(preds-f_)/2*(sig*sig))
    weights = weights / weights.sum(axis=0)
    return weights


def _get_agg_predictions (preds, method, weights=None, top_k=None):

    if top_k is not None and weights is not None:
        if top_k > len(weights):
            raise Exception("Top {} is greater than the number of classifiers".format(top_k))
        k_best = weights.argsort()[::-1][:top_k]
        weights = weights[k_best]
        weights = weights / weights.sum()
        preds = preds[k_best, :]

    if method == 'avg':
        if weights is not None:
            if len(weights) != preds.shape[0] and weights.shape != preds.T.shape:
                raise Exception ("The number of weights must be the same of classifiers")
            agg_preds = (preds.T * weights).mean(axis=1)
        else:
            agg_preds = preds.mean(axis=0)

    elif method == 'max':
        agg_preds = preds.max(axis=0)
        agg_preds = agg_preds / agg_preds.sum()

    elif method == 'majority':
        labels = preds.argmax(axis=1)
        agg_preds = np.zeros(preds.shape[1])
        for l in labels:
            agg_preds[l] += 1
        agg_preds = agg_preds / agg_preds.sum()

    elif method == 'product':
        agg_preds = preds.prod(axis=0)
        agg_preds = agg_preds / agg_preds.sum()

    elif method == 'topsis':
        preds = preds.T
        cb = [0] * preds.shape[1]
        t = TOPSIS(preds, weights, cb)
        t.normalizeMatrix()
        if weights is not None:
            t.introWeights()
        t.getIdealSolutions()
        t.distanceToIdeal()
        t.relativeCloseness()
        agg_preds = t.rCloseness
        agg_preds = agg_preds / agg_preds.sum()

    elif method == 'geo_avg':
        agg_preds = np.sqrt(preds.prod(axis=0))
        agg_preds = agg_preds / agg_preds.sum()

    elif method == 'dynamic':
        agg_preds = (preds.T * weights).sum(axis=1)

    else:
        raise Exception ("There is no aggregation method called {}".format(method))

    return agg_preds


def agg_ensemble (ensemble, labels, agg_method, weights=None, top_k=None, img_name_col='image',
                  true_lab_col='REAL'):

    nets = list(ensemble.keys())
    num_samples = len(ensemble[nets[0]]['data'])

    # sanity check
    for net in nets[1:]:
        if num_samples != len(ensemble[net]['data']):
            print ("The network {} has more samples than the remaining ones".format(net))
            raise Exception

    try:
        aux = ensemble[nets[0]]['data'][true_lab_col]
        df_cols = ['image', true_lab_col] + labels
    except KeyError:
        df_cols = ['image'] + labels
    df_values = list()

    print ("\n- Starting the ensemble aggregation...")
    print ("-- {} models: {}".format(len(nets), nets))
    print ("-- Method: {}\n-- Weights: {}\n-- Top k: {}\n".format(agg_method, weights, top_k))

    # Iterating in each row
    with tqdm(total=num_samples, ascii=True, ncols=100) as t:
        for row_i in range(num_samples):

            # Getting the values for row_i to each model in the ensemble
            predictions = list()
            previous_img_name, previous_img_lab = None, None
            for net in nets:
                pred_net = list(ensemble[net]['data'].iloc[row_i][labels].values)

                try:
                    img_name = ensemble[net]['data'].iloc[row_i][img_name_col]
                except:
                    img_name = None
                try:
                    img_lab = ensemble[net]['data'].iloc[row_i][true_lab_col]
                except KeyError:
                    img_lab = None

                # More sanities checking
                if previous_img_lab is None and img_lab is not None:
                    previous_img_lab = img_lab
                else:
                    if previous_img_lab != img_lab:
                        print ("Houston, we have a problem! We are comparing images with different labels: {} and {}".format(
                                previous_img_lab, img_lab))
                        raise Exception

                if previous_img_name is None:
                    previous_img_name = img_name
                elif img_name is not None:
                    if previous_img_name != img_name:
                        print ("Houston, we have a problem! We are comparing images with different names: {} and {}".format(
                                previous_img_name, img_name))
                        raise Exception

                # Stacking predictions from all models
                predictions.append(pred_net)

            predictions = np.asarray(predictions)

            weights_value = None
            if weights == 'LewDir':
                weights_value = _get_LewDir (ensemble, row_i)
            elif weights == 'NP-AVG' or weights == 'NP-MAX':
                weights_value = _get_NP_weights(predictions, weights)
            elif isinstance(weights, list):
                weights_value = weights

            # Getting the aggregated predictions
            agg_preds = _get_agg_predictions (predictions, agg_method, weights_value, top_k)

            # Saving the values to compose the DataFrame
            if img_lab is None:
                aux = [img_name]
            else:
                aux = [img_name, img_lab]
            aux.extend(agg_preds)
            df_values.append(aux)

            t.update()

    df = pd.DataFrame(df_values, columns=df_cols)
    return df











