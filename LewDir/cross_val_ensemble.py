# -*- coding: utf-8 -*-
"""

Autor: André Pacheco
Email: pacheco.comp@gmail.com

Script to run the cross validation ensemble in order to asses the approaches performance
"""
import sys
sys.path.insert(0,'../') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
import os
from raug.utils.classification_metrics import get_metrics_from_csv
import numpy as np
import pandas as pd
from agg_funcs import get_scores_test, agg_ensemble

########################################################################################################################
# ISIC
########################################################################################################################
# base_path = "/home/patcha/PHD_results/Ensemble/ISIC"
# labels = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
# labels.sort()
# kfold = 5
# agg_method = "avg" # avg, max, majority, topsis, product
# weights = "LewDir" # list of weights, None, LewDir, NP-AVG, NP-MAX
# metric_name = 'mahalanobis' # applied only to LewDir
# top_k = None # applied to only to selection
# networks = ['densenet-121', 'efficientnet-b4', 'efficientnet-b1', 'mobilenet', 'resnet-50', 'vgg-13']
########################################################################################################################

########################################################################################################################
# PAD
########################################################################################################################
base_path = "/home/patcha/PHD_results/Ensemble/PAD"
labels = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
labels.sort()
kfold = 5
agg_method = "avg" # avg, max, majority, topsis, product
weights = "LewDir" # list of weights, None, LewDir, NP-AVG, NP-MAX
metric_name = 'mahalanobis' # applied only to LewDir
top_k = None # applied to only to selection
networks = ['efficientnet-b4', 'densenet-121', 'mobilenet', 'vgg-13', 'resnet-50']
########################################################################################################################

########################################################################################################################
# NCT
########################################################################################################################
# base_output_path = "/home/patcha/PHD_results/Ensemble/NCT"
# base_path = "/home/patcha/PHD_results/Ensemble/NCT"
# labels = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
# labels.sort()
# kfold = 5
# agg_method = "avg" # avg, max, majority, topsis, product
# weights = "NP-AVG" # list of weights, None, LewDir, NP-AVG, NP-MAX
# metric_name = 'mahalanobis' # applied only to LewDir
# top_k = None # applied to only to selection
# networks = ['densenet-121', 'inceptionv4', 'googlenet', 'mobilenet', 'resnet-50', 'vgg-16']
########################################################################################################################

########################################################################################################################
# CheXpert
########################################################################################################################
# base_output_path = "/home/patcha/PHD_results/Ensemble/cheXpert"
# base_path = "/home/patcha/PHD_results/Ensemble/cheXpert"
# labels = ['NEG', 'POS', 'UNC']
# labels.sort()
# kfold = 5
# agg_method = "avg" # avg, max, majority, topsis, product
# weights = "NP-AVG" # list of weights, None, LewDir, NP-AVG, NP-MAX
# metric_name = 'mahalanobis' # applied only to LewDir
# top_k = None # applied to only to selection
# networks = ['densenet-121', 'inceptionv4', 'googlenet', 'mobilenet', 'resnet-50', 'vgg-16']
########################################################################################################################

########################################################################################################################
# OCT
########################################################################################################################
# base_output_path = "/home/patcha/PHD_results/Ensemble/OCT"
# base_path = "/home/patcha/PHD_results/Ensemble/OCT"
# labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# labels.sort()
# kfold = 5
# agg_method = "avg" # avg, max, majority, topsis, product
# weights = "LewDir" # list of weights, None, LewDir, NP-AVG, NP-MAX
# metric_name = 'mahalanobis' # applied only to LewDir
# top_k = None # applied to only to selection
# networks = ['densenet-121', 'inceptionv4', 'googlenet', 'mobilenet', 'resnet-50', 'vgg-16']
########################################################################################################################

def get_only_folders (path, sort=True):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if sort:
        folders.sort()
    return folders

def safe_create_folder (path):
    if not os.path.exists(path):
        os.makedirs(path)
########################################################################################################################

# Creating the folder for the base_output_path
if top_k is not None:
    out_path = os.path.join(base_path, "results", "top_{}".format(top_k))
else:
    out_path = os.path.join(base_path, "results")

if isinstance(weights, list):
    out_path = os.path.join(out_path, "static_weights")
elif weights == 'LewDir':
    out_path = os.path.join(out_path, "LewDir", metric_name)
    _lewdir_scores_folder = os.path.join(base_path, "LewDir-Scores")
    safe_create_folder(os.path.join(base_path, _lewdir_scores_folder))
elif weights is not None:
    out_path = os.path.join(out_path, weights)

out_path = os.path.join(out_path, agg_method)
safe_create_folder(out_path)

########################################################################################################################
measurements = list()
for k in range(1, kfold+1):
    print ("\n")
    print("-" * 50)
    print ("Working on folder", k)
    print ("-"*50)
    out_csv_path = os.path.join(out_path, "{}_agg_predictions.csv".format(k))
    ensemble = dict()
    for net in networks:
        print ("\n- Starting folder", k, net)

        data_test_path = os.path.join(base_path, "test-pred", "{}_{}_predictions.csv".format(k, net))
        df = pd.read_csv(data_test_path)

        if weights == 'LewDir':
            data_val_path = os.path.join(base_path, "val-pred", "{}_{}_predictions.csv".format(k, net))
            test_scores_path = os.path.join(base_path, _lewdir_scores_folder,
                                            "{}_{}_{}_metrics_test.pkl".format(k, net, metric_name))
            test_scores = get_scores_test(data_val_path, data_test_path, labels, metric_name, test_scores_path)
        else:
            test_scores = None

        ensemble[net] = {'metrics': test_scores, 'data': df}

    df = agg_ensemble (ensemble, labels, agg_method, weights, top_k=top_k)
    df.to_csv(out_csv_path, index=False)

    # Updating the metrics
    acc, topk_acc, ba, rep, auc, loss, _, _ = get_metrics_from_csv(df, labels)
    rec = rep['weighted avg']['recall']
    prec = rep['weighted avg']['precision']
    f1 = rep['weighted avg']['f1-score']
    auc = auc['macro']
    measurements.append([acc, topk_acc, ba, rec, prec, f1, auc, loss])
    print("-" * 50)

print("- The aggregation is done!")
out = "-" * 50
out += "\n- Final metrics\n"
out += "-" * 50
measurements = np.asarray(measurements)
avg = measurements.mean(axis=0)
std = measurements.std(axis=0)
out += "\n- Accuracy: {:.3f} +- {:.3f}\n".format(avg[0], std[0])
out += "- Top 2 accuracy: {:.3f} +- {:.3f}\n".format(avg[1], std[1])
out += "- Balanced accuracy: {:.3f} +- {:.3f}\n".format(avg[2], std[2])
out += "- Recall: {:.3f} +- {:.3f}\n".format(avg[3], std[3])
out += "- Precision: {:.3f} +- {:.3f}\n".format(avg[4], std[4])
out += "- F1-Score: {:.3f} +- {:.3f}\n".format(avg[5], std[5])
out += "- AUC: {:.3f} +- {:.3f}\n".format(avg[6], std[6])
out += "- Loss: {:.3f} +- {:.3f}\n".format(avg[7], std[7])
out += "-" * 50
print(out)
with open(os.path.join(out_path, "measurements.txt"), 'w') as f:
    f.write(out)




