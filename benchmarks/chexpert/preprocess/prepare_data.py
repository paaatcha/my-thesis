# -*- coding: utf-8 -*-
"""

Autor: André Pacheco
Email: pacheco.comp@gmail.com

Script to prepare data to train, validate and test CheXpert dataset
"""

import pandas as pd
import os
import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
from raug.utils.loader import split_k_folder_csv, split_train_val_test_csv, label_categorical_to_number


def get_new_csv (df, labels, target, path_to_save):
    df = df[df['Frontal/Lateral'] == 'Frontal'] # getting only frontal data
    labels.remove(target)
    new_df = df.drop(columns=labels)
    new_df[target] = new_df[target].replace([1.0, -1.0, 0.0], ['POS', 'UNC', 'NEG'])
    new_df.to_csv(path_to_save, index=False)
    return new_df

# Constants to define the dataset path
dataset_folder = '/home/patcha/Datasets/CheXpert/CheXpert-v1.0-small'
data_csv = os.path.join(dataset_folder, 'train.csv')
target_label = 'Pleural Effusion'

data_csv = pd.read_csv(data_csv).fillna(0.0)
labels = list(data_csv.columns)[5:]
data_csv = get_new_csv (data_csv, labels, target_label,
                        os.path.join("/home/patcha/Datasets/CheXpert/", 'data_{}.csv'.format(target_label)))

data_ = split_train_val_test_csv (data_csv, save_path=None, tr=0.95, tv=0.0, te=0.05, seed_number=8)
data_test = data_[ data_['partition'] == 'test' ]
data_train = data_[ data_['partition'] == 'train' ]
data_test.to_csv(os.path.join("/home/patcha/Datasets/CheXpert/", 'data_Pleural Effusion_test.csv'), index=False)
data_train = data_train.reset_index(drop=True)
data_train_ = split_k_folder_csv (data_train, "Pleural Effusion", save_path=None, k_folder=5, seed_number=8)
label_categorical_to_number (data_train_, "Pleural Effusion", col_target_number="diagnostic",
                             save_path=os.path.join("/home/patcha/Datasets/CheXpert/", 'data_Pleural Effusion_train.csv'))
