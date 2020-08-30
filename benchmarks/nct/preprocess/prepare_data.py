# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Script to prepare data to train, validate and test NCT dataset
"""

import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
import os
from raug.utils.loader import create_csv_from_folders, split_train_val_test_csv, split_k_folder_csv, label_categorical_to_number


data_csv = create_csv_from_folders ("/home/patcha/Datasets/CRC-VAL-HE-7K/imgs", img_exts=['tif'])
data_ = split_train_val_test_csv (data_csv, save_path=None, tr=0.95, tv=0.0, te=0.05, seed_number=8)
data_test = data_[ data_['partition'] == 'test' ]
data_train = data_[ data_['partition'] == 'train' ]
data_test.to_csv(os.path.join("/home/patcha/Datasets/CRC-VAL-HE-7K/", 'NCT_test.csv'), index=False)
data_train = data_train.reset_index(drop=True)
data_train_ = split_k_folder_csv (data_train, "target", save_path=None, k_folder=5, seed_number=8)
label_categorical_to_number (data_train_, "target", col_target_number="target_number",
                             save_path="/home/patcha/Datasets/CRC-VAL-HE-7K/NCT_train.csv")