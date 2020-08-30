# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Script to prepare data to train, validate and test OCT dataset
"""

import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
from raug.utils.loader import create_csv_from_folders, split_k_folder_csv, label_categorical_to_number


create_csv_from_folders ("/home/patcha/Datasets/OCT2017/test", img_exts=['jpeg'],
                                    save_path="/home/patcha/Datasets/OCT2017/OCT_test.csv")

data_csv = create_csv_from_folders ("/home/patcha/Datasets/OCT2017/train", img_exts=['jpeg'])

data_train_ = split_k_folder_csv (data_csv, "target", save_path=None, k_folder=5, seed_number=8)
label_categorical_to_number (data_train_, "target", col_target_number="target_number",
                             save_path="/home/patcha/Datasets/OCT2017/OCT_train.csv")