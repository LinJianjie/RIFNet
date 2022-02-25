import numpy as np
import os
import sys
import json
from utils import provider_utils as provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

hdf5_data_dir = os.path.join(BASE_DIR, 'hdf5_data')
color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
color_map = json.load(open(color_map_file, 'r'))
all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')

fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))

NUM_CATEGORIES = 16
NUM_PART_CATS = len(all_cats)

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')

train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
num_train_file = len(train_file_list)
test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
num_test_file = len(test_file_list)
print(num_train_file)
train_file_idx = np.arange(0, len(train_file_list))
cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[train_file_idx[0]])
cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)
cur_data, cur_labels, order = provider.shuffle_data(cur_data, np.squeeze(cur_labels))

print(cur_data.shape)
print(cur_labels.shape)
print(cur_seg.shape)
cur_seg = cur_seg[order, ...]
print(cur_seg.shape)


