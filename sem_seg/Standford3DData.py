import h5py
import torch

import os
import os.path as osp
import shutil
from utils import provider_utils as provider
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
Data_DIR = os.path.join(ROOT_DIR, 'data')

ALL_FILES = provider.getDataFiles(os.path.join(Data_DIR, 'indoor3d_sem_seg_hdf5_data/all_files.txt'))
room_filelist = [line.rstrip() for line in open(os.path.join(Data_DIR, 'indoor3d_sem_seg_hdf5_data/room_filelist.txt'))]

# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    h5_filename = os.path.join(Data_DIR, h5_filename)
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)

data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)
test_area = 6
test_area = 'Area_' + str(test_area)
train_idxs = []
test_idxs = []
for i, room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

train_data = data_batches[train_idxs, ...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs, ...]
test_label = label_batches[test_idxs]

# (20291, 4096, 9) --> 20291 number of training dataset, 4096 is the
# number of points , 9 -->
# each point is represeted as XYZ, RGB and normalized location as  to the room (from 0 to 1)

print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)
