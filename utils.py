import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import random

def search_files(emtn_dir, image_dir, dir_name=None, dataset=None):
    '''
    1. Recursively search Emotion folder to find emotion labels
        ex) 6.0000000e+00 (position: Emotion/S005/001/S005_001_00000011_emotion.txt)
    2. Find image directory name corresponding to each emotion direcotry name
        ex) cohn-kanade-images/S005/001/S005_001_00000011.png
    3. Append [label, image directory name] to self.dataset
    '''
    if dir_name is None:
        dir_name = emtn_dir
    if dataset is None:
        dataset = []
    try:
        filenames = os.listdir(dir_name)
        for filename in filenames: #Emotion/S005
            full_filename = os.path.join(dir_name, filename)
            if os.path.isdir(full_filename):
                print(full_filename)
                search_files(emtn_dir, image_dir, full_filename, dataset)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.txt':
                    #ex. Value of Emotion/S005/001/S005_001_00000011_emotion.txt
                    label = int(float(open(full_filename, 'r').readline().strip())) - 1
                    full_dirname, _ = os.path.split(full_filename) #ex. Emotion/S005/001
                    #ex. cohn-kanade-images/S005/001
                    dataset.append([label, full_dirname.replace(emtn_dir, image_dir)]) #ex. cohn-kanade-images/S005/001
                    break
    except PermissionError:
        pass
    return dataset

def search_files_oulu(image_dir, dir_name=None, dataset=None):
    '''
    1. Recursively search Emotion folder to find emotion labels
        ex) 6.0000000e+00 (position: Emotion/S005/001/S005_001_00000011_emotion.txt)
    2. Find image directory name corresponding to each emotion direcotry name
        ex) cohn-kanade-images/S005/001/S005_001_00000011.png
    3. Append [label, image directory name] to self.dataset
    '''
    if dir_name is None:
        dir_name = image_dir
    if dataset is None:
        dataset = []
    emotion = {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Sadness': 4, 'Surprise':5}

    try:
        filenames = os.listdir(dir_name) #oulu_align
        for filename in filenames: #ex. P001
            print(filename)
            full_filename = os.path.join(dir_name, filename) #oulu_align/P001
            if os.path.isdir(full_filename):
                search_files_oulu(image_dir, full_filename, dataset)
            else:
                emt = os.path.split(dir_name)[-1]
                print(emt)
                label = emotion[emt]
                dataset.append([label, dir_name]) #ex. cohn-kanade-images/S005/001
                break
    except PermissionError:
        pass
    return dataset


def count_data_per_cls(emtns):
    num_data = []
    total = 0
    for i in range(len(emtns)):
        num_data.append(len(emtns[i]))
        total += len(emtns[i])
    # num_data = np.array(num_data)
    num_data = total / np.array(num_data)
    return num_data

def get_data_list(emtn_dir, image_dir, num_cls, k, ith_fold):
    '''
    Divide dataset as k fold list
    Use ith_fold list as test data list
    Merge other folds into train data list
    Return train data list, test data list
    Type: [[label, path of image], ...]
    '''
    # print(num_cls)
    if num_cls == 7:
        dataset = search_files(emtn_dir, image_dir)
    else:
        dataset = search_files_oulu(image_dir)
    random.seed('1234')
    random.shuffle(dataset)
    print("dataset:{}".format(len(dataset)))


    #emtns[i]: dir paths of each image of i-th emotion type ex. '/data/cohn-kanade-images/S014/003'
    emtns = [[] for i in range(num_cls)]
    for i in range(len(dataset)):
        emtns[dataset[i][0]].append(dataset[i][1])

    num_data = count_data_per_cls(emtns)

    k_folds = [[] for i in range(k)]
    for i in range(num_cls):
        emtn_now = emtns[i]
        num_i_emtn = len(emtn_now) # number of data of each i-th emotion type
        emtn_per_kfold = int(num_i_emtn / k)
        for j in range(k):
            for m in range(emtn_per_kfold):
                    k_folds[j].append([i, emtn_now[j*emtn_per_kfold + m]])
        rem = num_i_emtn % k
        if rem:
            for j in range(rem):
                k_folds[j].append([i, emtn_now[emtn_per_kfold*k + j]])

    test_list = k_folds[ith_fold]
    train_list = []
    for i in range(k):
        if i != ith_fold:
            train_list += k_folds[i]
    train_list = oversample(train_list, num_cls)
    return train_list, test_list, num_data