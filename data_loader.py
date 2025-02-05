import os
import random
from random import randint
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from utils import get_data_list

class Dataset(data.Dataset):
    def __init__(self, data_list, transform, mode):
        '''
        1. Under Emotion directory, read all emotion file name
            ex) Emotion/S005/001/S005_001_00000011_emotion.txt
        2. Under extended-cohn-kanade-images, read all image file name corresponding to files of 1.
            ex) cohn-kanade-images/S005/001/S005_001_00000011.png
        '''
        self.transform = transform
        self.dataset = data_list
        self.mode = mode
        random.seed(1234)
        random.shuffle(self.dataset)

    def __getitem__(self, index):
        label, img_dirname = self.dataset[index]
        filenames = sorted(os.listdir(img_dirname))
        # if len(filenames) < 3:
        #     print(img_dirname)
        # unused. 
        degree = np.random.randint(-20, 20)
        seed = np.random.randint(2147483647) # make a seed with numpy generator

        imgs = self._stack_frames(0, degree, img_dirname, filenames, seed, False)
        img = self._stack_frames(1, degree, img_dirname, filenames, seed, False)
        imgs = torch.cat((imgs, img), 0)
        img = self._stack_frames(2, degree, img_dirname, filenames, seed, False)

        imgs = torch.cat((imgs, img), 0)
        label = torch.LongTensor([label])

        # print(imgs.size())
        # print(imgs.size())
        return imgs, label

    def __len__(self):
        return len(self.dataset)

    def _stack_frames(self, nf, degree, img_dirname, filenames, seed, rotate = False):
        img = Image.open(os.path.join(img_dirname, filenames[nf])).convert('RGB').convert('L')
        if self.mode == 'train' and rotate == True:
            img = TF.rotate(img, degree)
        random.seed(seed)
        # print(img)
        # img = img.convert('RGB')
        img = self.transform(img)
        return img

def get_loader(config):
    train_list, valid_list, num_data = get_data_list(config.emotion_dir,
                                                     config.image_dir,
                                                     config.cls,
                                                     config.kfold,
                                                     config.ithfold)
    transform = []

    transform=T.Compose([T.Resize(config.image_size),
                            T.CenterCrop(config.image_size),
                            T.ToTensor(),
                            T.Normalize([0.5], [0.5]),
                        ])
    # transform.append(T.RandomHorizontalFlip())
    # transform.append())
    # onfig.image_size
    # transform.append(T.CenterCrop(config.crop_size))
    # transform.append
    # transform.append(T.RandomCrop(config.crop_size, 4))
    # transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    # transform = T.Compose([T.Resize(config.image_size),
    #     T.ToTensor(), T.Normalize([0.5], [0.5])])
    # transform = T.Compose(transform)

    transform_valid=T.Compose([T.Resize(config.image_size),
                            T.CenterCrop(config.image_size),
                            T.ToTensor(),
                            T.Normalize([0.5], [0.5]),
                        ])
    # transform_valid = []
    # transform_valid.append(T.Resize(config.image_size))
    # transform_valid.append(T.CenterCrop(config.crop_size))
    # transform_valid = T.Compose([
    #     T.ToTensor(), T.Normalize([0.5], [0.5])])
    # transform_valid.append(T.ToTensor())
    # transform_valid.append(T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)))
    # transform_valid = T.Compose(transform_valid)

    # if config.dataset_name == 'Dataset':
    #     print('Dataset dataset for train and validation are created...')
    train_dataset = Dataset(train_list, transform, config.mode)
    # print(train_dataset[0])
    valid_dataset = Dataset(valid_list, transform_valid, 'valid')

    print(config.batch_size)
    print(config.num_workers)

    if config.mode == 'train':
        print('The number of train_dataset(before augmentation): {} '.format(len(train_dataset)))
        print('The number of valid_dataset: {}'.format(len(valid_dataset)))
        trainloader = data.DataLoader(dataset=train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=config.num_workers)
        validloader = data.DataLoader(dataset=valid_dataset,
                                      batch_size=len(valid_dataset),
                                      shuffle=False,
                                      num_workers=config.num_workers)
        return trainloader, validloader, num_data
    if config.mode == 'valid':
        print('The number of valid_dataset: {}'.format(len(valid_dataset)))
        validloader = data.DataLoader(dataset=valid_dataset,
                                      batch_size=len(valid_dataset),
                                      shuffle=False,
                                      num_workers=config.num_workers)
        return None, validloader, num_data