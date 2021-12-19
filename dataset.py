import os
import numpy as np
import pandas as pd
import nibabel as nib
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import medicaltorch.transforms as mt_transforms
import torchvision as tv
import torchvision.utils as vutils
import transforms as tf

from tqdm import *

def linked_augmentation(gm_batch, wm_batch, transform):

    gm_batch_size = gm_batch.size(0)

    gm_batch_cpu = gm_batch.cpu().detach()
    gm_batch_cpu = gm_batch_cpu.numpy()

    wm_batch_cpu = wm_batch.cpu().detach()
    wm_batch_cpu = wm_batch_cpu.numpy()

    samples_linked_aug = []
    sample_linked_aug = {'input': [gm_batch_cpu,
                                   wm_batch_cpu]}
    # print('GM: ', sample_linked_aug['input'][0].shape)
    # print('WM: ', sample_linked_aug['input'][1].shape)
    out = transform(sample_linked_aug)
    # samples_linked_aug.append(out)

    # samples_linked_aug = mt_datasets.mt_collate(samples_linked_aug)
    return out
    
class PAC20192D(Dataset):
    def __init__(self, ctx, set, split1=0.7, split2=0.8, portion=0.8): #set, split1=0.7, split2=0.8 ###
        """
        split: train/val split
        portion: portion of the axial slices that enter the dataset
        """
        self.ctx = ctx
        self.portion = portion
        dataset_path = ctx["dataset_path"]

        csv_path = os.path.join(dataset_path, "IXI1126.csv")

        dataset = []

        stratified_dataset = []

        with open(csv_path) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3]),
                    'filename': line[4].replace('\n','')
                })

        sites = defaultdict(list)
        
        for data in dataset:
            sites[data['site']].append(data)

        for site in sites.keys():
            length = len(sites[site])
            if set == 'train':
                stratified_dataset += sites[site][0:int(length*split1)]
                print(stratified_dataset)
            if set == 'val':
                stratified_dataset += sites[site][int(length*split1):int(length*split2)]
                print(stratified_dataset)
            if set == 'test':
                stratified_dataset += sites[site][int(length*split2):]
                print(stratified_dataset)

        self.dataset = stratified_dataset
        self.slices = []

        self.transform = tv.transforms.Compose([
            mt_transforms.ToPIL(labeled=False),
            mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                           sigma_range=(3.5, 4.0),
                                           p=0.3, labeled=False),
            mt_transforms.RandomAffine(degrees=4.6,
                                       scale=(0.98, 1.02),
                                       translate=(0.03, 0.03),
                                       labeled=False),
            mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
            mt_transforms.ToTensor(labeled=False),
        ])

        self.preprocess_dataset()

    def preprocess_dataset(self):
        for i, data in enumerate(tqdm(self.dataset, desc="Loading dataset")):

            #filename_gm = os.path.join(self.ctx["dataset_path"], 'gm', data['subject'] + '_gm.nii.gz')
            filename_gm = data['filename']
            input_image_gm = torch.FloatTensor(nib.load(filename_gm).get_fdata())
            input_image_gm = input_image_gm.permute(2, 0, 1)

            #filename_wm = os.path.join(self.ctx["dataset_path"], 'wm', data['subject'] + '_wm.nii.gz')
            filename_wm = data['filename']
            input_image_wm = torch.FloatTensor(nib.load(filename_wm).get_fdata())
            input_image_wm = input_image_wm.permute(2, 0, 1)

            start = int((1.-self.portion)*input_image_gm.shape[0])
            end = int(self.portion*input_image_gm.shape[0])
            input_image_gm = input_image_gm[start:end,:,:]
            input_image_wm = input_image_wm[start:end,:,:]
            for slice_idx in range(input_image_wm.shape[0]):
                slice_gm = input_image_gm[slice_idx,:,:]
                slice_wm = input_image_wm[slice_idx,:,:]

                slice_gm = slice_gm.unsqueeze(0)
                slice_wm = slice_wm.unsqueeze(0)

                slice = torch.cat([slice_gm, slice_wm], dim=0)

                # print(slice.max(), slice.min())
                self.slices.append({
                    'image': slice,
                    'age': data['age']
                })
                # plt.imshow(slice.squeeze())
                # plt.show()

    def __getitem__(self, idx):

        data = self.slices[idx]
        #transformed = { 
        #'input': data['image'] 
        # } 
        # plt.imshow(data['image'][0])
        # plt.title('gm')
        # plt.show()
        # plt.imshow(data['image'][1])
        # plt.title('wm')
        # plt.show()
        gm = data['image'][0].unsqueeze(0)
        wm = data['image'][1].unsqueeze(0)
        
        batch = linked_augmentation(gm, wm, self.transform)
        # print('gm: ', batch['input'][0].shape)
        # print('wm: ', batch['input'][1].shape)
        batch = torch.cat([batch['input'][0], batch['input'][1]], dim=0) 
        # print('Final shape: ', batch.shape)

        #transformed = self.transform(transformed)
    
        return {
            'input': batch,
            'label': data['age']
        }

    def __len__(self):
        return len(self.slices)

class PAC20193D(Dataset):
    def __init__(self, ctx, set): #set, split1=0.7, split2=0.8 ###
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]

        #csv_path = os.path.join(dataset_path, "IXI0923.csv")
        csv_path_train = os.path.join(dataset_path, "train1105.csv")
        csv_path_valid = os.path.join(dataset_path, "valid1105.csv")
        csv_path_test = os.path.join(dataset_path, "test1105.csv")
    
        #dataset = []
        dataset_train = []
        dataset_valid = []
        dataset_test = []
        dataset = []
        
        stratified_dataset = []
        
        with open(csv_path_train) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset_train.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3]),
                    'filename': line[4].replace('\n','')
                })
        with open(csv_path_valid) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset_valid.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3]),
                    'filename': line[4].replace('\n','')
                })
        with open(csv_path_test) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset_test.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3]),
                    'filename': line[4].replace('\n','')
                })

        #sites = defaultdict(list)
        sites_train = defaultdict(list)
        sites_valid = defaultdict(list)
        sites_test = defaultdict(list)
        
        for data in dataset_train:
            sites_train[data['site']].append(data)
        for data in dataset_valid:
            sites_valid[data['site']].append(data)
        for data in dataset_test:
            sites_test[data['site']].append(data)
        
        if set == 'train' :
            for site in sites_train.keys():
                length_train = len(sites_train[site])
                stratified_dataset += sites_train[site][0:int(length_train)]
                print(stratified_dataset)
        if set == 'valid':
            for site in sites_valid.keys():
                length_valid = len(sites_valid[site])
                stratified_dataset += sites_valid[site][0:int(length_valid)]
                print(stratified_dataset)
        if set == 'test':
            for site in sites_test.keys():
                length_test = len(sites_test[site])
                stratified_dataset += sites_test[site][0:int(length_test)]
                print(stratified_dataset)


        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([
            tf.ImgAugTranslation(10),
            tf.ImgAugRotation(40),
            tf.ToTensor(),
        ])


    def __getitem__(self, idx):
        data = self.dataset[idx]
        filename = data['filename']
        input_image = torch.FloatTensor(nib.load(filename).get_fdata())
        input_image = input_image.permute(2, 0, 1)

        transformed = {
            'input': input_image
        }

        transformed = self.transform(transformed['input'])
        transformed = transformed.unsqueeze(0)
        print(transformed.shape)


        return {
            'input': transformed,
            'label': data['age']
        }

    def __len__(self):
        return len(self.dataset)

class PAC2019(Dataset):
    def __init__(self, ctx, set, split=0.8):
        self.ctx = ctx
        dataset_path = ctx["dataset_path"]

        csv_path_train = os.path.join(dataset_path, "train1105.csv")
        csv_path_valid = os.path.join(dataset_path, "valid1105.csv")
        csv_path_test = os.path.join(dataset_path, "test1105.csv")
        
        dataset_train = []
        dataset_valid = []
        dataset_test = []
        dataset = []

        stratified_dataset = []

        with open(csv_path_train) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset_train.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3]),
                    'filename': line[4].replace('\n','')
                })
        with open(csv_path_valid) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset_valid.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3]),
                    'filename': line[4].replace('\n','')
                })
        with open(csv_path_test) as fid:
            for i, line in enumerate(fid):
                if i == 0:
                    continue
                line = line.split(',')
                dataset_test.append({
                    'subject': line[0],
                    'age': float(line[1]),
                    'gender': line[2],
                    'site': int(line[3]),
                    'filename': line[4].replace('\n','')
                })

       #sites = defaultdict(list)
        sites_train = defaultdict(list)
        sites_valid = defaultdict(list)
        sites_test = defaultdict(list)

        for data in dataset_train:
            sites_train[data['site']].append(data)
        for data in dataset_valid:
            sites_valid[data['site']].append(data)
        for data in dataset_test:
            sites_test[data['site']].append(data)

        if set == 'train' :
            for site in sites_train.keys():
                length_train = len(sites_train[site])
                stratified_dataset += sites_train[site][0:int(length_train)]
                print(stratified_dataset)
        if set == 'valid':
            for site in sites_valid.keys():
                length_valid = len(sites_valid[site])
                stratified_dataset += sites_valid[site][0:int(length_valid)]
                print(stratified_dataset)
        if set == 'test':
            for site in sites_test.keys():
                length_test = len(sites_test[site])
                stratified_dataset += sites_test[site][0:int(length_test)]
                print(stratified_dataset)


        self.dataset = stratified_dataset

        self.transform = tv.transforms.Compose([
            mt_transforms.ToPIL(labeled=False),
            mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                           sigma_range=(3.5, 4.0),
                                           p=0.3, labeled=False),
            mt_transforms.RandomAffine(degrees=4.6,
                                       scale=(0.98, 1.02),
                                       translate=(0.03, 0.03),
                                       labeled=False),
            mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
            mt_transforms.ToTensor(labeled=False),
        ])

    def __getitem__(self, idx):
        data = self.dataset[idx]

        filename = data['filename']
        t1_image = torch.FloatTensor(nib.load(filename).get_fdata())
        t1_image = t1_image.permute(2, 0, 1)

        # transformed = {
        #     'input': gm_image
        # }
        # self.transform(transformed)

        # plt.imshow(gm_image[60,:,:])
        # plt.show()
        # plt.imshow(gm_image[:,60,:])
        # plt.show()
        # plt.imshow(gm_image[:,:,60])
        # plt.show()
        #
        # raise


        return {
            #'t1':t1_image,
            'input': t1_image,
            'label': data['age']
        }

    def __len__(self):
        return len(self.dataset)
        