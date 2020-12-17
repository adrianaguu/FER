#coding=utf-8
import os, sys, shutil
import random as rd
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
import pdb
import csv
try:
    import cPickle as pickle
except:
    import pickle

def load_imgs_tsn(video_root, video_list, rectify_label):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()

    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video
            num_per_part = int(img_count) // 3

            if int(img_count) > 3:
                for i in range(img_count):

                    random_select_first = random.randint(0, num_per_part)
                    random_select_second = random.randint(num_per_part, num_per_part * 2)
                    random_select_third = random.randint(2 * num_per_part, len(img_lists) - 1)

                    img_path_first = os.path.join(video_path, img_lists[random_select_first])
                    img_path_second = os.path.join(video_path, img_lists[random_select_second])
                    img_path_third = os.path.join(video_path, img_lists[random_select_third])

                    imgs_first.append((img_path_first, label))
                    imgs_second.append((img_path_second, label))
                    imgs_third.append((img_path_third, label))

            else:
                for j in range(len(img_lists)):
                    img_path_first = os.path.join(video_path, img_lists[j])
                    img_path_second = os.path.join(video_path, random.choice(img_lists))
                    img_path_third = os.path.join(video_path, random.choice(img_lists))

                    imgs_first.append((img_path_first, label))
                    imgs_second.append((img_path_second, label))
                    imgs_third.append((img_path_third, label))

            ###  return video frame index  #####
            index.append(np.ones(img_count) * id)  
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, imgs_second, imgs_third, index


def load_imgs_total_frame(video_root, video_list, rectify_label):
    imgs_first = list()

    with open(video_list, 'r') as imf:
        index = []
        video_names = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####
            
            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video

            for frame in img_lists:
                # pdb.set_trace()
                imgs_first.append((os.path.join(video_path, frame), label))
            ###  return video frame index  #####
            video_names.append(video_name)
            index.append(np.ones(img_count) * id)
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, index


def load_imgs_region(video_root, video_list, rectify_label):

    imgs_copy = list()
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    imgs_fourth = list()
    imgs_fifth = list()

    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists) 
            #print(img_lists[0])

            img_path_copy = os.path.join(video_path, img_lists[0])
            img_path_first = os.path.join(video_path, img_lists[1])
            img_path_second = os.path.join(video_path, img_lists[2])
            img_path_third = os.path.join(video_path, img_lists[3])
            img_path_fourth = os.path.join(video_path, img_lists[4])
            img_path_fifth = os.path.join(video_path, img_lists[5])
            
            
            
            imgs_copy.append((img_path_copy, label))
            imgs_first.append((img_path_first, label))
            imgs_second.append((img_path_second, label))
            imgs_third.append((img_path_third, label))
            imgs_fourth.append((img_path_fourth, label))
            imgs_fifth.append((img_path_fifth, label))

            ###  return video frame index  #####
            index.append(np.ones(img_count) * id)  
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_copy,imgs_first, imgs_second, imgs_third,imgs_fourth,imgs_fifth, index

class VideoDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None, csv = False):

        self.imgs_first, self.index = load_imgs_total_frame(video_root, video_list, rectify_label)
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


class byRegionDataset(data.Dataset):
     def __init__(self, video_root, video_list, rectify_label=None, transform=None):

        self.imgs_copy, self.imgs_first, self.imgs_second, self.imgs_third,self.imgs_fourth,self.imgs_fifth, self.index = load_imgs_region(video_root, video_list, rectify_label)
        self.transform = transform

     def __getitem__(self, index):

        path_copy, target_copy = self.imgs_copy[index]
        img_copy = Image.open(path_copy).convert("RGB")
        if self.transform is not None:
            img_copy = self.transform(img_copy)

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)

        path_fourth, target_fourth = self.imgs_fourth[index]
        img_fourth = Image.open(path_fourth).convert("RGB")
        if self.transform is not None:
            img_fourth = self.transform(img_fourth)

        path_fifth, target_fifth = self.imgs_fifth[index]
        img_fifth = Image.open(path_fifth).convert("RGB")
        if self.transform is not None:
            img_fifth = self.transform(img_fifth)
        return img_copy, target_copy, img_first, target_first, img_second, target_second, img_third, target_third, img_fourth, target_fourth, img_fifth, target_fifth, self.index[index]

     def __len__(self):
        return len(self.imgs_first)


# 
class TripleImageDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None):

        self.imgs_first, self.imgs_second, self.imgs_third, self.index = load_imgs_tsn(video_root, video_list,
                                                                                           rectify_label)
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)
        return img_first, target_first, img_second, target_second, img_third, target_third, self.index[index]

    def __len__(self):
        return len(self.imgs_first)
