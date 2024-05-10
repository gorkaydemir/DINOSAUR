import os
import sys
import tarfile
import collections
import shutil
import numpy as np
from PIL import Image

from scipy.ndimage import label

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.utils.data as data

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def semantic_to_instance(seg_map, min_size=200):
    """
    Convert a semantic segmentation map to an instance segmentation map.
    
    Args:
    - seg_map (np.array): A 2D numpy array of shape [H, W] where each item corresponds to a semantic class.

    Returns:
    - A 2D numpy array of shape [H, W] where each item corresponds to an instance ID. 0 is reserved for background.
    """

    # Get unique classes excluding background
    classes = np.unique(seg_map)
    if 0 in classes:
        classes = classes[1:]

    instance_map = np.zeros_like(seg_map, dtype=np.uint8)
    current_instance_id = 1

    for cls in classes:
        # Create a binary map for the current class
        binary_map = (seg_map == cls)

        # Label connected components in the binary map
        labeled, num_features = label(binary_map)

        # Assign unique IDs to each instance in the labeled map
        for i in range(1, num_features + 1):
            region_size = np.sum(labeled == i)
            
            if region_size >= min_size:
                instance_map[labeled == i] = current_instance_id
                current_instance_id += 1
            else:
                instance_map[labeled == i] = 0  # Merge with background
    return instance_map


class VOCSegmentation(data.Dataset):
    def __init__(self,
                 args,
                 year='2012_aug',
                 image_set='train'):

        is_aug=False
        if year=='2012_aug':
            is_aug = True
            year = '2012'
        
        self.root = os.path.expanduser(args.root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.args = args
        # self.transform = T.Compose([T.Resize(args.resize_to),
        #                                T.ToTensor(),
        #                                T.Normalize(mean=[0.485, 0.456, 0.406],
        #                                             std=[0.229, 0.224, 0.225])])
        
        validate = (image_set == "val")
        self.transform = T.Compose([T.Resize(min(args.resize_to)),
                                    T.CenterCrop(args.resize_to) if validate else T.RandomCrop(args.resize_to, pad_if_needed=True),
                                    T.RandomHorizontalFlip(p=0 if validate else 0.5),
                                       T.ToTensor(),
                                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        
        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if is_aug and image_set=='train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            split_f = os.path.join("datasets/train_aug.txt")
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        img = self.transform(img)
        
        # semantic_seg = TF.resize(target, size=self.args.resize_to, interpolation=TF.InterpolationMode.NEAREST)
        semantic_seg = TF.resize(target, size=min(self.args.resize_to), interpolation=TF.InterpolationMode.NEAREST)
        semantic_seg = TF.center_crop(semantic_seg, self.args.resize_to)
        semantic_seg = np.array(semantic_seg)
        semantic_seg[semantic_seg == 255] = 0

        instance_seg = semantic_to_instance(semantic_seg)

        return img, instance_seg, semantic_seg


    def __len__(self):
        return len(self.images)