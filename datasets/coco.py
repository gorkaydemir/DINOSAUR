import os
import sys
import tarfile
import collections
import shutil
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.utils.data as data

from tqdm import trange

from pycocotools.coco import COCO
from pycocotools import mask

class COCOSegmentation(data.Dataset):
    def __init__(self, args, image_set="train"):
        super().__init__()
        
        self.root  = args.root
        self.image_set = image_set
        self.args = args
        
        ann_file = os.path.join(self.root, f"annotations/instances_{image_set}2017.json")
        self.img_dir = os.path.join(self.root, f"{image_set}2017")
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        

        self.ids = self.preprocess()
        
        validate = (image_set == "val")
        self.transform = T.Compose([T.Resize(min(args.resize_to)),
                                    T.CenterCrop(args.resize_to),
                                    T.RandomHorizontalFlip(p=0 if validate else 0.5),
                                       T.ToTensor(),
                                       T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        

    def __getitem__(self, index):
        img, semantic_seg, instance_seg = self.make_img_gt_point_pair(index)

        img = self.transform(img)

        semantic_seg = TF.resize(semantic_seg, size=min(self.args.resize_to), interpolation=TF.InterpolationMode.NEAREST)
        semantic_seg = TF.center_crop(semantic_seg, self.args.resize_to)
        semantic_seg = np.array(semantic_seg)

        instance_seg = TF.resize(instance_seg, size=min(self.args.resize_to), interpolation=TF.InterpolationMode.NEAREST)
        instance_seg = TF.center_crop(instance_seg, self.args.resize_to)
        instance_seg = np.array(instance_seg)
        
        return img, instance_seg, semantic_seg

    def make_img_gt_point_pair(self, index):
        img_id = self.ids[index]
        img_metadata = self.coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        
        sem, ins = self.gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
        sem = Image.fromarray(sem)
        ins = Image.fromarray(ins)

        return _img, sem, ins

    def preprocess(self):
        ids = list(self.coco.imgs.keys())
        new_ids = []
        for i in range(len(ids)):
            img_id = ids[i]
            new_ids.append(img_id)
        
        new_ids = sorted(new_ids)
        return new_ids
    
    def gen_seg_mask(self, target, h, w):
        mask_semantic = np.zeros((h, w), dtype=np.uint8)
        mask_instance = np.zeros((h, w), dtype=np.uint8)
        mask_overlap = np.zeros((h, w), dtype=np.uint8)
        
        coco_mask = self.coco_mask
        
        target = sorted(target, key=lambda x: x["id"])
        
        i = 1
        for instance in target:
            if instance["iscrowd"]:
                continue
                
            rle = coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = coco_mask.decode(rle)
            c = instance["category_id"]
            
            assert m.shape[:2] == (h, w), f"m.shape: {m.shape}"
    
            if len(m.shape) >= 3:
                m = (np.sum(m, axis=2) > 0)

            mask_semantic += (mask_semantic == 0) * (m * c).astype(np.uint8)
            mask_instance += (mask_instance == 0) * (m * i).astype(np.uint8)
            mask_overlap += m.astype(np.uint8)
            
            i += 1

            
        overlap_id = 255
        mask_overlap = ((mask_overlap > 1) * overlap_id).astype(np.uint8)

        mask_semantic = (mask_semantic * (mask_overlap != overlap_id))
        mask_instance = (mask_instance * (mask_overlap != overlap_id))
        
        return mask_semantic, mask_instance
    

    def __len__(self):
        return len(self.ids)


