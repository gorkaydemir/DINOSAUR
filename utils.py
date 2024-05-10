import os
import sys
import math
import random

import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import shutil
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

from datasets import pascal, coco

def get_dataloaders(args):
    if args.dataset == "pascal_voc12":
        train_dataset = pascal.VOCSegmentation(args, "2012_aug", image_set="train")
        val_dataset = pascal.VOCSegmentation(args, "2012_aug", image_set="val")

        assert len(train_dataset) == 10582, f"len(train_dataset): {len(train_dataset)}"
        assert len(val_dataset) == 1449, f"len(train_dataset): {len(val_dataset)}"

    elif args.dataset == "coco":
        train_dataset = coco.COCOSegmentation(args, image_set="train")
        val_dataset = coco.COCOSegmentation(args, image_set="val")

        assert len(train_dataset) == 118287, f"len(train_dataset): {len(train_dataset)}"
        assert len(val_dataset) == 5000, f"len(train_dataset): {len(val_dataset)}"

    elif args.dataset == "movi":
        pass
    elif args.dataset == "ytvis":
        pass
    else:
        print("Not available dataset")


    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=args.gpu, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size // args.gpus,
        num_workers=5,      # cpu per gpu
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=5, 
        drop_last=False, 
        pin_memory=True)
     
    return train_dataloader, val_dataloader


# === Evaluation Related ===

class Evaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.object_numbers = []
        self.mious = []
        self.mbos = []
        self.fg_aris = []

    def iou_single(self, pred, gt):
        # :arg pred_map: (320, 320)
        # :arg gt_map: (320, 320)

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        
        if union == 0:
            return 0  # If there's no ground truth, consider IoU = 0
        
        iou = intersection / union
        return iou
    
    def get_miou(self, pred_map, gt_map):
        # :arg pred_map: (320, 320)
        # :arg gt_map: (320, 320)

        unique_pred = np.unique(pred_map)
        unique_gt = np.unique(gt_map)
        
        iou_matrix = np.zeros((len(unique_pred), len(unique_gt)))
        for i, pred_id in enumerate(unique_pred):
            for j, gt_id in enumerate(unique_gt):
                if gt_id == 0:  # Skip background
                    continue
                iou_matrix[i, j] = -self.iou_single(pred_map == pred_id, gt_map == gt_id)  # Negative because we want to maximize
        
        row_inds, col_inds = linear_sum_assignment(iou_matrix)
        matched_ious = -iou_matrix[row_inds, col_inds]  # Retrieve the original positive IoU values
    
        return matched_ious.mean()
    
    # def get_miou_one_hot(self, pred_map, gt_map):
    #     # :arg pred_map: (320, 320)
    #     # :arg gt_map: (320, 320, C)

    #     unique_pred = np.unique(pred_map)
    #     gt_num = gt_map.shape[-1]
        
    #     iou_matrix = np.zeros((len(unique_pred), gt_num))
    #     for i, pred_id in enumerate(unique_pred):
    #         for j in range(gt_num):
    #             iou_matrix[i, j] = -self.iou_single(pred_map == pred_id, gt_map[:, :, j])  # Negative because we want to maximize
        
    #     row_inds, col_inds = linear_sum_assignment(iou_matrix)
    #     matched_ious = -iou_matrix[row_inds, col_inds]  # Retrieve the original positive IoU values
    
    #     return matched_ious.mean()
    
    def get_mbo(self, pred_map, gt_map):
        # :arg pred_map: (320, 320)
        # :arg gt_map: (320, 320)

        unique_pred = np.unique(pred_map)
        unique_gt = np.unique(gt_map)
        
        best_overlaps = []
        for gt_id in unique_gt:
            if gt_id == 0:  # Skip background
                continue

            max_iou = 0
            for pred_id in unique_pred:
                iou = self.iou_single(pred_map == pred_id, gt_map == gt_id)
                if iou > max_iou:
                    max_iou = iou
            best_overlaps.append(max_iou)

        if len(best_overlaps) == 0:
            return 0
        return np.mean(best_overlaps)
    
    # def get_mbo_onehot(self, pred_map, gt_map):
    #     # :arg pred_map: (320, 320)
    #     # :arg gt_map: (320, 320, C)

    #     unique_pred = np.unique(pred_map)
    #     gt_num = gt_map.shape[-1]
        
    #     best_overlaps = []
    #     for gt_id in range(gt_num):
    #         max_iou = 0
    #         for pred_id in unique_pred:
    #             iou = self.iou_single(pred_map == pred_id, gt_map[:, :, gt_id])
    #             if iou > max_iou:
    #                 max_iou = iou
    #         best_overlaps.append(max_iou)

    #     if len(best_overlaps) == 0:
    #         return 0
    #     return np.mean(best_overlaps)
    
    def get_fgari(self, pred_map, gt_map):
        # :arg pred_map: (320, 320)
        # :arg gt_map: (320, 320)

        pred_map = np.ravel(pred_map[gt_map != 0])
        gt_map = np.ravel(gt_map[gt_map != 0])
        ari_score = adjusted_rand_score(pred_map, gt_map)
        return ari_score
    
    # def get_fgari_onehot(self, pred_map, gt_map):
    #     # :arg pred_map: (320, 320)
    #     # :arg gt_map: (320, 320, C)

    #     gt_map = np.argmax(gt_map, axis=-1)
    #     pred_map = np.ravel(pred_map[gt_map != 0])
    #     gt_map = np.ravel(gt_map[gt_map != 0])
    #     ari_score = adjusted_rand_score(pred_map, gt_map)
    #     return ari_score

    def update(self, pred_map, gt_map):
        # :args pred_map: (H, W)
        # :args gt_map: (H, W)

        pred_map = pred_map.detach().cpu().numpy()
        gt_map = gt_map.detach().cpu().numpy()

        fgari = self.get_fgari(pred_map, gt_map)
        mbo = self.get_mbo(pred_map, gt_map)
        miou = self.get_miou(pred_map, gt_map)

        self.fg_aris.append(fgari)
        self.mbos.append(mbo)
        self.mious.append(miou)

        return self.get_results()
       

    def get_results(self, reset=False):

        miou = sum(self.mious) / len(self.mious)
        mbo = sum(self.mbos) / len(self.mbos)
        fgari = sum(self.fg_aris) / len(self.fg_aris)

        if reset:
            self.reset()

        return miou * 100, mbo * 100, fgari * 100
    

def get_writer(args):
    writer_path = os.path.join(args.model_save_path, "writer.log")
    if os.path.exists(writer_path):
        shutil.rmtree(writer_path)

    comment = f"lr: {args.learning_rate:.5f} bs: {args.batch_size}"
    writer = SummaryWriter(log_dir=writer_path, comment=comment)

    return writer

# === Training Related ===

def restart_from_checkpoint(args, run_variables, **kwargs):

    checkpoint_path = args.checkpoint_path

    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path)

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint with msg {}".format(key, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint".format(key))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint".format(key))
        else:
            print("=> key '{}' not found in checkpoint".format(key))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def get_scheduler(args, optimizer, train_loader):
    T_max = len(train_loader) * args.num_epochs
    warmup_steps = int(T_max * 0.05)
    steps = T_max - warmup_steps
    gamma = math.exp(math.log(0.5) / (steps // 3))

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=warmup_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, scheduler], milestones=[warmup_steps])
    return scheduler

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

# === Distributed Settings ===

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L467C1-L499C42

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = args.gpus
        args.gpu = args.rank % torch.cuda.device_count()

    # launched naively with `python train.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, 'env://'), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L452C1-L464C30
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)