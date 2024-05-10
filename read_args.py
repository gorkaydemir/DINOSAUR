import sys
import argparse

import torch

def set_remaining_args(args):
    args.gpus = torch.cuda.device_count()

    args.patch_size = int(args.encoder.split("-")[-1])
    args.token_num = (args.resize_to[0] * args.resize_to[1]) // (args.patch_size ** 2)

def print_args(args):
    print("====== Arguments ======")
    print(f"training name: {args.model_save_path.split('/')[-1]}\n")

    print(f"dataset: {args.dataset}")
    print(f"resize_to: {args.resize_to}\n")

    print(f"encoder: {args.encoder}\n")

    print(f"num_slots: {args.num_slots}")
    print(f"slot_att_iter: {args.slot_att_iter}")
    print(f"slot_dim: {args.slot_dim}")
    print(f"query_opt: {args.query_opt}")
    print(f"ISA: {args.ISA}\n")

    print(f"learning_rate: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_epochs: {args.num_epochs}")
    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("Dinosaur++")

    # === Data Related Parameters ===
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="pascal_voc12", choices=["pascal_voc12", "coco"]) 
    # /datasets/pascal_voc2012
    # /datasets/COCO
    parser.add_argument('--resize_to',  nargs='+', type=int, default=[320, 320])

    # === ViT Related Parameters ===
    parser.add_argument('--encoder', type=str, default="dinov2-vitb-14", 
                        choices=["dinov2-vitb-14", "dino-vitb-16", "dino-vitb-8", "sup-vitb-16"])

    # === Slot Attention Related Parameters ===
    parser.add_argument('--num_slots', type=int, default=7)
    parser.add_argument('--slot_att_iter', type=int, default=3)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--query_opt', action="store_true")
    parser.add_argument('--ISA', action="store_true")


    # === Training Related Parameters ===    
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)

    # === Misc ===
    parser.add_argument('--use_checkpoint', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--validation_epoch', type=int, default=10)

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_save_path', type=str, required=True)

    args = parser.parse_args()

    set_remaining_args(args)

    return args