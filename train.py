import os
import time
import math
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm 

from models.model import DINOSAURpp, Visual_Encoder

from read_args import get_args, print_args
import utils

def train_epoch(args, vis_encoder, model, optimizer, scheduler, train_dataloader, total_iter, writer):
    total_loss = 0.0
    vis_encoder.eval()
    model.train()

    loader = tqdm(train_dataloader, disable=(args.gpu != 0))

    for i, (frames, _, _) in enumerate(loader):
        frames = frames.cuda(non_blocking=True)      # (B, 3, H, W)

        B = frames.shape[0]

        features = vis_encoder(frames)                      # (B, token, 768)
        reconstruction, _, masks = model(features)              # (B, token, 768)
        loss = F.mse_loss(reconstruction, features.detach())

        total_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if args.gpu == 0:
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            mean_loss = total_loss / (i + 1)
            loader.set_description(f"lr: {lr:.6f} | loss: {mean_loss:.5f}")

            writer.add_scalar("batch/loss", loss.item(), total_iter)
        
        total_iter += 1

    mean_loss = total_loss / (i + 1)
    return mean_loss, total_iter

@torch.no_grad()
def val_epoch(args, vis_encoder, model, val_dataloader, evaluator_inst, evaluator_sem, writer, epoch):
    vis_encoder.eval()
    model.eval()

    loader = tqdm(val_dataloader)

    slot_num = args.num_slots
    ps = args.patch_size

    for i, (model_input, instance_gt, semantic_gt) in enumerate(loader):

        model_input = model_input.cuda(non_blocking=True)                # (B, 3, H, W)
        instance_gt = instance_gt.cuda(non_blocking=True)                      # (B, *, H_t, W_t)
        semantic_gt = semantic_gt.cuda(non_blocking=True)                      # (B, *, H_t, W_t)

        H_t, W_t = instance_gt.shape[-2:]

        H, W = args.resize_to

        features = vis_encoder(model_input)                      # (B, token, 768)
        reconstruction, slots, masks = model(features)           # (B, token, 768), (B, S, D_slot), (B, S, token)
            
        masks = masks.view(-1, slot_num, H // ps, W // ps)                      # (B,  S, H // ps, W // ps)
        predictions = F.interpolate(masks, size=(H_t, W_t), mode="bilinear")    # (B, S, H_t, W_t)
        predictions = torch.argmax(predictions, dim=1)                          # (B, H_t, W_t)

        # === Instance Segmentation Evaluation ===
        miou_i, mbo_i, fgari_i = evaluator_inst.update(predictions, instance_gt)
        miou_s, mbo_s, fgari_s = evaluator_sem.update(predictions, semantic_gt)
        loss_desc = f"mBO_i: {mbo_i:.2f} mBO_s: {mbo_s:.2f}"
        # === Logger ===
        loader.set_description(loss_desc)
        # === === ===

    # === Evaluation Results ====
    miou_i, mbo_i, fgari_i = evaluator_inst.get_results(reset=True)
    miou_s, mbo_s, fgari_s = evaluator_sem.get_results(reset=True)

    # === Logger ===
    print("\n=== Results ===")
    print("Instance Segmentation")
    print(f"\tmIoU: {miou_i:.5f}")
    print(f"\tmBO: {mbo_i:.5f}")
    print(f"\tFG-ARI: {fgari_i:.5f}\n")

    print("Semantic Segmentation")
    print(f"\tmIoU: {miou_s:.5f}")
    print(f"\tmBO: {mbo_s:.5f}")
    print(f"\tFG-ARI: {fgari_s:.5f}")

    # === Tensorboard ===
    writer.add_scalar("object_discovery/mIoU_i", miou_i, epoch)
    writer.add_scalar("object_discovery/mBO_i", mbo_i, epoch)
    writer.add_scalar("object_discovery/FG-ARI_i", fgari_i, epoch)

    writer.add_scalar("object_discovery/mIoU_s", miou_s, epoch)
    writer.add_scalar("object_discovery/mBO_s", mbo_s, epoch)
    writer.add_scalar("object_discovery/FG-ARI_s", fgari_s, epoch)

    return


def main_worker(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print_args(args)

    # === Dataloaders ====
    train_dataloader, val_dataloader = utils.get_dataloaders(args)

    # === Model ===
    vis_encoder = Visual_Encoder(args).cuda()
    model = DINOSAURpp(args).cuda()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # === Training Items ===
    optimizer = torch.optim.Adam(utils.get_params_groups(model), lr=args.learning_rate)
    scheduler = utils.get_scheduler(args, optimizer, train_dataloader)

    # === Misc ===
    evaluator_instance = utils.Evaluator() if args.gpu == 0 else None
    evaluator_semantic = utils.Evaluator() if args.gpu == 0 else None
    writer = utils.get_writer(args) if args.gpu == 0 else None

    print(f"Loss, optimizer and schedulers ready.")

    # === Load from checkpoint ===
    to_restore = {"epoch": 0}
    if args.use_checkpoint:
        utils.restart_from_checkpoint(args, 
                                      run_variables=to_restore, 
                                      model=model,
                                      optimizer=optimizer, 
                                      scheduler=scheduler)
    start_epoch = to_restore["epoch"]


    start_time = time.time()

    dist.barrier()

    print("Starting training!")

    total_iter = 0
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"===== ===== [Epoch {epoch}] ===== =====")

        mean_loss, total_iter = train_epoch(args, vis_encoder, model, optimizer, scheduler, train_dataloader, total_iter, writer)

        # === Save Checkpoint ===
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }

        utils.save_on_master(save_dict, os.path.join(args.model_save_path, "checkpoint.pt"))
        if epoch % 5 == 4:
            utils.save_on_master(save_dict, os.path.join(args.model_save_path, f"checkpoint_epoch_{epoch}.pt"))

        # === Validate ===
        if args.gpu == 0:
            if (epoch == 0) or ((epoch + 1) % args.validation_epoch == 0):
                val_epoch(args, vis_encoder, model, val_dataloader, evaluator_instance, evaluator_semantic, writer, epoch)

            # === Log ===
            writer.add_scalar("epoch/train-loss", mean_loss, epoch)
            writer.flush()
            writer.close()

        dist.barrier()

        print("===== ===== ===== ===== =====\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    dist.destroy_process_group()



if __name__ == '__main__':
    args = get_args()
    main_worker(args)