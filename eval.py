#!/usr/bin/python
"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python eval.py --checkpoint=data/models/model_checkpoint_h36m_up3d.pt --config=data/config.json --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. UP-3D ```--dataset=up-3d```
4. LSP ```--dataset=lsp```
"""
from __future__ import print_function
from __future__ import division
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm

import utils.config as cfg
from datasets.base_dataset import BaseDataset
from datasets.surreal_dataset import SurrealDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
from os.path import join, exists
from models import SMPL
from models.dmr import DMR


# Define command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='DecoMR', choices=['DecoMR'])
parser.add_argument('--checkpoint', help='Path to network checkpoint')
parser.add_argument('--config', default=None, help='Path to config file containing model architecture etc.')
parser.add_argument('--dataset', default='h36m-p2', help='eval dataset')

parser.add_argument('--log_freq', default=20, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')

parser.add_argument('--gt_root', type=str, default='/home/wzeng/mydata/h3.6m/test')
parser.add_argument('--save_root', type=str, default='./results')
parser.add_argument('--ngpu', type=int, default=1)


def run_evaluation(model, opt, options, dataset_name, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Create SMPL model
    smpl = SMPL().to(device)
    if dataset_name == '3dpw' or dataset_name == 'surreal':
        smpl_male = SMPL(cfg.MALE_SMPL_FILE).to(device)
        smpl_female = SMPL(cfg.FEMALE_SMPL_FILE).to(device)

    batch_size = opt.batch_size

    # Create dataloader for the dataset
    if dataset_name == 'surreal':
        dataset = SurrealDataset(options, use_augmentation=False, is_train=False, use_IUV=False)
    else:
        dataset = BaseDataset(options, dataset_name, use_augmentation=False, is_train=False, use_IUV=False)

    data_loader = DataLoader(dataset,  batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.num_workers),
                             pin_memory=True)

    print('data loader finish')

    # Transfer model to the GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    mpjpe_pa = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2, 1))
    fp = np.zeros((2, 1))
    fn = np.zeros((2, 1))
    parts_tp = np.zeros((7, 1))
    parts_fp = np.zeros((7, 1))
    parts_fn = np.zeros((7, 1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    eval_pose = False
    eval_shape = False
    eval_masks = False
    eval_parts = False
    joint_mapper = cfg.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else cfg.J24_TO_J14
    # Choose appropriate evaluation for each dataset
    if 'h36m' in dataset_name or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name in ['up-3d', 'surreal']:
        eval_shape = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = cfg.DATASET_FOLDERS['upi-s1h']

    if eval_parts or eval_masks:
        from utils.part_utils import PartRenderer
        renderer = PartRenderer()

    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl(gt_pose, gt_betas)
        images = batch['img'].to(device)

        curr_batch_size = images.shape[0]

        # Run inference
        with torch.no_grad():
            out_dict = model(images)

        pred_vertices = out_dict['pred_vertices']
        camera = out_dict['camera']
        # 3D pose evaluation
        if eval_pose:
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper, :-1]
                gt_pelvis = (gt_keypoints_3d[:, [2]] + gt_keypoints_3d[:, [3]]) / 2
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            else:
                gender = batch['gender'].to(device)
                gt_vertices = smpl_male(gt_pose, gt_betas)
                gt_vertices_female = smpl_female(gt_pose, gt_betas)
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]

                gt_keypoints_3d = smpl.get_train_joints(gt_vertices)[:, joint_mapper]
                # gt_keypoints_3d = smpl.get_lsp_joints(gt_vertices)    # joints_regressor used in cmr
                gt_pelvis = (gt_keypoints_3d[:, [2]] + gt_keypoints_3d[:, [3]]) / 2
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # Get 14 predicted joints from the non-parametic mesh
            pred_keypoints_3d = smpl.get_train_joints(pred_vertices)[:, joint_mapper]
            # pred_keypoints_3d = smpl.get_lsp_joints(pred_vertices)    # joints_regressor used in cmr
            pred_pelvis = (pred_keypoints_3d[:, [2]] + pred_keypoints_3d[:, [3]]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                           reduction=None)
            mpjpe_pa[step * batch_size:step * batch_size + curr_batch_size] = r_error

        # Shape evaluation (Mean per-vertex error)
        if eval_shape:
            if dataset_name == 'surreal':
                gender = batch['gender'].to(device)
                gt_vertices = smpl_male(gt_pose, gt_betas)
                gt_vertices_female = smpl_female(gt_pose, gt_betas)
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]

            gt_pelvis_mesh = smpl.get_eval_joints(gt_vertices)
            pred_pelvis_mesh = smpl.get_eval_joints(pred_vertices)
            gt_pelvis_mesh = (gt_pelvis_mesh[:, [2]] + gt_pelvis_mesh[:, [3]]) / 2
            pred_pelvis_mesh = (pred_pelvis_mesh[:, [2]] + pred_pelvis_mesh[:, [3]]) / 2

            # se = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            se = torch.sqrt(((pred_vertices - pred_pelvis_mesh - gt_vertices + gt_pelvis_mesh) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            shape_err[step * batch_size:step * batch_size + curr_batch_size] = se

        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = renderer(pred_vertices, camera)
        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                gt_mask = cv2.imread(os.path.join(annot_path, batch['maskname'][i]), 0) > 0
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] += (~cgt & cpred).sum()
                    fn[c] += (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                    cgt = gt_parts == c
                    cpred = pred_parts == c
                    cpred[gt_parts == 255] = 0
                    parts_tp[c] += (cgt & cpred).sum()
                    parts_fp[c] += (~cgt & cpred).sum()
                    parts_fn[c] += (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('MPJPE-PA: ' + str(1000 * mpjpe_pa[:step * batch_size].mean()))
                print()
            if eval_shape:
                print('Shape Error: ' + str(1000 * shape_err[:step * batch_size].mean()))
                print()
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean())
                print()

    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('MPJPE-PA: ' + str(1000 * mpjpe_pa.mean()))
        print()
    if eval_shape:
        print('Shape Error: ' + str(1000 * shape_err.mean()))
        print()
    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()
    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean())
        print()

    # Save final results to .txt file
    txt_name = join(opt.save_root, dataset_name + '.txt')
    f = open(txt_name, 'w')
    f.write('*** Final Results ***')
    f.write('\n')
    if eval_pose:
        f.write('MPJPE: ' + str(1000 * mpjpe.mean()))
        f.write('\n')
        f.write('MPJPE-PA: ' + str(1000 * mpjpe_pa.mean()))
        f.write('\n')
    if eval_shape:
        f.write('Shape Error: ' + str(1000 * shape_err.mean()))
        f.write('\n')
    if eval_masks:
        f.write('Accuracy: ' + str(accuracy / pixel_count))
        f.write('\n')
        f.write('F1: ' + str(f1.mean()))
        f.write('\n')
    if eval_parts:
        f.write('Parts Accuracy: ' + str(parts_accuracy / parts_pixel_count))
        f.write('\n')
        f.write('Parts F1 (BG): ' + str(parts_f1[[0, 1, 2, 3, 4, 5, 6]].mean()))
        f.write('\n')


if __name__ == '__main__':
    args = parser.parse_args()

    if not exists(args.save_root):
        os.makedirs(args.save_root)

    if args.config is None:
        args.config = join(os.path.dirname(args.checkpoint), '../config.json')

    with open(args.config, 'r') as f:
        options = json.load(f)
        options = namedtuple('options', options.keys())(**options)

    model = DMR(options, args.checkpoint, args.ngpu)
    model.eval()
    run_evaluation(model, args, options, args.dataset, args.log_freq)



