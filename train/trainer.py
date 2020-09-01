"""
This file includes the full training procedure.
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torchvision.utils import make_grid
from train.base_trainer import BaseTrainer
from datasets import create_dataset
from models import SMPL
from models.dense_cnn import DPNet, get_LNet
from models.geometric_layers import orthographic_projection, rodrigues
from utils.renderer import Renderer, visualize_reconstruction, vis_mesh
from utils import CheckpointDataLoader, CheckpointSaver
import sys
import time
from tqdm import tqdm
from models.uv_generator import Index_UV_Generator, cal_uv_weight
import numpy as np
import torch.nn.functional as F
import os
import utils.config as cfg


class Trainer(BaseTrainer):
    def init_fn(self):
        # create training dataset
        self.train_ds = create_dataset(self.options.dataset, self.options, use_IUV=True)
        self.dp_res = int(self.options.img_res // (2 ** self.options.warp_level))

        self.CNet = DPNet(warp_lv=self.options.warp_level,
                            norm_type=self.options.norm_type).to(self.device)

        self.LNet = get_LNet(self.options).to(self.device)
        self.smpl = SMPL().to(self.device)
        self.female_smpl = SMPL(cfg.FEMALE_SMPL_FILE).to(self.device)
        self.male_smpl = SMPL(cfg.MALE_SMPL_FILE).to(self.device)

        uv_res = self.options.uv_res
        self.uv_type = self.options.uv_type
        self.sampler = Index_UV_Generator(UV_height=uv_res, UV_width=-1, uv_type=self.uv_type).to(self.device)

        weight_file = 'data/weight_p24_h{:04d}_w{:04d}_{}.npy'.format(uv_res, uv_res, self.uv_type)
        if not os.path.exists(weight_file):
            cal_uv_weight(self.sampler, weight_file)

        uv_weight = torch.from_numpy(np.load(weight_file)).to(self.device).float()
        uv_weight = uv_weight * self.sampler.mask.to(uv_weight.device).float()
        uv_weight = uv_weight / uv_weight.mean()
        self.uv_weight = uv_weight[None, :, :, None]
        self.tv_factor = (uv_res -1) * (uv_res - 1)

        # Setup an optimizer
        if self.options.stage == 'dp':
            self.optimizer = torch.optim.Adam(
                params=list(self.CNet.parameters()),
                lr=self.options.lr,
                betas=(self.options.adam_beta1, 0.999),
                weight_decay=self.options.wd)
            self.models_dict = {'CNet': self.CNet}
            self.optimizers_dict = {'optimizer': self.optimizer}

        else:
            self.optimizer = torch.optim.Adam(
                params=list(self.LNet.parameters()) + list(self.CNet.parameters()),
                lr=self.options.lr,
                betas=(self.options.adam_beta1, 0.999),
                weight_decay=self.options.wd)
            self.models_dict = {'CNet': self.CNet, 'LNet': self.LNet}
            self.optimizers_dict = {'optimizer': self.optimizer}

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_uv = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_keypoints_3d = nn.L1Loss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)

        # LSP indices from full list of keypoints
        self.to_lsp = list(range(14))
        self.renderer = Renderer(faces=self.smpl.faces.cpu().numpy())

        # Optionally start training from a pretrained checkpoint
        # Note that this is different from resuming training
        # For the latter use --resume
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def train_step(self, input_batch):
        """Training step."""
        dtype = torch.float32

        if self.options.stage == 'dp':
            self.CNet.train()

            # Grab data from the batch
            has_dp = input_batch['has_dp']
            images = input_batch['img']
            gt_dp_iuv = input_batch['gt_iuv']
            gt_dp_iuv[:, 1:] = gt_dp_iuv[:, 1:] / 255.0
            batch_size = images.shape[0]

            if images.is_cuda and self.options.ngpu > 1:
                pred_dp, dp_feature, codes = data_parallel(self.CNet, images, range(self.options.ngpu))
            else:
                pred_dp, dp_feature, codes = self.CNet(images)

            if self.options.adaptive_weight:
                fit_joint_error = input_batch['fit_joint_error']
                ada_weight = self.error_adaptive_weight(fit_joint_error).type(dtype)
            else:
                # ada_weight = pred_scale.new_ones(batch_size).type(dtype)
                ada_weight = None

            losses = {}
            '''loss on dense pose result'''
            loss_dp_mask, loss_dp_uv = self.dp_loss(pred_dp, gt_dp_iuv, has_dp, ada_weight)
            loss_dp_mask = loss_dp_mask * self.options.lam_dp_mask
            loss_dp_uv = loss_dp_uv * self.options.lam_dp_uv
            losses['dp_mask'] = loss_dp_mask
            losses['dp_uv'] = loss_dp_uv
            loss_total = sum(loss for loss in losses.values())
            # Do backprop
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            # for visualize
            if (self.step_count + 1) % self.options.summary_steps == 0:
                data = {}
                vis_num = min(4, batch_size)
                data['image'] = input_batch['img_orig'][0:vis_num].detach()
                data['pred_dp'] = pred_dp[0:vis_num].detach()
                data['gt_dp'] = gt_dp_iuv[0:vis_num].detach()
                self.vis_data = data

            # Pack output arguments to be used for visualization in a list
            out_args = {key: losses[key].detach().item() for key in losses.keys()}
            out_args['total'] = loss_total.detach().item()
            self.loss_item = out_args

        elif self.options.stage == 'end':
            self.CNet.train()
            self.LNet.train()

            # Grab data from the batch
            # gt_keypoints_2d = input_batch['keypoints']
            # gt_keypoints_3d = input_batch['pose_3d']
            # gt_keypoints_2d = torch.cat([input_batch['keypoints'], input_batch['keypoints_smpl']], dim=1)
            # gt_keypoints_3d = torch.cat([input_batch['pose_3d'], input_batch['pose_3d_smpl']], dim=1)
            gt_keypoints_2d = input_batch['keypoints']
            gt_keypoints_3d = input_batch['pose_3d']
            has_pose_3d = input_batch['has_pose_3d']

            gt_keypoints_2d_smpl = input_batch['keypoints_smpl']
            gt_keypoints_3d_smpl = input_batch['pose_3d_smpl']
            has_pose_3d_smpl = input_batch['has_pose_3d_smpl']

            gt_pose = input_batch['pose']
            gt_betas = input_batch['betas']
            has_smpl = input_batch['has_smpl']
            has_dp = input_batch['has_dp']
            images = input_batch['img']
            gender = input_batch['gender']

            # images.requires_grad_()
            gt_dp_iuv = input_batch['gt_iuv']
            gt_dp_iuv[:, 1:] = gt_dp_iuv[:, 1:] / 255.0
            batch_size = images.shape[0]

            gt_vertices = images.new_zeros([batch_size, 6890, 3])
            if images.is_cuda and self.options.ngpu > 1:
                with torch.no_grad():
                    gt_vertices[gender < 0] = data_parallel(
                        self.smpl, (gt_pose[gender < 0], gt_betas[gender < 0]), range(self.options.ngpu))
                    gt_vertices[gender == 0] = data_parallel(
                        self.male_smpl, (gt_pose[gender == 0], gt_betas[gender == 0]), range(self.options.ngpu))
                    gt_vertices[gender == 1] = data_parallel(
                        self.female_smpl, (gt_pose[gender == 1], gt_betas[gender == 1]), range(self.options.ngpu))
                    gt_uv_map = data_parallel(self.sampler, gt_vertices, range(self.options.ngpu))
                pred_dp, dp_feature, codes = data_parallel(self.CNet, images, range(self.options.ngpu))
                pred_uv_map, pred_camera = data_parallel(self.LNet, (pred_dp, dp_feature, codes),
                                                                         range(self.options.ngpu))
            else:
                # gt_vertices = self.smpl(gt_pose, gt_betas)
                with torch.no_grad():
                    gt_vertices[gender < 0] = self.smpl(gt_pose[gender < 0], gt_betas[gender < 0])
                    gt_vertices[gender == 0] = self.male_smpl(gt_pose[gender == 0], gt_betas[gender == 0])
                    gt_vertices[gender == 1] = self.female_smpl(gt_pose[gender == 1], gt_betas[gender == 1])
                    gt_uv_map = self.sampler.get_UV_map(gt_vertices.float())
                pred_dp, dp_feature, codes = self.CNet(images)
                pred_uv_map, pred_camera = self.LNet(pred_dp, dp_feature, codes)

            if self.options.adaptive_weight:
                # Get the confidence of the GT mesh, which is used as the weight of loss item.
                # The confidence is related to the fitting error and for the data with GT SMPL parameters,
                # the confidence is 1.0
                fit_joint_error = input_batch['fit_joint_error']
                ada_weight = self.error_adaptive_weight(fit_joint_error).type(dtype)
            else:
                ada_weight = None

            losses = {}
            '''loss on dense pose result'''
            loss_dp_mask, loss_dp_uv = self.dp_loss(pred_dp, gt_dp_iuv, has_dp, ada_weight)
            loss_dp_mask = loss_dp_mask * self.options.lam_dp_mask
            loss_dp_uv = loss_dp_uv * self.options.lam_dp_uv
            losses['dp_mask'] = loss_dp_mask
            losses['dp_uv'] = loss_dp_uv

            '''loss on location map'''
            sampled_vertices = self.sampler.resample(pred_uv_map.float()).type(dtype)
            loss_uv = self.uv_loss(gt_uv_map.float(), pred_uv_map.float(), has_smpl, ada_weight).type(
                dtype) * self.options.lam_uv
            losses['uv'] = loss_uv

            if self.options.lam_tv > 0:
                loss_tv = self.tv_loss(pred_uv_map) * self.options.lam_tv
                losses['tv'] = loss_tv

            '''loss on mesh'''
            if self.options.lam_mesh > 0:
                loss_mesh = self.shape_loss(sampled_vertices, gt_vertices, has_smpl, ada_weight) * self.options.lam_mesh
                losses['mesh'] = loss_mesh

            '''loss on joints'''
            weight_key = sampled_vertices.new_ones(batch_size)
            if self.options.gtkey3d_from_mesh:
                # For the data without GT 3D keypoints but with SMPL parameters,
                # we can get the GT 3D keypoints from the mesh.
                # The confidence of the keypoints is related to the confidence of the mesh.
                gt_keypoints_3d_mesh = self.smpl.get_train_joints(gt_vertices)
                gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                                  gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                                 dim=-1)
                valid = has_smpl > has_pose_3d
                gt_keypoints_3d[valid] = gt_keypoints_3d_mesh[valid]
                has_pose_3d[valid] = 1
                if ada_weight is not None:
                    weight_key[valid] = ada_weight[valid]

            sampled_joints_3d = self.smpl.get_train_joints(sampled_vertices)
            loss_keypoints_3d = self.keypoint_3d_loss(sampled_joints_3d, gt_keypoints_3d, has_pose_3d, weight_key)
            loss_keypoints_3d = loss_keypoints_3d * self.options.lam_key3d
            losses['key3D'] = loss_keypoints_3d

            sampled_joints_2d = orthographic_projection(sampled_joints_3d, pred_camera)[:, :, :2]
            loss_keypoints_2d = self.keypoint_loss(sampled_joints_2d, gt_keypoints_2d) * self.options.lam_key2d
            losses['key2D'] = loss_keypoints_2d

            # We add the 24 joints of SMPL model for the training on SURREAL dataset.
            weight_key_smpl = sampled_vertices.new_ones(batch_size)
            if self.options.gtkey3d_from_mesh:
                gt_keypoints_3d_mesh = self.smpl.get_smpl_joints(gt_vertices)
                gt_keypoints_3d_mesh = torch.cat([gt_keypoints_3d_mesh,
                                                  gt_keypoints_3d_mesh.new_ones([batch_size, 24, 1])],
                                                 dim=-1)
                valid = has_smpl > has_pose_3d_smpl
                gt_keypoints_3d_smpl[valid] = gt_keypoints_3d_mesh[valid]
                has_pose_3d_smpl[valid] = 1
                if ada_weight is not None:
                    weight_key_smpl[valid] = ada_weight[valid]

            if self.options.use_smpl_joints:
                sampled_joints_3d_smpl = self.smpl.get_smpl_joints(sampled_vertices)
                loss_keypoints_3d_smpl = self.smpl_keypoint_3d_loss(sampled_joints_3d_smpl, gt_keypoints_3d_smpl,
                                                                    has_pose_3d_smpl, weight_key_smpl)
                loss_keypoints_3d_smpl = loss_keypoints_3d_smpl * self.options.lam_key3d_smpl
                losses['key3D_smpl'] = loss_keypoints_3d_smpl

                sampled_joints_2d_smpl = orthographic_projection(sampled_joints_3d_smpl, pred_camera)[:, :, :2]
                loss_keypoints_2d_smpl = self.keypoint_loss(sampled_joints_2d_smpl, gt_keypoints_2d_smpl) * self.options.lam_key2d_smpl
                losses['key2D_smpl'] = loss_keypoints_2d_smpl

            '''consistent loss'''
            if not self.options.lam_con == 0:
                loss_con = self.consistent_loss(gt_dp_iuv, pred_uv_map, pred_camera, ada_weight) * self.options.lam_con
                losses['con'] = loss_con

            loss_total = sum(loss for loss in losses.values())
            # Do backprop
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            # for visualize
            if (self.step_count + 1) % self.options.summary_steps == 0:
                data = {}
                vis_num = min(4, batch_size)
                data['image'] = input_batch['img_orig'][0:vis_num].detach()
                data['gt_vert'] = gt_vertices[0:vis_num].detach()
                data['pred_vert'] = sampled_vertices[0:vis_num].detach()
                data['pred_cam'] = pred_camera[0:vis_num].detach()
                data['pred_joint'] = sampled_joints_2d[0:vis_num].detach()
                data['gt_joint'] = gt_keypoints_2d[0:vis_num].detach()
                data['pred_uv'] = pred_uv_map[0:vis_num].detach()
                data['gt_uv'] = gt_uv_map[0:vis_num].detach()
                data['pred_dp'] = pred_dp[0:vis_num].detach()
                data['gt_dp'] = gt_dp_iuv[0:vis_num].detach()
                self.vis_data = data

            # Pack output arguments to be used for visualization in a list
            out_args = {key: losses[key].detach().item() for key in losses.keys()}
            out_args['total'] = loss_total.detach().item()
            self.loss_item = out_args

        return out_args

    def train_summaries(self, batch, epoch):
        """Tensorboard logging."""
        if self.options.stage == 'dp':
            dtype = self.vis_data['pred_dp'].dtype
            rend_imgs = []
            vis_size = self.vis_data['pred_dp'].shape[0]
            # Do visualization for the first 4 images of the batch
            for i in range(vis_size):
                img = self.vis_data['image'][i].cpu().numpy().transpose(1, 2, 0)
                H, W, C = img.shape
                rend_img = img.transpose(2, 0, 1)

                gt_dp = self.vis_data['gt_dp'][i]
                gt_dp = torch.nn.functional.interpolate(gt_dp[None, :], size=[H, W])[0]
                # gt_dp = torch.cat((gt_dp, gt_dp.new_ones(1, H, W)), dim=0).cpu().numpy()
                gt_dp = gt_dp.cpu().numpy()
                rend_img = np.concatenate((rend_img, gt_dp), axis=2)

                pred_dp = self.vis_data['pred_dp'][i]
                pred_dp[0] = (pred_dp[0] > 0.5).type(dtype)
                pred_dp[1:] = pred_dp[1:] * pred_dp[0]
                pred_dp = torch.nn.functional.interpolate(pred_dp[None, :], size=[H, W])[0]
                pred_dp = pred_dp.cpu().numpy()
                rend_img = np.concatenate((rend_img, pred_dp), axis=2)

                # import matplotlib.pyplot as plt
                # plt.imshow(rend_img.transpose([1, 2, 0]))
                rend_imgs.append(torch.from_numpy(rend_img))

            rend_imgs = make_grid(rend_imgs, nrow=1)
            self.summary_writer.add_image('imgs', rend_imgs, self.step_count)

        else:
            gt_keypoints_2d = self.vis_data['gt_joint'].cpu().numpy()
            pred_vertices = self.vis_data['pred_vert']
            pred_keypoints_2d = self.vis_data['pred_joint']
            pred_camera = self.vis_data['pred_cam']
            dtype = pred_camera.dtype
            rend_imgs = []
            vis_size = pred_vertices.shape[0]
            # Do visualization for the first 4 images of the batch
            for i in range(vis_size):
                img = self.vis_data['image'][i].cpu().numpy().transpose(1, 2, 0)
                H, W, C = img.shape

                # Get LSP keypoints from the full list of keypoints
                gt_keypoints_2d_ = gt_keypoints_2d[i, self.to_lsp]
                pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, self.to_lsp]
                vertices = pred_vertices[i].cpu().numpy()
                cam = pred_camera[i].cpu().numpy()
                # Visualize reconstruction and detected pose
                rend_img = visualize_reconstruction(img, self.options.img_res, gt_keypoints_2d_, vertices,
                                                    pred_keypoints_2d_, cam, self.renderer)
                rend_img = rend_img.transpose(2, 0, 1)

                if 'gt_vert' in self.vis_data.keys():
                    rend_img2 = vis_mesh(img, self.vis_data['gt_vert'][i].cpu().numpy(), cam, self.renderer, color='blue')
                    rend_img2 = rend_img2.transpose(2, 0, 1)
                    rend_img = np.concatenate((rend_img, rend_img2), axis=2)

                gt_dp = self.vis_data['gt_dp'][i]
                gt_dp = torch.nn.functional.interpolate(gt_dp[None, :], size=[H, W])[0]
                gt_dp = gt_dp.cpu().numpy()
                # gt_dp = torch.cat((gt_dp, gt_dp.new_ones(1, H, W)), dim=0).cpu().numpy()
                rend_img = np.concatenate((rend_img, gt_dp), axis=2)

                pred_dp = self.vis_data['pred_dp'][i]
                pred_dp[0] = (pred_dp[0] > 0.5).type(dtype)
                pred_dp[1:] = pred_dp[1:] * pred_dp[0]
                pred_dp = torch.nn.functional.interpolate(pred_dp[None, :], size=[H, W])[0]
                pred_dp = pred_dp.cpu().numpy()
                rend_img = np.concatenate((rend_img, pred_dp), axis=2)

                # import matplotlib.pyplot as plt
                # plt.imshow(rend_img.transpose([1, 2, 0]))
                rend_imgs.append(torch.from_numpy(rend_img))

            rend_imgs = make_grid(rend_imgs, nrow=1)

            uv_maps = []
            for i in range(vis_size):
                uv_temp = torch.cat((self.vis_data['pred_uv'][i], self.vis_data['gt_uv'][i]), dim=1)
                uv_maps.append(uv_temp.permute(2, 0, 1))

            uv_maps = make_grid(uv_maps, nrow=1)
            uv_maps = uv_maps.abs()
            uv_maps = uv_maps / uv_maps.max()

            # Save results in Tensorboard
            self.summary_writer.add_image('imgs', rend_imgs, self.step_count)
            self.summary_writer.add_image('uv_maps', uv_maps, self.step_count)

        for key in self.loss_item.keys():
            self.summary_writer.add_scalar('loss_' + key, self.loss_item[key], self.step_count)

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.num_epochs):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)

            # Iterate over all batches in an epoch
            batch_len = len(self.train_ds) // self.options.batch_size
            data_stream = tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                               total=len(self.train_ds) // self.options.batch_size,
                               initial=train_data_loader.checkpoint_batch_idx)
            for step, batch in enumerate(data_stream, train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:

                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    loss_dict = self.train_step(batch)
                    self.step_count += 1

                    tqdm_info = 'Epoch:%d| %d/%d ' % (epoch, step, batch_len)
                    for k, v in loss_dict.items():
                        tqdm_info += ' %s:%.4f' % (k, v)
                    data_stream.set_description(tqdm_info)

                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(step, epoch)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0 and self.step_count > 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step + 1,
                                                   self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                                   self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()

                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step,
                                               self.options.batch_size, train_data_loader.sampler.dataset_perm,
                                               self.step_count)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs just iterate over the dataset as usual
            self.checkpoint = None
            # save checkpoint after each 10 epoch
            if (epoch + 1) % 10 == 0:
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                           self.options.batch_size, None, self.step_count)

        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                   self.options.batch_size, None, self.step_count, checkpoint_filename='final')
        return
