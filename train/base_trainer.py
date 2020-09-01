'''
Codes are adapted from https://github.com/nkolot/GraphCMR
'''

from __future__ import division
import sys
import time

import torch
from tqdm import tqdm
tqdm.monitor_interval = 100
from tensorboardX import SummaryWriter
from utils import CheckpointDataLoader, CheckpointSaver
import torch.nn.functional as F

class BaseTrainer(object):
    """
    Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model])
                    print('Checkpoint loaded')

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
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):

                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1

                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, *out)

                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count) 
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count) 
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        pass

    def error_adaptive_weight(self, fit_joint_error):
        weight = (1 - 10 * fit_joint_error)
        weight[weight <= 0] = 0
        return weight

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, weight=None):
        """
        Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the weight
        The available keypoints are different for each dataset.
        """
        if gt_keypoints_2d.shape[2] == 3:
            conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        else:
            conf = 1

        if weight is not None:
            weight = weight[:, None, None]
            conf = conf * weight

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """
        if gt_keypoints_3d.shape[2] == 3:
            tmp = gt_keypoints_3d.new_ones(gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1)
            gt_keypoints_3d = torch.cat((gt_keypoints_3d, tmp), dim=2)

        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]

        if weight is not None:
            weight = weight[has_pose_3d == 1, None, None]
            conf = conf * weight

        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            # Align the origin of the first 24 keypoints with the pelvis.
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

            # # Align the origin of the first 24 keypoints with the pelvis.
            # gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            # pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            # gt_keypoints_3d[:, :24, :] = gt_keypoints_3d[:, :24, :] - gt_pelvis[:, None, :]
            # pred_keypoints_3d[:, :24, :] = pred_keypoints_3d[:, :24, :] - pred_pelvis[:, None, :]
            #
            # # Align the origin of the 24 SMPL keypoints with the root joint.
            # gt_root_joint = gt_keypoints_3d[:, 24]
            # pred_root_joint = pred_keypoints_3d[:, 24]
            # gt_keypoints_3d[:, 24:, :] = gt_keypoints_3d[:, 24:, :] - gt_root_joint[:, None, :]
            # pred_keypoints_3d[:, 24:, :] = pred_keypoints_3d[:, 24:, :] - pred_root_joint[:, None, :]

            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, weight=None):
        """
        Compute 3D SMPL keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the weight
        """
        if gt_keypoints_3d.shape[2] == 3:
            tmp = gt_keypoints_3d.new_ones(gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1)
            gt_keypoints_3d = torch.cat((gt_keypoints_3d, tmp), dim=2)

        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]

        if weight is not None:
            weight = weight[has_pose_3d == 1, None, None]
            conf = conf * weight

        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_root_joint = gt_keypoints_3d[:, 0, :]
            pred_root_joint = pred_keypoints_3d[:, 0, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_root_joint[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_root_joint[:, None, :]
            return (conf * self.criterion_keypoints_3d(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl, weight=None):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]

        if weight is not None:
            weight = weight[has_smpl == 1, None, None]
        else:
            weight = 1

        if len(gt_vertices_with_shape) > 0:
            loss = self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
            loss = (loss * weight).mean()
            return loss
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def uv_loss(self, pred_uv_map, gt_uv_map, has_smpl, weight=None):
        # self.uv_mask = self.uv_mask.to(pred_uv_map.device)
        self.uv_weight = self.uv_weight.to(pred_uv_map.device).type(pred_uv_map.dtype)
        max = self.uv_weight.max()
        pred_uv_map_shape = pred_uv_map[has_smpl == 1]
        gt_uv_map_with_shape = gt_uv_map[has_smpl == 1]
        if len(gt_uv_map_with_shape) > 0:
            # return self.criterion_uv(pred_uv_map_shape * self.uv_mask, gt_uv_map_with_shape * self.uv_mask)
            if weight is not None:
                ada_weight = weight[has_smpl > 0, None, None, None]
            else:
                ada_weight = 1.0
            loss = self.criterion_uv(pred_uv_map_shape * self.uv_weight, gt_uv_map_with_shape * self.uv_weight)
            loss = (loss * ada_weight).mean()
            return loss

        else:
            # return torch.FloatTensor(1).fill_(0.).to(self.device)
            return torch.tensor(0.0, dtype=pred_uv_map.dtype, device=self.device)

    def tv_loss(self, uv_map):
        self.uv_weight = self.uv_weight.to(uv_map.device)
        tv = torch.abs(uv_map[:,0:-1, 0:-1, :] - uv_map[:,0:-1, 1:, :]) \
            + torch.abs(uv_map[:,0:-1, 0:-1, :] - uv_map[:,1:, 0:-1, :])
        return torch.sum(tv) / self.tv_factor
        # return torch.sum(tv * self.uv_weight[:, 0:-1, 0:-1]) / self.tv_factor

    def dp_loss(self, pred_dp, gt_dp, has_dp, weight=None):
        dtype = pred_dp.dtype
        pred_dp_shape = pred_dp[[has_dp > 0]]
        gt_dp_shape = gt_dp[[has_dp > 0]]
        if len(gt_dp_shape) > 0:

            gt_mask_shape = (gt_dp_shape[:, 0].unsqueeze(1) > 0).type(dtype)
            gt_uv_shape = gt_dp_shape[:, 1:]

            pred_mask_shape = pred_dp_shape[:, 0].unsqueeze(1)
            pred_uv_shape = pred_dp_shape[:, 1:]
            pred_mask_shape = F.interpolate(pred_mask_shape, [gt_dp.shape[2], gt_dp.shape[3]], mode='bilinear')
            pred_uv_shape = F.interpolate(pred_uv_shape, [gt_dp.shape[2], gt_dp.shape[3]], mode='nearest')

            if weight is not None:
                weight = weight[has_dp > 0, None, None, None]
            else:
                weight = 1.0

            pred_mask_shape = pred_mask_shape.clamp(min=0.0, max=1.0)
            loss_mask = torch.nn.BCELoss(reduction='none')(pred_mask_shape, gt_mask_shape)
            loss_mask = (loss_mask * weight).mean()

            gt_uv_weight = (gt_uv_shape.abs().max(dim=1, keepdim=True)[0] > 0).type(dtype)
            weight_ratio = (gt_uv_weight.mean(dim=-1).mean(dim=-1)[:, :, None, None] + 1e-8)
            gt_uv_weight = gt_uv_weight / weight_ratio  # normalized the weight according to mask area
            loss_uv = self.criterion_uv(gt_uv_weight * pred_uv_shape, gt_uv_weight * gt_uv_shape)
            loss_uv = (loss_uv * weight).mean()
            return loss_mask, loss_uv
        else:
            return pred_dp.sum() * 0, pred_dp.sum() * 0

    def consistent_loss(self, dp, uv_map, camera, weight=None):

        tmp = torch.arange(0, dp.shape[-1], 1, dtype=dp.dtype, device=dp.device) / (dp.shape[-1] -1)
        tmp = tmp * 2 - 1
        loc_y, loc_x = torch.meshgrid(tmp, tmp)
        loc = torch.stack((loc_x, loc_y), dim=0).expand(dp.shape[0], -1, -1, -1)
        dp_mask = (dp[:, 0] > 0.5).float().unsqueeze(1)
        loc = dp_mask * loc

        dp_tmp = dp_mask * (dp[:, 1:] * 2 - 1)
        '''uv_map need to be transfered to img coordinate first'''
        uv_map = uv_map[:, :, :, :-1]
        camera = camera.view(-1, 1, 1, 3)
        uv_map = uv_map + camera[:, :, :, 1:]       # trans
        uv_map = uv_map * camera[:, :, :, 0].unsqueeze(-1)        # scale
        warp_loc = F.grid_sample(uv_map.permute(0, 3, 1, 2), dp_tmp.permute(0, 2, 3, 1))[:, :2]
        warp_loc = warp_loc * dp_mask

        if weight is not None:
            weight = weight[:, None, None, None]
            dp_mask = dp_mask * weight

        loss_con = torch.nn.MSELoss()(warp_loc * dp_mask, loc * dp_mask)
        return loss_con