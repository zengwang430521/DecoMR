'''
Codes are adapted from https://github.com/nkolot/GraphCMR
'''

from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join
import torch.nn.functional as F
import utils.config as cfg
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import os


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """
    def __init__(self, options, dataset, use_augmentation=True, is_train=True, use_IUV=False):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = cfg.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
        self.data = np.load(cfg.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # If False, do not do augmentation
        self.use_augmentation = use_augmentation

        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            self.has_smpl = np.ones(len(self.imgname)).astype(np.int)
            if dataset == 'mpi-inf-3dhp':
                self.has_smpl = self.data['has_smpl'].astype(np.int)
                t = self.has_smpl.mean()
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname)).astype(np.int)

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            self.keypoints = self.data['part']
        except KeyError:
            self.keypoints = np.zeros((len(self.imgname), 24, 3))

        self.length = self.scale.shape[0]
        self.use_IUV = use_IUV
        self.has_dp = np.zeros(len(self.imgname))

        if self.use_IUV:
            if self.dataset in ['h36m-train', 'up-3d', 'h36m-test', 'h36m-train-hmr']:
                self.iuvname = self.data['iuv_names']
                self.has_dp = self.has_smpl
                self.uv_type = options.uv_type
                self.iuv_dir = join(self.img_dir, '{}_IUV_gt'.format(self.uv_type))

        # Using fitted SMPL parameters from SPIN or not
        if self.is_train and options.use_spin_fit and self.dataset in ['coco', 'lsp-orig', 'mpii', 'lspet', 'mpi-inf-3dhp']:
            fit_file = cfg.FIT_FILES[is_train][self.dataset]
            fit_data = np.load(fit_file)
            self.pose = fit_data['pose'].astype(np.float)
            self.betas = fit_data['betas'].astype(np.float)
            self.has_smpl = fit_data['valid_fit'].astype(np.int)

            if self.use_IUV:
                self.uv_type = options.uv_type
                self.iuvname = self.data['iuv_names']
                self.has_dp = self.has_smpl
                self.fit_joint_error = self.data['fit_errors'].astype(np.float32)
                self.iuv_dir = join(self.img_dir, '{}_IUV_SPIN_fit'.format(self.uv_type))

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            if self.options.use_augmentation:
                # We flip with probability 1/2
                if np.random.uniform() <= 0.5:
                    flip = 1

                # Each channel is multiplied with a number
                # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
                pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)

                # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
                rot = min(2*self.options.rot_factor,
                        max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))

                # The scale is multiplied with a number
                # in the area [1-scaleFactor,1+scaleFactor]
                sc = min(1+self.options.scale_factor,
                        max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
                # but it is zero with probability 3/5
                if np.random.uniform() <= 0.6:
                    rot = 0

        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))

        rgb_img = torch.FloatTensor(rgb_img).permute(2, 0, 1) / 255.0
        channel, H, W = rgb_img.shape
        theta = torch.FloatTensor([[200.0 * scale / W, 0, (2.0 * center[0]) / W - 1],
                                   [0, 200.0 * scale / H, (2.0 * center[1]) / H - 1]])

        if rot != 0:
            rot_rad = -1 * -1 * rot * np.pi / 180.0
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            theta_rot = torch.FloatTensor([[cs, -sn, 0],
                                           [sn, cs, 0],
                                           [0, 0, 1]])
            theta = torch.mm(theta, theta_rot)

        # flip the IUV map
        if flip:
            theta[:, 0] = - theta[:, 0]

        theta = theta.unsqueeze(0)
        sample_grid = F.affine_grid(theta, [1, channel, self.options.img_res, self.options.img_res])
        rgb_img = F.grid_sample(rgb_img.unsqueeze(0).float(), sample_grid, mode='bilinear', padding_mode='zeros')
        rgb_img = rgb_img.squeeze(0)
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [self.options.img_res, self.options.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.options.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S = np.einsum('ij,kj->ki', rot_mat, S) 
        # flip the x coordinates
        if f:
             S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def iuv_processing(self, IUV, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        IUV = torch.from_numpy(IUV)
        IUV = IUV.permute(2, 0, 1)
        channel, H, W = IUV.shape
        theta = torch.FloatTensor([[200.0 * scale / W, 0, (2.0 * center[0]) / W - 1],
                                   [0, 200.0 * scale / H, (2.0 * center[1]) / H - 1]])

        if rot != 0:
            rot_rad = -1 * -1 * rot * np.pi / 180.0
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            theta_rot = torch.FloatTensor([[cs, -sn,    0],
                                           [sn, cs,     0],
                                           [0,  0,      1]])
            theta = torch.mm(theta, theta_rot)

        # flip the UV map
        if flip:
            theta[:, 0] = - theta[:, 0]

        theta = theta.unsqueeze(0)
        sample_grid = F.affine_grid(theta, [1, IUV.shape[0], self.options.img_res, self.options.img_res])
        IUV = F.grid_sample(IUV.float().unsqueeze(0), sample_grid, mode='nearest', padding_mode='zeros')
        IUV = IUV.squeeze(0)
        # not realized yet
        if flip:
            if self.uv_type == 'BF':
                mask = (IUV[0] > 0).float()
                IUV[1] = (255 - IUV[1]) * mask
            else:
                print('Flip augomentation for SMPL defalt UV map is not supported yet.')

        return IUV

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = join(self.img_dir, str(self.imgname[index]))
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        # Store image before normalization to use it in visualization
        item['img_orig'] = img.clone()
        item['img'] = self.normalize_img(img)
        item['imgname'] = imgname

        # Get SMPL parameters, if available
        has_smpl = self.has_smpl[index]
        item['has_smpl'] = has_smpl
        if has_smpl:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()

        # Get 3D pose, if available
        item['has_pose_3d'] = self.has_pose_3d
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            St = self.j3d_processing(S.copy()[:,:-1], rot, flip)
            S[:,:-1] = St
            item['pose_3d'] = torch.from_numpy(S).float()
        else:
            item['pose_3d'] = torch.zeros(24, 4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()

        # Get GT SMPL joints (For the compatibility with SURREAL dataset)
        item['keypoints_smpl'] = torch.zeros(24, 3, dtype=torch.float32)
        item['pose_3d_smpl'] = torch.zeros(24, 4, dtype=torch.float32)
        item['has_pose_3d_smpl'] = 0

        # Pass path to segmentation mask, if available
        # Cannot load the mask because each mask has different size, so they cannot be stacked in one tensor
        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''
        item['gender'] = self.gender[index]

        if self.use_IUV:
            IUV = torch.zeros([3, img.shape[1], img.shape[2]], dtype=torch.float)
            iuvname = ''
            has_dp = self.has_dp[index]
            try:
                fit_error = self.fit_joint_error[index]
            except AttributeError:
                fit_error = 0.0         # For the dataset with GT mesh, fit_error is set 0

            if has_dp:
                iuvname = join(self.iuv_dir, str(self.iuvname[index]))
                if os.path.exists(iuvname):
                    IUV = cv2.imread(iuvname).copy()
                    IUV = self.iuv_processing(IUV, center, sc * scale, rot, flip, pn)  # process UV map
                else:
                    has_dp = 0
                    print("GT IUV image: {} does not exist".format(iuvname))

            item['gt_iuv'] = IUV
            item['iuvname'] = iuvname
            item['has_dp'] = has_dp
            item['fit_joint_error'] = fit_error

        return item

    def __len__(self):
        return len(self.imgname)



