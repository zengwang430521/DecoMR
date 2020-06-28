import torch
from utils.renderer import render_IUV
import numpy as np
import cv2
import utils.config as cfg
from os.path import join, exists
import os
from tqdm import tqdm


def cal_cam(origin_2d, target_2d):
    tmp_o = origin_2d - origin_2d.mean(dim=0)
    tmp_t = target_2d - target_2d.mean(dim=0)
    scale = (tmp_t * tmp_o).sum() / (tmp_o * tmp_o).sum()
    trans = target_2d.mean(dim=0) / scale - origin_2d.mean(dim=0)

    err = (origin_2d + trans) * scale - target_2d
    err = err.norm(p=2, dim=1).mean()
    cam = torch.zeros(3)
    cam[0] = scale
    cam[1:] = trans
    return cam, err


def process_image(img, joint, pose, beta, smpl, renderer, uv_type):
    device = smpl.J_regressor.device
    to_lsp = list(range(14))

    H, W, C = img.shape
    pose = torch.Tensor(pose).to(device)
    beta = torch.Tensor(beta).to(device)
    joint = torch.Tensor(joint).to(device)
    vertices = smpl(pose.unsqueeze(0), beta.unsqueeze(0))
    img = img.astype('float') / 255

    joint3d = smpl.get_joints(vertices)[0, to_lsp]

    origin_2d = joint3d[:, :2]
    target_2d = joint[to_lsp, :2]
    vis = joint[to_lsp, -1]
    origin_2d = origin_2d[vis>0]
    target_2d = target_2d[vis>0]

    target_2d[:, 0] = (2 * target_2d[:, 0] - W) / W
    target_2d[:, 1] = (2 * target_2d[:, 1] - H) / W

    cam, err = cal_cam(origin_2d, target_2d)
    uv_tmp = render_IUV(img, vertices[0].detach().cpu().numpy(), cam.detach().cpu().numpy(), renderer)

    uv_im = np.zeros(uv_tmp.shape)
    uv_im[:, :, 0] = 1 - uv_tmp[:, :, 0]
    uv_im[:, :, 1] = uv_tmp[:, :, 1]
    mask_im = uv_im.max(axis=-1) > 0
    mask_im = mask_im[:, :, np.newaxis]

    uv_im_int = np.around(uv_im * 255).astype('uint8')
    mask_im_int = mask_im.astype('uint8')

    iuv_im_out = np.concatenate((mask_im_int, uv_im_int), axis=-1)
    return iuv_im_out


def process_dataset(dataset, is_train, uv_type, smpl, renderer):
    dataset_file = cfg.DATASET_FILES[is_train][dataset]
    data = np.load(dataset_file)
    imgnames = data['imgname']
    centers = data['center']
    scales = data['scale']
    keypoints = data['part']

    if dataset in ['coco', 'lsp-orig', 'mpii', 'lspet', 'mpi-inf-3dhp'] and is_train:
        fit_file = cfg.FIT_FILES[is_train][dataset]
        fit_data = np.load(fit_file)
        poses = fit_data['pose'].astype(np.float)
        betas = fit_data['betas'].astype(np.float)
        has_smpl = fit_data['valid_fit'].astype(np.int)
    else:
        poses = data['pose']
        betas = data['shape']
        has_smpl = np.ones(poses.shape[0])

    img_dir = cfg.DATASET_FOLDERS[dataset]

    iuv_dir = join(img_dir, '{}_IUV_gt'.format(uv_type))

    if dataset in ['coco', 'lsp-orig', 'mpii', 'lspet', 'mpi-inf-3dhp'] and is_train:
        iuv_dir =join(img_dir, '{}_IUV_SPIN_fit'.format(uv_type))

    iuvnames = []
    for i in tqdm(range(len(imgnames))):

        img_path = join(img_dir, imgnames[i])

        center = np.round(centers[i]).astype('int')

        im_name = imgnames[i]
        iuv_name = im_name[:-4] + '_{0}_{1}.png'.format(center[0], center[1])
        iuvnames.append(iuv_name)

        output_path = join(iuv_dir, iuv_name)
        if not exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        if not exists(output_path) and has_smpl[i] > 0:
            im = cv2.imread(img_path)
            joint = keypoints[i]
            pose = poses[i]
            beta = betas[i]
            gt_iuv = process_image(im, joint, pose, beta, smpl, renderer, uv_type)
            cv2.imwrite(output_path, gt_iuv)

    save_data = dict(data)
    save_data['iuv_names'] = iuvnames
    np.savez(dataset_file, **save_data)

    return 0


