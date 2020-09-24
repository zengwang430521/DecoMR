import torch
from utils.renderer import render_IUV
import numpy as np
import cv2
import utils.config as cfg
from os.path import join, exists
import os
from tqdm import tqdm
from models.smpl import SMPL


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


def cal_projection_err(joint, pose, beta, smpl):
    device = smpl.J_regressor.device
    to_lsp = list(range(14))

    pose = torch.Tensor(pose).to(device)
    beta = torch.Tensor(beta).to(device)
    joint = torch.Tensor(joint).to(device)
    vertices = smpl(pose.unsqueeze(0), beta.unsqueeze(0))
    joint3d = smpl.get_joints(vertices)[0, to_lsp]

    origin_2d = joint3d[:, :2]
    target_2d = joint[to_lsp, :2]
    vis = joint[to_lsp, -1]
    origin_2d = origin_2d[vis > 0]
    target_2d = target_2d[vis > 0]
    size = (target_2d.max(dim=0)[0] - target_2d.min(dim=0)[0]).max()  # The bbox size
    cam, err = cal_cam(origin_2d, target_2d)
    normalized_err = err / (size + 1e-8)
    return normalized_err.detach().cpu().item()


def process_dataset(dataset, is_train, uv_type, smpl, renderer):
    dataset_file = cfg.DATASET_FILES[is_train][dataset]
    data = np.load(dataset_file, allow_pickle=True)
    imgnames = data['imgname']
    centers = data['center']
    scales = data['scale']
    keypoints = data['part']

    flag_fit = False
    if dataset in ['coco', 'lsp-orig', 'mpii', 'lspet', 'mpi-inf-3dhp'] and is_train:
        flag_fit = True
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
    if flag_fit:
        iuv_dir = join(img_dir, '{}_IUV_SPIN_fit'.format(uv_type))
        fit_errors = []
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

        if flag_fit:
            projection_err = 1.0        # fit error = 1 means invalid fits
            if has_smpl[i] > 0:
                joint = keypoints[i]
                pose = poses[i]
                beta = betas[i]
                projection_err = cal_projection_err(joint, pose, beta, smpl)
            fit_errors.append(projection_err)

    save_data = dict(data)
    save_data['iuv_names'] = iuvnames
    if flag_fit:
        save_data['fit_errors'] = fit_errors

    np.savez(dataset_file, **save_data)

    return 0


# The joint of SURREAL is different from other dataset,
# so the generation of IUV image is also a little different from other datasets.
def process_surreal(is_train, uv_type, renderer):
    dataset_file = cfg.DATASET_FILES[is_train]['surreal']
    root_dir = cfg.DATASET_FOLDERS['surreal']
    iuv_dir = join(root_dir, '{}_IUV_gt'.format(uv_type), 'data', 'cmu', 'train')

    smpl_female = SMPL(cfg.FEMALE_SMPL_FILE)
    smpl_male = SMPL(cfg.MALE_SMPL_FILE)
    H = 240
    W = 320
    img_empty = np.zeros([H, W, 3])

    data = np.load(dataset_file, allow_pickle=True)
    shape_list = data['shape']
    pose_list = data['pose']
    gender_list = data['gender']
    part24_list = data['part_smpl']
    videoname_list = data['videoname']
    framenum_list = data['framenum']
    dataset_size = len(data['gender'])
    joint3d_list = data['S_smpl']
    iuvnames = []

    for i in tqdm(range(dataset_size)):

        videoname = videoname_list[i]
        framenum = framenum_list[i]
        iuv_name = videoname[:-4] + '_{}.png'.format(framenum)
        output_path = join(iuv_dir, iuv_name)
        if not exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        iuvnames.append(iuv_name)

        if not exists(output_path):
            shape = shape_list[i]
            pose = pose_list[i]
            gender = gender_list[i]
            part24 = part24_list[i, :, :-1]

            pose_t = torch.from_numpy(pose).float()
            shape_t = torch.from_numpy(shape).float()
            if gender == 'f':
                vertices = smpl_female(pose_t.unsqueeze(0), shape_t.unsqueeze(0))
                joint3d = smpl_female.get_smpl_joints(vertices)[0]
            else:
                vertices = smpl_male(pose_t.unsqueeze(0), shape_t.unsqueeze(0))
                joint3d = smpl_male.get_smpl_joints(vertices)[0]

            target_3d = joint3d_list[i]

            origin_2d = joint3d[:, :2]
            target_2d = torch.tensor(part24).float()

            target_2d[:, 0] = (2 * target_2d[:, 0] - W) / W
            target_2d[:, 1] = (2 * target_2d[:, 1] - H) / W
            cam, err = cal_cam(origin_2d, target_2d)

            uv_tmp = render_IUV(img_empty, vertices[0].detach().cpu().numpy(), cam.detach().cpu().numpy(), renderer)
            uv_im = np.zeros(uv_tmp.shape)
            uv_im[:, :, 0] = 1 - uv_tmp[:, :, 0]
            uv_im[:, :, 1] = uv_tmp[:, :, 1]
            mask_im = uv_im.max(axis=-1) > 0
            mask_im = mask_im[:, :, np.newaxis]

            uv_im_int = np.around(uv_im * 255).astype('uint8')
            mask_im_int = mask_im.astype('uint8')

            iuv_im_out = np.concatenate((mask_im_int, uv_im_int), axis=-1)

            flag_plt = False
            if flag_plt:
                import matplotlib.pyplot as plt
                from skimage.draw import circle
                from models.dense_cnn import warp_feature
                from models.uv_generator import Index_UV_Generator
                uv_sampler = Index_UV_Generator(128, uv_type=uv_type)

                video_dir = join(root_dir, 'data', 'cmu', 'train')
                cap = cv2.VideoCapture(join(video_dir, videoname))
                cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
                a, img = cap.read()
                # the img should be flipped first
                img = np.fliplr(img)[:, :, ::-1].copy().astype(np.float32)

                joint = part24
                for j2d in joint[:, :2]:
                    rr, cc = circle(j2d[1], j2d[0], 2, img.shape[0:2])
                    img[rr, cc] = [255, 0, 0]

                plt.subplot(2, 2, 1)
                plt.imshow(img[:, :, ::-1] / 255)

                plt.subplot(2, 2, 2)
                tmp = iuv_im_out
                plt.imshow(tmp[:, :, ::-1])

                plt.subplot(2, 2, 3)
                iuv = torch.FloatTensor(iuv_im_out)
                iuv[:, :, 1:] = iuv[:, :, 1:] / 255.0
                uv_map = warp_feature(iuv.permute(2, 0, 1).unsqueeze(0),
                                      torch.Tensor(img).permute(2, 0, 1).unsqueeze(0), 128)
                uv_map = uv_map[0, :3].permute(1, 2, 0).cpu().numpy()
                plt.imshow(uv_map[:, :, ::-1] / 255)

                plt.subplot(2, 2, 4)
                texture = uv_sampler.resample(torch.Tensor(uv_map).unsqueeze(0))[0]
                vert = (vertices[0, :, :2].cpu() + cam[1:]) * cam[0]
                vert[:, 0] = (vert[:, 0] * W + W) / 2
                vert[:, 1] = (vert[:, 1] * W + H) / 2

                vert = vert.long()
                back_img = texture.new_zeros(img.shape)
                for v_i in range(vert.shape[0]):
                    back_img[vert[v_i, 1], vert[v_i, 0], :] = back_img[vert[v_i, 1], vert[v_i, 0], :] + texture[v_i, :]
                # back_img[vert[:, 1], vert[:, 0], :] = texture

                plt.imshow(uv_sampler.mask.cpu().numpy())
                plt.imshow(back_img.cpu().numpy()[:, :, ::-1] / 255)

                # back_img = torch.nn.functional.grid_sample(torch.Tensor(uv_map).permute(2,0,1).unsqueeze(0), torch.Tensor(uv_im * 2 - 1).unsqueeze(0))
                # back_img = back_img[0].permute(1,2,0).cpu().numpy()
                # plt.imshow(back_img[:, :, ::-1])

            cv2.imwrite(output_path, iuv_im_out)

    save_data = dict(data)
    save_data['iuv_names'] = iuvnames
    np.savez(dataset_file, **save_data)
    return 0

