
import argparse
import utils.config as cfg
import torch
from models import SMPL
import numpy as np
from utils.renderer import UVRenderer
from utils import objfile

from datasets.preprocess import \
    h36m_extract, \
    coco_extract, \
    lsp_dataset_extract, \
    lsp_dataset_original_extract, \
    mpii_extract, \
    up_3d_extract,\
    process_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
    parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

    parser.add_argument('--gt_iuv', default=False, action='store_true', help='Extract files needed for evaluation')
    parser.add_argument('--uv_type', type=str, default='BF', choices=['BF', 'SMPL'])
    parser.add_argument('--from_end', dest='from_end', default=False, action='store_true', help='from_end')
    parser.add_argument('--begin_index', type=int, default=0)

    args = parser.parse_args()

    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH

    if args.train_files:
        # UP-3D dataset preprocessing (trainval set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'trainval')

        # LSP dataset original preprocessing (training set)
        lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, out_path)

        # MPII dataset preprocessing
        mpii_extract(cfg.MPII_ROOT, out_path)

        # COCO dataset prepreocessing
        coco_extract(cfg.COCO_ROOT, out_path)

    if args.eval_files:

        h36m_extract(cfg.H36M_ROOT_ORIGIN, out_path, protocol=1, extract_img=True)
        h36m_extract(cfg.H36M_ROOT_ORIGIN, out_path, protocol=2, extract_img=False)

        # LSP dataset preprocessing (test set)
        lsp_dataset_extract(cfg.LSP_ROOT, out_path)

        # UP-3D dataset preprocessing (lsp_test set)
        up_3d_extract(cfg.UP_3D_ROOT, out_path, 'lsp_test')

    if args.gt_iuv:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        smpl = SMPL().to(device)
        uv_type = args.uv_type

        if uv_type == 'SMPL':
            data = objfile.read_obj_full('data/uv_sampler/smpl_fbx_template.obj')
        elif uv_type == 'BF':
            data = objfile.read_obj_full('data/uv_sampler/smpl_boundry_free_template.obj')

        vt = np.array(data['texcoords'])
        face = [f[0] for f in data['faces']]
        face = np.array(face) - 1
        vt_face = [f[2] for f in data['faces']]
        vt_face = np.array(vt_face) - 1
        renderer = UVRenderer(faces=face, tex=np.zeros([256, 256, 3]), vt=1 - vt, ft=vt_face)

        for dataset_name in ['up-3d', 'h36m-train']:
            process_dataset(dataset_name, is_train=True, uv_type=uv_type, smpl=smpl, renderer=renderer)
