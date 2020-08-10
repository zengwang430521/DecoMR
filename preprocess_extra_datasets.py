
import argparse
import utils.config as cfg
import torch
from models import SMPL
import numpy as np
from utils.renderer import UVRenderer
from utils import objfile

from datasets.preprocess import \
    process_dataset, process_surreal,\
    pw3d_extract,\
    hr_lspet_extract,\
    mpi_inf_3dhp_extract,\
    extract_surreal_eval, extract_surreal_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
    parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')
    parser.add_argument('--gt_iuv', default=False, action='store_true', help='Extract files needed for evaluation')
    parser.add_argument('--uv_type', type=str, default='BF', choices=['BF', 'SMPL'])

    args = parser.parse_args()

    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = None
    if args.train_files:
        # SURREAL dataset preprocessing (training set)
        extract_surreal_train(cfg.SURREAL_ROOT, out_path)

        # MPI-INF-3DHP dataset preprocessing (training set)
        mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=True, static_fits=None)

        # LSP Extended training set preprocessing - HR version
        hr_lspet_extract(cfg.LSPET_ROOT, openpose_path, out_path)

    if args.eval_files:
        # SURREAL dataset preprocessing (validation set)
        extract_surreal_eval(cfg.SURREAL_ROOT, out_path)

        # 3DPW dataset preprocessing (test set)
        pw3d_extract(cfg.PW3D_ROOT, out_path)

        # MPI-INF-3DHP dataset preprocessing (test set)
        mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'test')

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

        process_surreal(is_train=True, uv_type=uv_type, renderer=renderer)

        for dataset_name in ['lspet', 'coco', 'lsp-orig', 'mpii', 'lspet', 'mpi-inf-3dhp']:
            process_dataset(dataset_name, is_train=True, uv_type=uv_type, smpl=smpl, renderer=renderer)
