import torch
import torch.nn as nn
from .uv_generator import Index_UV_Generator
from models.dense_cnn import DPNet, get_LNet
from torch.nn.parallel import data_parallel


class DMR(nn.Module):

    def __init__(self, options, pretrained_checkpoint=None, ngpu=1):
        super(DMR, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options
        self.ngpu = ngpu
        self.to_lsp = list(range(14))

        self.dp_res = int(self.options.img_res // (2 ** self.options.warp_level))
        self.CNet = DPNet(warp_lv=self.options.warp_level,
                            norm_type=self.options.norm_type).to(self.device)

        self.LNet = get_LNet(self.options).to(self.device)

        uv_res = self.options.uv_res
        self.uv_type = self.options.uv_type
        self.sampler = Index_UV_Generator(UV_height=uv_res, UV_width=-1, uv_type=self.uv_type).to(self.device)

        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint)
            try:
                self.CNet.load_state_dict(checkpoint['CNet'])
                self.LNet.load_state_dict(checkpoint['LNet'])
                print('Checkpoint loaded')
            except KeyError:
                print('loading failed')

    def forward(self, images, IUV=None, train_mix_cnn=False, detach=True):
        out_dict = {}

        if detach:
            with torch.no_grad():
                if self.ngpu > 1 and images.shape[0] % self.ngpu == 0:
                    pred_dp, dp_feature, codes = data_parallel(self.CNet, images, range(self.ngpu))
                    pred_uv_map, pred_camera = data_parallel(self.LNet, (pred_dp, dp_feature, codes), range(self.ngpu))
                    pred_vertices = self.sampler.resample(pred_uv_map)
                else:
                    pred_dp, dp_feature, codes = self.CNet(images)
                    pred_uv_map, pred_camera= self.LNet(pred_dp, dp_feature, codes)
                    pred_vertices = self.sampler.resample(pred_uv_map)
        else:
            if self.ngpu > 1 and images.shape[0] % self.ngpu == 0:
                pred_dp, dp_feature, codes = data_parallel(self.CNet, images, range(self.ngpu))
                pred_uv_map, pred_camera = data_parallel(self.LNet, (pred_dp, dp_feature, codes), range(self.ngpu))
                pred_vertices = self.sampler.resample(pred_uv_map)
            else:
                pred_dp, dp_feature, codes = self.CNet(images)
                pred_uv_map, pred_camera = self.LNet(pred_dp, dp_feature, codes)
                pred_vertices = self.sampler.resample(pred_uv_map)

        out_dict['pred_vertices'] = pred_vertices
        out_dict['camera'] = pred_camera
        out_dict['uv_map'] = pred_uv_map
        out_dict['dp_map'] = pred_dp
        return out_dict


