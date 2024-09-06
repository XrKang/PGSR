# -*- coding: utf-8 -*
# !/usr/local/bin/python
# kornia==0.1.4.post2
import argparse, os
import torch
from functools import partial
import pickle
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dsr_utils import *
from dsr_dataloader import *
from PDSR import *
import cv2

parser = argparse.ArgumentParser(description="DepthSR")
# data loader
# parser.add_argument("--depth_path_valid", type=str,
#                     default='./RawDatasets/NYUDepthv2_Test/Depth',
#                     help="HyperSet path")
# parser.add_argument("--rgb_path_valid", type=str,
#                     default='./RawDatasets/NYUDepthv2_Test/RGB',
#                     help="RGBSet path")

# parser.add_argument("--model_path", type=str,
#                     default=r"./PDSR_4x.pth",
#                     # default=r"./PDSR_8x.pth",
#                     # default=r"./PDSR_16x.pth",
#                     help="model path")

# parser.add_argument('--scale', type=int, default=4, help='scale')
# # parser.add_argument('--scale', type=int, default=8, help='scale')
# # parser.add_argument('--scale', type=int, default=16, help='scale')


# parser.add_argument('--save_dir', type=str,
#                     default='./PDSR_4x')
#                     # default='./PDSR_8x')
#                     # default='./PDSR_16x')

parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()



def load_parallel(model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def valid(arg, model):
    torch.cuda.empty_cache()
    val_set = ValidLoader(arg)
    val_set_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)
    RMSE_epoch = 0
    ssim_epoch = 0

    model.eval()
    for iteration, (depth_lr, rgb, depth_hr, depth_min, depth_max, file_name) in enumerate(val_set_loader):

        if arg.cuda:
            rgb = rgb.cuda()
            depth_lr = depth_lr.cuda()
            depth_hr = depth_hr.cuda()
            depth_min = depth_min.cuda()
            depth_max = depth_max.cuda()

        _, _, H, W = depth_hr.shape
        depth_lr = torch.nn.functional.interpolate(depth_lr, size=(H, W), mode='bicubic')

        pred = model(depth_lr, rgb)
        pred = torch.clamp(pred, 0, 1)

        pred = pred * (depth_max - depth_min) + depth_min
        depth_hr = depth_hr * (depth_max - depth_min) + depth_min

        if arg.dataset_name == "NYU":
            pred = pred[..., 6: -6, 6: -6]
            depth_hr = depth_hr[..., 6: -6, 6: -6]

        if arg.dataset_name == "NYU" or arg.dataset_name == "RGBDD":
            pred = pred * 100
            depth_hr = depth_hr * 100

        RMSE = computeRMSE(pred, depth_hr)
        RMSE_epoch = RMSE_epoch + RMSE
        ssim = computeSSIM(pred, depth_hr, data_range=float((depth_hr.max() - depth_lr.min()).item()))
        ssim_epoch = ssim_epoch + ssim

        if opt.save_dir:
            if not os.path.exists(opt.save_dir):
                os.makedirs(opt.save_dir)
            save_name = os.path.join(opt.save_dir, arg.dataset_name + '_' + str(file_name[0]) + '.npy')
            pred_np = pred[0][0].cpu().detach().numpy()
            np.save(save_name, pred_np)

            visual_path = opt.save_dir.replace("test_result", "test_result_visualization")
            save_name = os.path.join(visual_path, arg.dataset_name + '_' + str(file_name[0]) + '.png')
            if not os.path.exists(visual_path):
                os.makedirs(visual_path)
            pred_visual = pred[0][0].cpu().detach().numpy()
            pred_visual = (pred_visual - pred_visual.min()) / (pred_visual.max() - pred_visual.min()) * 255.0
            pred_visual = pred_visual.astype("uint8")
            pred_visual = cv2.applyColorMap(pred_visual, cv2.COLORMAP_INFERNO)
            cv2.imwrite(save_name, pred_visual)
            
           



    RMSE_valid = RMSE_epoch / (iteration + 1)
    ssim_valid = ssim_epoch / (iteration + 1)
    print("Dataset:{}-VAL_x{} ===> Val_Avg.RMSE: {:.4f} SSIM: {:.4f}".format(opt.dataset_name, opt.scale, RMSE_valid,
                                                                             ssim_valid))


if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    torch.cuda.empty_cache()
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opt.save = True
    opt.embed_ch = 32
    opt.n_rcablocks = 2
    opt.front_RBs = 2
    opt.back_RBs = 2



    file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(file_path)
    scale_factors = [4, 8, 16]
    for scale in scale_factors:
        opt.scale = scale
        print("------------Scale:{}----------".format(scale))


        if scale==4:
            opt.model_path =  current_directory + "\PDSR_4x.pth"
            opt.save_dir = current_directory + "\PDSR_4x"
        elif scale==8:
            opt.model_path = current_directory + "\PDSR_8x.pth"
            opt.save_dir = current_directory + "\PDSR_8x"
        elif  scale==16:
            opt.model_path = current_directory + "\PDSR_16x.pth"
            opt.save_dir = current_directory + "\PDSR_16x"

        model = PDSR(opt)
        model_path = opt.model_path

        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            model.load_state_dict(load_parallel(model_path))

        model.eval()
        if opt.cuda:
            if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
            model.cuda()
        dataset_names = ['NYU', 'Middlebury', 'Lu', 'RGBDD']
        for dataset_name in dataset_names:
            print("------------Dataset:{}----------".format(dataset_name))

            opt.dataset_name = dataset_name
            if opt.dataset_name=='NYU':
                opt.depth_path_valid='./RawDatasets/NYUDepthv2_Test/Depth'
                opt.rgb_path_valid='./RawDatasets/NYUDepthv2_Test/RGB'
            elif opt.dataset_name=='Middlebury':
                opt.depth_path_valid='./RawDatasets/Middlebury/Depth'
                opt.rgb_path_valid='./RawDatasets/Middlebury/RGB'
            elif opt.dataset_name == 'Lu':
                opt.depth_path_valid='./RawDatasets/Lu/Depth'
                opt.rgb_path_valid='./RawDatasets/Lu/RGB'
            elif opt.dataset_name == 'RGBDD':
                opt.depth_path_valid='./RawDatasets/RGBDD/Depth'
                opt.rgb_path_valid='./RawDatasets/RGBDD/RGB'

            with torch.no_grad():
                valid(opt, model)
