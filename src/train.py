# -*- coding: utf-8 -*
# !/usr/local/bin/python
# pip install kornia==0.1.4.post2
import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import *
import numpy as np
from dsr_utils import *
from dsr_dataloader import *
from PDSR import *


parser = argparse.ArgumentParser(description="DepthSR")
# data loader
parser.add_argument("--depth_path", type=str,
                    default='./RawDatasets/NYUDepthv2_Train/Depth',
                    help="DepthSet path")

parser.add_argument("--rgb_path", type=str,
                    default='./RawDatasets/NYUDepthv2_Train/RGB',
                    help="RGBSet path")
#
parser.add_argument('--dataset_name', type=str, default='NYU', help='valid dataset name')
parser.add_argument("--depth_path_valid", type=str,
                    default='./RawDatasets/NYUDepthv2_Valid/Depth',
                    help="DepthSet path")
parser.add_argument("--rgb_path_valid", type=str,
                    default='./RawDatasets/NYUDepthv2_Valid/RGB',
                    help="RGBSet path")

parser.add_argument('--augmentation', type=bool, default=True, help='augmentation')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--scale', type=int, default=4, help='patch_size')

# model&events path
parser.add_argument('--load_path', default='', help='pth')
parser.add_argument('--log_path', default='./log', help='log path')
parser.add_argument('--model_path', default='./save_path', help='model')
parser.add_argument('--model_name', default='PDSR', help='model')

# train setting
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--lr", type=float, default=1 * 1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--milestones", type=list, default=[2000, 3200, 4100], help="how many epoch to reduce the lr")
parser.add_argument("--gamma", type=int, default=0.5, help="how much to reduce the lr each time")

parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

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
    val_set_loader = DataLoader(dataset=val_set, num_workers=arg.threads, batch_size=1, shuffle=False)
    RMSE_epoch = 0
    model.eval()
    for iteration, (depth_lr, rgb, depth_hr, depth_min, depth_max) in enumerate(val_set_loader):

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
        if arg.dataset_name == "NYU" or arg.dataset_name == "RGBDD":
            if iteration % 50 == 0:
                print("VAL===> Val.RMSE: {:.4f}".format(RMSE))
        else:
            print("VAL===> Val.RMSE: {:.4f}".format(RMSE))

    RMSE_valid = RMSE_epoch / (iteration + 1)
    print("VAL===> Val_Avg.RMSE: {:.4f}".format(RMSE_valid))
    return  RMSE_valid



def main(arg):
    torch.manual_seed(arg.seed)

    cuda = arg.cuda
    if cuda:
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
        torch.cuda.manual_seed(arg.seed)
    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = Train_NYU(arg)
    training_data_loader = DataLoader(dataset=train_set, num_workers=arg.threads, batch_size=arg.batchSize,
                                      shuffle=True)

    print("===> Building model")
    arg.cuda = torch.cuda.is_available()
    model = PDSR(arg)

    # criterion = Loss_train()
    criterion = nn.MSELoss()
    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()

    load_model_path = arg.load_path
    if os.path.isfile(load_model_path):
        print("=> loading checkpoint '{}'".format(load_model_path))
        model.load_state_dict(torch.load(load_model_path))

    
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-6, betas=(0.9, 0.999))
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, arg.milestones, arg.gamma)

    print("===> Training")
    event_dir = os.path.join(arg.log_path, arg.model_name, 'event')
    print("===> event dir", event_dir)
    event_writer = SummaryWriter(event_dir)

    model_out_path = os.path.join(arg.model_path, arg.model_name)
    print("===> model_path", model_out_path)
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    print()

    Best = 100000
    Best_val = 100000
    total_iter = 0
    for epoch in range(arg.start_epoch, arg.nEpochs + 1):
        lr_scheduler.step()
        loss_epoch = 0
        RMSE_epoch = 0
        model.train()
        print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
        for iteration, (depth_lr, rgb, depth_hr) in enumerate(training_data_loader):
            total_iter = total_iter + 1

            if arg.cuda:
                rgb = rgb.cuda()
                depth_lr = depth_lr.cuda()
                depth_hr = depth_hr.cuda()
             
            _, _, H, W = depth_hr.shape
            depth_lr = torch.nn.functional.interpolate(depth_lr, size=(H, W), mode='bicubic')
            pred, depth_FromFusion1, depth_FromFusion2, depth_FromFusion3,\
             rgb2depth1, depth_offset1, rgb2depth2, depth_offset2, rgb2depth3, depth_offset3 = model(depth_lr, rgb)


            if arg.dataset_name == "NYU" or arg.dataset_name == "RGBDD":
                pred = pred * 100
                depth_hr = depth_hr * 100
            loss_final = criterion(pred, depth_hr)
            depth_hr_down4 = torch.nn.functional.interpolate(depth_hr, scale_factor=0.25, mode='bicubic')

            loss_middle = 0.0005 * (criterion(depth_FromFusion3 * 100, depth_hr)\
                          + criterion(depth_FromFusion2 * 100, depth_hr)\
                          + criterion(depth_FromFusion1 * 100, depth_hr))
            loss_plane = 0.00001 * (criterion(rgb2depth1 * 100, depth_hr_down4) + criterion(depth_offset1 * 100, depth_hr_down4) \
                     + criterion(rgb2depth2 * 100, depth_hr_down4) + criterion(depth_offset2 * 100, depth_hr_down4) \
                        + criterion(rgb2depth3 * 100, depth_hr_down4) + criterion(depth_offset3 * 100, depth_hr_down4))
            loss = loss_final + loss_middle + loss_plane

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            with torch.no_grad():
                RMSE = computeRMSE(pred, depth_hr)
                RMSE_epoch += RMSE

            if iteration % 50 == 0:
                print("===> Epoch[{}] Iteration[{}]: loss_final: {:.5f}, loss_middle: {:.5f}, loss_plane: {:.5f}, RMSE: {:.4f}, lr {}"
                      .format(epoch, iteration, loss_final.item(), loss_middle.item(), loss_plane.item(), RMSE, optimizer.param_groups[0]["lr"]))


            if iteration % 50 == 0:
                event_writer.add_scalar('Loss', loss.item(), total_iter)
                event_writer.add_scalar('RMSE', RMSE, total_iter)

        is_best = RMSE_epoch < Best
        Best = min(RMSE_epoch, Best)
        if is_best or epoch % 100 == 0:
            model_save = os.path.join(model_out_path, "model_epoch_{}_{}.pth".format(epoch, Best / (iteration + 1)))
            torch.save(model.state_dict(), model_save)
            print("Checkpoint saved to {}".format(model_save))

        print("===> Epoch[{}]|loss: {:.5f}|Best: {:.5f}".format(epoch,
                                                                loss_epoch / (iteration + 1),
                                                                Best / (iteration + 1)))
        if epoch % 20 == 0:

            with torch.no_grad():
                RMSE_val = valid(arg, model)
                is_best_val = RMSE_val < Best_val
                Best_val = min(RMSE_val, Best_val)
                if is_best_val:
                    model_save = os.path.join(model_out_path, "model_epoch_{}_val{}.pth".format(epoch, Best_val))
                    torch.save(model.state_dict(), model_save)
                    print("Checkpoint saved to {}".format(model_save))




if __name__ == "__main__":
    args = parser.parse_args()
    args.embed_ch = 32

    args.n_rcablocks = 2
    args.front_RBs = 2
    args.back_RBs = 2

    print(args)
    main(args)
