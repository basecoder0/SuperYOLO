#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# import test_up  # import test.py to get mAP after each epoch
import test
from models.experimental import attempt_load
from models.SRyolo import Model #zjq
from train_utils.setup import save_run_settings, setup_directories, setup_logging_and_wandb, setup_seeds_and_config
from train_utils.setup_model import build_model, freeze_layers, setup_optimizer_params
from train_utils.training_loop import start_training_loop
from train_utils.checkpoint import resume_from_checkpoint
from train_utils.finalize_training import finalize_training
from utils.autoanchor import check_anchors


from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution,plot_lr_scheduler
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from train_utils import resume_interrupted_run, ddp_mode, load_hyperparameters, evolve_hyperparameters
# from optimizer import build_optimizer
########swin#######
#from config import get_config

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    last, best, results_file = setup_directories(save_dir)
    save_run_settings(save_dir, opt, hyp)
    plots, cuda, data_dict, is_coco = setup_seeds_and_config(opt, device, init_seeds, rank)
    loggers, wandb_logger, data_dict, weights, epochs, hyp = setup_logging_and_wandb(opt, hyp, weights, rank, data_dict, save_dir)


    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Setup Model
    model, train_path, test_path, ckpt, pretrained, down_factor = build_model(opt, hyp, weights, nc, data_dict, device, rank, show_model=opt.show_model)

    # Freeze layers
    freeze_layers(model, opt.freeze)

    # Setup Optimizer
    pg0, accumulate, hyp, optimizer, nbs = setup_optimizer_params(model, hyp, opt, total_batch_size)


    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    # del pg0, pg1, pg2
    # optimizer = build_optimizer(config, model)


    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness, epochs = resume_from_checkpoint(ckpt, pretrained, optimizer, ema, results_file, weights, epochs, opt)

    # start_epoch = 98 #zjq
    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    # if not opt.super and not opt.super_attention:
    # if not opt.data.endswith('SRvedai.yaml'):
    if opt.data.endswith('vedai.yaml') or opt.data.endswith('SRvedai.yaml'):
        from utils.datasets import create_dataloader_sr as create_dataloader
    else:
        from utils.datasets import create_dataloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                        hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                        #world_size=opt.world_size,
                                        workers=opt.workers,
                                        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    # else:
        # dataloader, dataset = create_dataloader_sr(train_path, imgsz, batch_size, gs, opt,
        #                                 hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
        #                                 world_size=opt.world_size, workers=opt.workers,
        #                                 image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        # if not opt.data.endswith('SRvedai.yaml'):
        testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,  # testloader
                                    hyp=hyp, cache=opt.cache_images and not opt.notest, rect=False, rank=-1,
                                    #world_size=opt.world_size,
                                    workers=opt.workers,pad=0.5,
                                    prefix=colorstr('val: '))[0]
        # else:
        #     testloader = create_dataloader_sr(test_path, imgsz_test, batch_size, gs, opt,  # testloader
        #                                hyp=hyp, cache=opt.cache_images and not opt.notest, rect=False, rank=-1,
        #                                world_size=opt.world_size, workers=opt.workers,pad=0.5,
        #                                prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    # attention_loss = LevelAttention_loss()
    # superloss = Superresolution_loss()
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')


    # def Log_UP(K_min, K_max, epoch):
    #     Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    #     return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / epochs * epoch)]).float().cuda()

    # print (model.module)

    best_fitness, epoch = start_training_loop(start_epoch, epochs, best_fitness,
                        model, ema, optimizer, scheduler, compute_loss, scaler,
                        dataloader, testloader, dataset, data_dict,
                        opt, hyp, device, rank, cuda,
                        batch_size, total_batch_size, accumulate, nbs, nw, nb, nc,
                        imgsz, imgsz_test, gs, down_factor,
                        save_dir, last, best, results_file,
                        plots, wandb_logger, tb_writer,
                        maps, results, is_coco,
                        lf, opt.early_stp, opt.early_stp_pat, opt.min_delta,
                        opt.det_labels)

    results = finalize_training(rank, plots, save_dir, wandb_logger, opt, epoch,
                               start_epoch, t0, nc, last, best, batch_size,
                               imgsz_test, device, testloader, is_coco, results)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str,default='models/SRyolo_MF.yaml', help='model.yaml path') #yolov5s
    parser.add_argument('--data', type=str,default='data/SRvedai.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--ch_steam', type=int, default=3)
    parser.add_argument('--ch', type=int,default=64, help = '3 4 16 midfusion1:64 midfusion2,3:128 midfusion4:256')
    parser.add_argument('--input_mode', type=str,default='RGB+IR+MF',help ='RGB IR RGB+IR(pixel-level fusion) RGB+IR+fusion(feature-level fusion)')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--freeze', nargs='*', type=int, default=[], help='Freeze layers: backbone of yolov3 is 18, yolov5 is 10')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')

    # Togglable Flag Options
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader') #1/4的数据集
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--det_labels', action='store_true', help='show detection confidence score labels during training and testing')
    parser.add_argument('--show-model', action='store_true', help='print model architecture')

    # Early Stoping options
    parser.add_argument('--early_stp', action='store_true', help='Enable early stopping based on validation performance')
    parser.add_argument('--early_stp_pat', type=int, default=0, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum change in the monitored quantity to qualify as an improvement for early stopping')


    # TODO: Set Safe guards for Super-Resolution and Multi-Modal Training flags
    parser.add_argument('--super', action='store_true', help='super resolution')
    parser.add_argument('--train_img_size', type=int,default=1024, help='train image sizes,if use SR,please set 1024')
    parser.add_argument('--test_img_size', type=int, default=512, help='test image sizes')
    parser.add_argument('--hr_input', default=True,action='store_true', help='high resolution input(1024*1024)') #if use SR,please set True

    opt = parser.parse_args()

    # python3 train.py --cfg models/SRyolo_MF.yaml --super --train_img_size 1024 --hr_input --data data/SRvedai.yaml --ch 64 --input_mode RGB+IR+MF

    ######swin####
    #args, unparsed = parser.parse_known_args()
    #config = get_config(args)

    # Set DDP (Distributed Data Parallel) variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    set_logging(opt.global_rank)

    if opt.early_stp:
        try:
            if opt.early_stp_pat <= 0:
                raise ValueError("\nEarly stopping patience (--early_stp_pat) must be greater than 0")
            if opt.min_delta < 0:
                raise ValueError("\nEarly stopping minimum delta (--min_delta) must be non-negative")

            print('\n' + '*' * 30 + '\n')
            logger.info(f"Early stopping enabled with patience of {opt.early_stp_pat} epochs and minimum delta of {opt.min_delta}")
            print('\n' + '*' * 30 + '\n')
        except ValueError as e:
            logger.error(e)
            exit(1)

    if opt.global_rank in [-1, 0]:
        check_git_status()
        check_requirements()

    opt.img_size = [opt.train_img_size,opt.test_img_size]

    # Resume Interrupted Run
    wandb_run = check_wandb_resume(opt)
    resume_interrupted_run(opt, wandb_run)

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = ddp_mode(opt)

    # Hyperparameters
    hyp = load_hyperparameters(opt)

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        evolve_hyperparameters(opt, device, train)

