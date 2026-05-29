import logging
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm

import test
from utils.general import fitness, labels_to_image_weights
from utils.plots import plot_images, plot_results
from utils.torch_utils import is_parallel
from train_utils.earlyStopping import EarlyStopper

logger = logging.getLogger(__name__)


def start_training_loop(
    # Training state
    start_epoch, epochs, best_fitness,
    # Model components
    model, ema, optimizer, scheduler, compute_loss, scaler,
    # Data components
    dataloader, testloader, dataset, data_dict,
    # Configuration
    opt, hyp, device, rank, cuda,
    # Training parameters
    batch_size, total_batch_size, accumulate, nbs, nw, nb, nc,
    # Image sizes and grid
    imgsz, imgsz_test, gs, down_factor,
    # Paths and logging
    save_dir, last, best, results_file,
    plots, wandb_logger, tb_writer,
    # Validation state
    maps, results, is_coco,
    # Learning rate function
    lf,
    # Early stopping parameters
    early_stopping, early_stopping_patience,
    early_stopping_min_delta, det_labels
):
    """Main training loop for SuperYOLO

    Args:
        start_epoch: Starting epoch number
        epochs: Total number of epochs to train
        best_fitness: Best fitness score so far
        model: The model to train
        ema: Exponential moving average model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        compute_loss: Loss computation function
        scaler: Gradient scaler for mixed precision
        dataloader: Training dataloader
        testloader: Validation dataloader
        dataset: Training dataset
        data_dict: Dataset configuration dictionary
        opt: Training options
        hyp: Hyperparameters
        device: Training device
        rank: DDP rank (-1 for single GPU)
        cuda: Whether CUDA is available
        batch_size: Batch size per GPU
        total_batch_size: Total batch size across all GPUs
        accumulate: Gradient accumulation steps
        nbs: Nominal batch size for scaling
        nw: Number of warmup iterations
        nb: Number of batches per epoch
        nc: Number of classes
        imgsz: Training image size
        imgsz_test: Test image size
        gs: Grid size (max stride)
        down_factor: Downsampling factor for SR mode
        save_dir: Directory to save results
        last: Path to last checkpoint
        best: Path to best checkpoint
        results_file: Path to results file
        plots: Whether to plot
        wandb_logger: Weights & Biases logger
        tb_writer: TensorBoard writer
        maps: mAP per class array
        results: Validation results tuple
        is_coco: Whether dataset is COCO
        lf: Learning rate lambda function
        early_stopping: Early stopping boolan flag
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        early_stopping_min_delta: Minimum change in the monitored quantity to qualify as an improvement for early stopping
        det_labels: Display Detection labels
    Returns:
        Tuple of (best_fitness, epoch): Updated best fitness value and final epoch number
    """
    earlyStopping = EarlyStopper(early_stopping_patience, early_stopping_min_delta)
    e_stop = None  # Early stopping flag

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, irs, targets, paths, _) in pbar:  # batch zjq  -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            image = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            ir_image = irs.to(device, non_blocking=True).float() / 255.0 #zjq

            # if imgsz==imgsz_test*2:
            #     imgs=F.interpolate(image,size=[i//2 for i in image.size()[2:]], mode='bilinear', align_corners=True)
            #     irs=F.interpolate(ir_image,size=[i//2 for i in ir_image.size()[2:]], mode='bilinear', align_corners=True)
            #down_factor = int(imgsz/imgsz_test)
            if down_factor>1:
                imgs=F.interpolate(image,size=[i//down_factor for i in image.size()[2:]], mode='bilinear', align_corners=True)
                irs=F.interpolate(ir_image,size=[i//down_factor for i in ir_image.size()[2:]], mode='bilinear', align_corners=True)
            else:
                imgs = image
                irs = ir_image
            # imgs = get_edge(imgs).to(device, non_blocking=True)
            # irs = get_edge(irs).to(device, non_blocking=True)
            # Warmup
            # t0 = time.time()
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
            # t1 = time.time()
            # print(t1-t0)
            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    irs = F.interpolate(irs, size=ns, mode='bilinear', align_corners=False) #zjq

            # Forward
            with amp.autocast(enabled=cuda, device_type='cuda'):
                # t0 = time.time()
                if opt.super:# and not opt.attention and not opt.super_attention:
                    pred,output_sr,_ = model(imgs,irs,opt.input_mode)  # forward #zjq
                # elif (opt.super and opt.attention) or opt.super_attention:
                #     pred,output_sr, attention_mask,_ = model(imgs,irs,opt.input_mode)
                # elif not opt.super and not opt.super_attention and opt.attention:
                #     pred, attention_mask,_ = model(imgs,irs,opt.input_mode)
                else:
                    pred,_ = model(imgs,irs,opt.input_mode)
                # t1 = time.time()
                # print(t1-t0)

                loss, lbox , lobj , lcls  = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                loss_items = torch.cat((lbox, lobj, lcls, loss)).detach()
                if opt.super: #and not opt.attention and not opt.super_attention:
                    if opt.input_mode =='IR':
                        sr_loss = 0.1*torch.nn.L1Loss()(output_sr,ir_image)
                    elif opt.input_mode =='RGB':
                        sr_loss = 0.1*torch.nn.L1Loss()(output_sr,image)
                    else:
                        sr_loss = 0.1*(torch.nn.L1Loss()(output_sr[:,0:3,:,:,],image)+torch.nn.L1Loss()(output_sr[:,3:,:,:,],ir_image[:,0:1,:,:,]))
                    loss += sr_loss
                # if (opt.super or opt.super_attention) and opt.attention:
                #     if opt.input_mode =='IR':
                #         sr_loss = 0.01*torch.nn.MSELoss()(output_sr,ir_image)
                #     elif opt.input_mode =='RGB':
                #         sr_loss = 0.01*torch.nn.MSELoss()(output_sr,image)
                #     else:
                #         sr_loss = 0.01*(torch.nn.MSELoss()(output_sr[:,0:3,:,:,],image)+torch.nn.MSELoss()(output_sr[:,3:,:,:,],ir_image[:,0:1,:,:,]))
                #     loss += sr_loss
                # if opt.cal_att_loss:
                #     att_loss = attention_loss(imgs.shape,attention_mask, targets)
                #     loss += 0.01*att_loss
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
            # break #zjq
            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, targets, paths, f, ), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            #break
            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1

                results, maps, times, e_stop = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 input_mode = opt.input_mode,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 save_json=(is_coco or opt.save_json) and (e_stop or final_epoch or early_stopping),
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and (final_epoch or early_stopping),
                                                 plots=plots and (final_epoch or early_stopping),  # Plot on final epoch or when early stopping is enabled
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 early_stopping=early_stopping,
                                                 earlyStopping=earlyStopping,
                                                 final_epoch=final_epoch,
                                                 det_labels=det_labels,
                                                 isTrain=True
                                                )

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None,}
                        #'ch':opt.ch,
                        #'input_mode':opt.input_mode,
                        #'cfg':opt.cfg,
                        #'ch_steam':opt.ch_steam,
                        #'anchors':hyp.get('anchors')}


                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                    print('Saving a best checkpoint ...')
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        if e_stop:
            print('*' * 30)
            logger.info("Early stopping triggered. Stopping training...")
            break

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    return best_fitness, epoch
