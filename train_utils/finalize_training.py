import logging
import os
import time

import torch
import torch.distributed as dist

import test
from models.experimental import attempt_load
from utils.general import strip_optimizer
from utils.plots import plot_results

logger = logging.getLogger(__name__)


def finalize_training(rank, plots, save_dir, wandb_logger, opt, epoch, start_epoch, t0, 
                      nc, last, best, batch_size, imgsz_test, device, testloader, 
                      is_coco, results):
    """Finalize training with plots, testing, and cleanup
    
    Args:
        rank: DDP rank (-1 for single GPU)
        plots: Whether to generate plots
        save_dir: Directory to save results
        wandb_logger: Weights & Biases logger
        opt: Training options
        epoch: Final epoch number
        start_epoch: Starting epoch number
        t0: Training start time
        nc: Number of classes
        last: Path to last checkpoint
        best: Path to best checkpoint
        batch_size: Batch size per GPU
        imgsz_test: Test image size
        device: Training device
        testloader: Validation dataloader
        is_coco: Whether dataset is COCO
        results: Final validation results tuple
        
    Returns:
        Final results tuple
    """
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                                if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                            batch_size=batch_size * 2,
                                            imgsz=imgsz_test,
                                            conf_thres=0.001,
                                            iou_thres=0.7,
                                            model=attempt_load(m, device).half(),
                                            single_cls=opt.single_cls,
                                            dataloader=testloader,
                                            save_dir=save_dir,
                                            save_json=True,
                                            plots=False,
                                            is_coco=is_coco)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results