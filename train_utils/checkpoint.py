import logging

logger = logging.getLogger(__name__)


def resume_from_checkpoint(ckpt, pretrained, optimizer, ema, results_file, weights, epochs, opt):
    """Resume training from a checkpoint if weights is a .pt file
    
    Args:
        ckpt: Checkpoint dictionary
        pretrained: Whether loading from a pretrained checkpoint
        optimizer: Optimizer to load state into
        ema: EMA model to load state into
        results_file: Path to results file
        weights: Path to weights file
        epochs: Total epochs to train
        opt: Training options
        
    Returns:
        Tuple of (start_epoch, best_fitness, epochs): Updated training state
    """
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

    return start_epoch, best_fitness, epochs
