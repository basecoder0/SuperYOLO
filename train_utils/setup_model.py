"""
Model setup utilities for SuperYOLO
"""
import logging
import torch
from torch import optim
from utils.google_utils import attempt_download
from utils.general import check_dataset
from utils.torch_utils import intersect_dicts, torch_distributed_zero_first
from models.SRyolo import Model
from optimizer import set_weight_decay

logger = logging.getLogger(__name__)


def build_model(opt, hyp, weights, nc, data_dict, device, rank):
    """Build and load model (pretrained or from scratch)
    
    Args:
        opt: Training options
        hyp: Hyperparameters
        weights: Path to weights file
        nc: Number of classes
        data_dict: Dataset configuration dictionary
        device: torch device
        rank: Process rank for distributed training
        
    Returns:
        model: Initialized model
        train_path: Path to training data
        test_path: Path to validation data
        ckpt: Checkpoint dict (or None if not pretrained)
    """
    pretrained = weights.endswith('.pt')
    down_factor = int(opt.train_img_size / opt.test_img_size)
    ckpt = None
    
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint
        model = Model(
            opt.cfg or ckpt['model'].yaml,
            input_mode=opt.input_mode,
            ch_steam=opt.ch_steam,
            ch=opt.ch,
            nc=nc,
            anchors=hyp.get('anchors'),
            config=None,
            sr=opt.super,
            factor=down_factor
        ).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(
            opt.cfg,
            input_mode=opt.input_mode,
            ch_steam=opt.ch_steam,
            ch=opt.ch,
            nc=nc,
            anchors=hyp.get('anchors'),
            config=None,
            sr=opt.super,
            factor=down_factor
        ).to(device)  # create
    
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    
    train_path = data_dict['train']
    test_path = data_dict['val']

    print(f'\n\nModel Architecture: {model}\n\n')
    
    return model, train_path, test_path, ckpt, pretrained, down_factor


def freeze_layers(model, freeze_list=None):
    """Freeze model layers
    
    Args:
        model: Model to freeze layers in
        freeze_list: List of parameter name patterns to freeze (empty list = freeze nothing)
    """
    freeze = freeze_list if freeze_list is not None else []
    
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers by default
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False


def setup_optimizer_params(model, hyp, opt, total_batch_size):
    """Setup optimizer parameter groups with weight decay
    
    Args:
        model: Model to optimize
        hyp: Hyperparameters
        opt: Training options
        total_batch_size: Total batch size across all GPUs
        
    Returns:
        pg0: Parameter groups for optimizer
        accumulate: Gradient accumulation steps
        hyp: Updated hyperparameters (weight_decay is scaled)
        optimizer: Configured optimizer
        nbs: Nominal batch size
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # Get parameters to skip from weight decay
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    
    pg0 = set_weight_decay(model, skip, skip_keywords)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    
    return pg0, accumulate, hyp, optimizer, nbs