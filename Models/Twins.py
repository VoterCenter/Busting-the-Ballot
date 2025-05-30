# Non-torch libraries
import argparse
import numpy as np
import time
import os

# Torch libraries
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.init as init
from torch.autograd import Variable

# Twins specific libraries (in Twins_Environ)
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from Twins.losses import DistillationLoss
from Twins.samplers import RASampler
import Twins.gvt
import Twins.utils
import Twins.collections

# Import functions from other directories
import Utilities.DataManagerPytorch as DMP
import Utilities.VoterLab_Classifier_Functions as voterlab

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Get device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' 
GetTWINS() returns Twins model trained on specified dataset

Args:
    twins_type: 'BalCombined' for Combined RGB, 'GrayCombined' for Combined Grayscale, 'BalBubbles' for Bubbles RGB, 'GrayBubbles' for Bubbles Grayscale

Use TWINSResizeLoader(dataLoader, twins_type) to resize 40x50 0-1 datarange loader to 224 x 224 0-255 datarange loader

'''


# Load arguements for Twins
def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='alt_gvt_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.3, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=5, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='alt_gvt_base.pth', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='alt_gvt_base.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # test throught
    parser.add_argument('--throughout', action='store_true', help='Perform throughout only')
    return parser


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def ReturnTWINS(args, twins_type):
    utils.init_distributed_mode(args)
    #print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Load grayscale bubble dataloader
    #trainLoader, valLoader = ReturnVoterLabDataLoaders(imgSize = (1, 40, 50), loaderCreated = True, batchSize = 64, loaderType = 'Bubbles')

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=2)

    # For each loader type, we load a separate model
    loaderReference = {'BalBubbles': 'bubbleonly', 'BalCombined': 'combined', 'GrayBubbles': 'bubblesgrey', 'GrayCombined': 'combinedgrey'}
    name = loaderReference[twins_type]
    acc = []
    print("Currently on ..." + name)

    print(f"Creating model: {args.model}")
    print(args.drop)
        
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=2,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None, 
        in_chans = (1 if 'grey' in name else 3)
    )

    #summary(model.to(device), input_size = (1, 224, 226))
        
    model_ema = None

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = DistillationLoss(
        criterion, None, 'none', 0, 0
    )

    if not args.output_dir:
        args.output_dir = args.model
        if utils.is_main_process():
            #import os
            if not os.path.exists(args.model):
                os.mkdir(args.model)

    args.resume = os.getcwd() + "/alt_gvt_base//" + name + "//checkpoint_best.pth"
    
    model.reset_classifier(2, global_pool='')
    #args.resume = False
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        # Load denoiser weights then model weights
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'], strict = False)
        else:
            model_without_ddp.load_state_dict(checkpoint, strict = False)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    #model.reset_classifier(2, global_pool='')
    model = model.to(device).eval()

    return model


def TWINSResizeLoader(dataloader, twins_type):
    xData, yData = DMP.DataLoaderToTensor(dataloader)
    xData = 255.0 * xData
    resize = T.Resize((224, 224))
    return DMP.TensorToDataLoader(xData, yData, transforms = resize)


def GetTWINS(twins_type):
    parser = argparse.ArgumentParser('Twins training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Return TWINs
    model = ReturnTWINS(args, twins_type)
    return model


def GetTWINSCleanTrainAcc():
    # Load dataloaders
    import Utilities.VoterLab_Classifier_Functions as voterlab
    colorBalBubbleTrain, colorBubbleVal = voterlab.ReturnVoterLabDataLoaders(imgSize = (3, 40, 50), loaderCreated = True, batchSize = 64, loaderType = 'BalBubbles')
    colorBalFullTrain, colorFullVal = voterlab.ReturnVoterLabDataLoaders(imgSize = (3, 40, 50), loaderCreated = True, batchSize = 64, loaderType = 'BalCombined')
    greyBalBubbleTrain, greyBubbleVal = voterlab.ReturnVoterLabDataLoaders(imgSize = (1, 40, 50), loaderCreated = True, batchSize = 64, loaderType = 'BalBubbles')
    greyBalFullTrain, greyFullVal = voterlab.ReturnVoterLabDataLoaders(imgSize = (1, 40, 50), loaderCreated = True, batchSize = 64, loaderType = 'BalCombined')

    # Resize dataloaders 
    colorBalBubbleTrain = TWINSResizeLoader(dataloader = colorBalBubbleTrain, twins_type = 'Color')
    colorBalFullTrain = TWINSResizeLoader(dataloader = colorBalFullTrain, twins_type = 'Color') 
    greyBalBubbleTrain = TWINSResizeLoader(dataloader = greyBalBubbleTrain, twins_type = 'Grey')
    greyBalFullTrain = TWINSResizeLoader(dataloader = greyBalFullTrain, twins_type = 'Grey') 

    for twins_type in ['BalBubbles', 'BalCombined', 'GreyBubbles', 'GreyCombined']:
        model = GetTWINS(twins_type)
        if not 'Grey' in twins_type:
            fullAcc, _, _, _, _ = voterlab.validateReturn(model, colorBalFullTrain, device, returnLoaders = True, printAcc = False, returnWhereWrong = True)
            bubbleAcc, _, _, _, _ = voterlab.validateReturn(model, colorBalBubbleTrain, device, returnLoaders = True, printAcc = False, returnWhereWrong = True)
        else:
            fullAcc, _, _, _, _ = voterlab.validateReturn(model, greyBalFullTrain, device, returnLoaders = True, printAcc = False, returnWhereWrong = True)
            bubbleAcc, _, _, _, _ = voterlab.validateReturn(model, greyBalBubbleTrain, device, returnLoaders = True, printAcc = False, returnWhereWrong = True)
        print("Full Training Acc: " + str(fullAcc))
        print("Full Bubble Acc: " + str(bubbleAcc))


if __name__ == '__main__':
    GetTWINSCleanTrainAcc()
