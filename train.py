"""
Code for MoCo pre-training

MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""
import argparse
import os
import time
import json

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms


from moco.NCE import MemoryMoCo, NCESoftmaxLoss
from moco.dataset import ImageFolderInstance
from moco.logger import setup_logger
from moco.models.resnet import resnet50
from moco.util import AverageMeter, MyHelpFormatter, DistributedShufle, set_bn_train, moment_update
from moco.lr_scheduler import get_scheduler

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('moco training', formatter_class=MyHelpFormatter)

    # dataset
    parser.add_argument('--data-dir', type=str, required=True, help='root director of dataset')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'],
                        help='dataset to training')
    parser.add_argument('--crop', type=float, default=0.08, help='minimum crop')
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'],
                        help="augmentation type: NULL for normal supervised aug, CJ for aug with ColorJitter")
    parser.add_argument('--batch-size', type=int, default=128, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')

    # model and loss function
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50'], help="backbone model")
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--nce-k', type=int, default=65536, help='num negative sampler')
    parser.add_argument('--nce-t', type=float, default=0.07, help='NCE temperature')

    # optimization
    parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.1,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    args = parser.parse_args()

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)

    return args


def get_loader(args):
    # set the data loader
    train_folder = os.path.join(args.data_dir, 'train')

    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.aug == 'NULL':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.aug == 'CJ':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(args.crop, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('augmentation not supported: {}'.format(args.aug))

    train_dataset = ImageFolderInstance(train_folder, transform=train_transform, two_crop=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)

    return train_loader


def build_model(args):
    model = resnet50(width=args.model_width).cuda()
    model_ema = resnet50(width=args.model_width).cuda()

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    return model, model_ema


def load_checkpoint(args, model, model_ema, contrast, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.resume))

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    contrast.load_state_dict(checkpoint['contrast'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, model_ema, contrast, optimizer, scheduler):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth'))


def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")

    model, model_ema = build_model(args)
    contrast = MemoryMoCo(128, args.nce_k, args.nce_t).cuda()
    criterion = NCESoftmaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.amp_opt_level != "O0":
        if amp is None:
            logger.warning(f"apex is not installed but amp_opt_level is set to {args.amp_opt_level}, ignoring.\n"
                           "you should install apex from https://github.com/NVIDIA/apex#quick-start first")
            args.amp_opt_level = "O0"
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)
            model_ema = amp.initialize(model_ema, opt_level=args.amp_opt_level)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, model_ema, contrast, optimizer, scheduler)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args)

        logger.info('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', loss, epoch)
            summary_writer.add_scalar('ins_prob', prob, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if dist.get_rank() == 0:
            # save model
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer, scheduler)


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, scheduler, args):
    """
    one epoch training for moco
    """
    model.train()
    set_bn_train(model_ema)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _,) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        # forward
        x1, x2 = torch.split(inputs, [3, 3], dim=1)
        x1.contiguous()
        x2.contiguous()
        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)

        feat_q = model(x1)
        with torch.no_grad():
            x2_shuffled, backward_inds = DistributedShufle.forward_shuffle(x2, epoch)
            feat_k = model_ema(x2_shuffled)
            feat_k_all, feat_k = DistributedShufle.backward_shuffle(feat_k, backward_inds, return_local=True)

        out = contrast(feat_q, feat_k, feat_k_all)
        loss = criterion(out)
        prob = F.softmax(out, dim=1)[:, 0].mean()

        # backward
        optimizer.zero_grad()
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        moment_update(model, model_ema, args.alpha)

        # update meters
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            logger.info(f'Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                        f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                        f'prob {prob_meter.val:.3f} ({prob_meter.avg:.3f})')

    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    opt = parse_option()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="moco")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)
