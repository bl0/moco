"""
evaluating MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
"""
import argparse
import json
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from moco.logger import setup_logger
from moco.lr_scheduler import get_scheduler
from moco.models.LinearModel import LinearClassifierResNet
from moco.models.resnet import resnet50
from moco.util import AverageMeter, MyHelpFormatter, accuracy, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('moco eval', formatter_class=MyHelpFormatter)

    # dataset
    parser.add_argument('--data-dir', type=str, required=True, help='root director of dataset')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet100'], help='dataset name')
    parser.add_argument('--crop', type=float, default=0.08, help='minimum crop')
    parser.add_argument('--aug', type=str, default='NULL', choices=['NULL', 'CJ'],
                        help='augmentation type: NULL for normal supervised aug, CJ for aug with ColorJitter')
    parser.add_argument('--total-batch-size', type=int, default=256, help='total train batch size for all GPU')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50'], help="backbone model")
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

    # optimization
    parser.add_argument('--learning-rate', type=float, default=30, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[30, 60, 90], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--pretrained-model', type=str, required=True, help="pretrained model path")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--output-dir', type=str, default='./output', help='root director for output')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=5, help='save frequency')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')
    parser.add_argument('-e', '--eval', action='store_true', help='only evaluate')

    args = parser.parse_args()

    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)

    return args


def get_loader(args):
    image_size = 224
    crop_padding = 32
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

    val_transform = transforms.Compose([
        transforms.Resize(image_size + crop_padding),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # set the data loader
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), val_transform)
    batch_size = args.total_batch_size // dist.get_world_size()
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              sampler=DistributedSampler(train_dataset),
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=args.num_workers,
                            sampler=DistributedSampler(val_dataset, shuffle=False),
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

    return train_loader, val_loader


def build_model(args, num_class):
    # create model
    model = resnet50(width=args.model_width).cuda()
    for p in model.parameters():
        p.requires_grad = False
    classifier = LinearClassifierResNet(args.layer, num_class, 'avg', args.model_width).cuda()
    return model, classifier


def load_pretrained(args, model):
    ckpt = torch.load(args.pretrained_model, map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    logger.info(f"==> loaded checkpoint '{args.pretrained_model}' (epoch {ckpt['epoch']})")


def load_checkpoint(args, classifier, optimizer, scheduler):
    logger.info("=> loading checkpoint '{args.resume'")

    checkpoint = torch.load(args.resume, map_location='cpu')

    global best_acc1
    best_acc1 = checkpoint['best_acc1']
    args.start_epoch = checkpoint['epoch'] + 1
    classifier.load_state_dict(checkpoint['classifier'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")


def save_checkpoint(args, epoch, classifier, test_acc, optimizer, scheduler):
    state = {
        'opt': args,
        'epoch': epoch,
        'classifier': classifier.state_dict(),
        'best_acc1': test_acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    torch.save(state, os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth'))
    torch.save(state, os.path.join(args.output_dir, f'current.pth'))


def main(args):
    global best_acc1

    train_loader, val_loader = get_loader(args)
    logger.info(f"length of training dataset: {len(train_loader.dataset)}")

    model, classifier = build_model(args, num_class=len(train_loader.dataset.classes))
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.amp_opt_level != "O0":
        if amp is None:
            logger.warning(f"apex is not installed but amp_opt_level is set to {args.amp_opt_level}, ignoring.\n"
                           "you should install apex from https://github.com/NVIDIA/apex#quick-start first")
            args.amp_opt_level = "O0"
        else:
            model = amp.initialize(model, opt_level=args.amp_opt_level)
            classifier, optimizer = amp.initialize(classifier, optimizer, opt_level=args.amp_opt_level)

    classifier = DistributedDataParallel(classifier, device_ids=[args.local_rank], broadcast_buffers=False)

    model.eval()

    load_pretrained(args, model)
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), f"no checkpoint found at '{args.resume}'"
        load_checkpoint(args, classifier, optimizer, scheduler)

    if args.eval:
        logger.info("==> testing...")
        validate(val_loader, model, classifier, criterion, args)
        return

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        train(epoch, train_loader, model, classifier, criterion, optimizer, scheduler, args)
        logger.info(f'epoch {epoch}, total time {time.time() - tic:.2f}')

        logger.info("==> testing...")
        test_acc = validate(val_loader, model, classifier, criterion, args)

        if dist.get_rank() == 0 and epoch % args.save_freq == 0:
            logger.info('==> Saving...')
            save_checkpoint(args, epoch, classifier, test_acc, optimizer, scheduler)

        if summary_writer is not None:
            # tensorboard logger
            summary_writer.add_scalar('ins_loss', test_acc, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)


def train(epoch, train_loader, model, classifier, criterion, optimizer, scheduler, args):
    """
    one epoch training
    """

    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (x, y) in enumerate(train_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # ===================forward=====================
        with torch.no_grad():
            feat = model(x, args.layer)

        output = classifier(feat)
        loss = criterion(output, y)

        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            logger.info(
                f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Lr {optimizer.param_groups[0]["lr"]:.3f} \t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})\t'
                f'Acc@5 {top5.val:.3%} ({top5.avg:.3%})')


def validate(val_loader, model, classifier, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (x, y) in enumerate(val_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # compute output
            feat = model(x, args.layer)
            output = classifier(feat)
            loss = criterion(output, y)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, y, topk=(1, 5))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Acc@1 {top1.val:.3%} ({top1.avg:.3%})\t'
                    f'Acc@5 {top5.val:.3%} ({top5.avg:.3%})')

        logger.info(f' * Acc@1 {top1.avg:.3%} Acc@5 {top5.avg:.3%}')

    return top1.avg


if __name__ == '__main__':
    opt = parse_option()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True
    best_acc1 = 0

    os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="moco")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, "w") as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)
