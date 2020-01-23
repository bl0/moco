"""
evaluating MoCo

MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""

import argparse
import os
import time
from pprint import pprint

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms, datasets

from lib.models.resnet import resnet50
from lib.models.LinearModel import LinearClassifierResNet
from lib.util import adjust_learning_rate, AverageMeter, check_dir, accuracy


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb-freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save-freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch-size', type=int, default=256, help='train batch size')
    parser.add_argument('--val-batch-size', type=int, default=256, help='valiate batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning-rate', type=float, default=30, help='learning rate')
    parser.add_argument('--lr-decay-epochs', type=int, default=[30, 60, 90], nargs='+',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    # model definition
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

    # crop
    parser.add_argument('--crop', type=float, default=0.08, help='minimum crop')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'])

    # specify folder
    parser.add_argument('--model-path', type=str, help="model path")
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')
    parser.add_argument('--data-root', type=str, default='./data', help='root directory of dataset')

    # experiment name
    parser.add_argument('--exp-name', type=str, default='exp', help='experiment name')

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # others
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])
    parser.add_argument('--bn', action='store_true', help='use parameter-free BN')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('-e', '--eval', action='store_true', help='only evaluate')

    # for DistributedDataParallel
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    output_dir = check_dir(os.path.join(args.output_root, args.dataset, args.exp_name))
    args.model_folder = check_dir(os.path.join(output_dir, 'linear_models'))
    args.tb_folder = check_dir(os.path.join(output_dir, 'linear_tensorboard'))

    if args.dataset == 'imagenet100':
        args.n_label = 100
    if args.dataset == 'imagenet':
        args.n_label = 1000

    return args


def get_loader(args):
    # set the data loader
    train_folder = os.path.join(args.data_root, args.dataset, 'train')
    val_folder = os.path.join(args.data_root, args.dataset, 'val')

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

    train_dataset = datasets.ImageFolder(train_folder, train_transform)
    val_dataset = datasets.ImageFolder(
        val_folder,
        transforms.Compose([
            transforms.Resize(image_size + crop_padding),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, shuffle=False, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader


def main(args):
    global best_acc1
    best_acc1 = 0

    train_loader, val_loader = get_loader(args)
    if args.local_rank == 0:
        print(f"length of training dataset: {len(train_loader.dataset)}")

    # create model and optimizer
    model = resnet50(width=args.model_width).cuda()
    for p in model.parameters():
        p.requires_grad = False
    classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', args.model_width).cuda()
    classifier = DistributedDataParallel(classifier, device_ids=[args.local_rank], broadcast_buffers=False)

    ckpt = torch.load(args.model_path, map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    if args.local_rank == 0:
        print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))

    criterion = torch.nn.CrossEntropyLoss().cuda()

    if not args.adam:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(classifier.parameters(),
                                     lr=args.learning_rate,
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay,
                                     eps=1e-8)

    model.eval()

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            if args.local_rank == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc1 = checkpoint['best_acc1']
            if args.local_rank == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            if 'opt' in checkpoint.keys():
                # resume optimization hyper-parameters
                if args.local_rank == 0:
                    print('=> resume hyper parameters')
                    if 'bn' in vars(checkpoint['opt']):
                        print('using bn: ', checkpoint['opt'].bn)
                    if 'adam' in vars(checkpoint['opt']):
                        print('using adam: ', checkpoint['opt'].adam)
                    if 'cosine' in vars(checkpoint['opt']):
                        print('using cosine: ', checkpoint['opt'].cosine)
                args.learning_rate = checkpoint['opt'].learning_rate
                args.lr_decay_rate = checkpoint['opt'].lr_decay_rate
                args.momentum = checkpoint['opt'].momentum
                args.weight_decay = checkpoint['opt'].weight_decay
                args.beta1 = checkpoint['opt'].beta1
                args.beta2 = checkpoint['opt'].beta2
            del checkpoint
            torch.cuda.empty_cache()
        else:
            if args.local_rank == 0:
                print("=> no checkpoint found at '{}'".format(args.resume))

    # set cosine annealing scheduler
    if args.cosine:
        assert not args.resume, "cosine lr scheduler not support resume now."
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3) * 0.1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)

    if args.eval:
        if args.local_rank == 0:
            print("==> testing...")
            validate(val_loader, model, classifier.module, criterion, args)
        return

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        torch.distributed.barrier()
        train_loader.sampler.set_epoch(epoch)

        if not args.cosine:
            adjust_learning_rate(epoch, args, optimizer)

        if args.local_rank == 0:
            print("==> training...")

        time1 = time.time()
        train_acc, train_acc5, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, args)

        if args.cosine:
            scheduler.step()

        if args.local_rank == 0:
            print('train epoch {}, total time {:.2f}'.format(epoch, time.time() - time1))
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_acc5', train_acc5, epoch)
            logger.log_value('train_loss', train_loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            print("==> testing...")
            test_acc, test_acc5, test_loss = validate(val_loader, model, classifier.module, criterion, args)

            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_acc5', test_acc5, epoch)
            logger.log_value('test_loss', test_loss, epoch)

            # save model
            if epoch % args.save_freq == 0:
                print('==> Saving...')
                state = {
                    'opt': args,
                    'epoch': epoch,
                    'classifier': classifier.state_dict(),
                    'best_acc1': test_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(args.model_folder, f'ckpt_epoch_{epoch}.pth'))


def train(epoch, train_loader, model, classifier, criterion, optimizer, opt):
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
        # measure data loading time
        data_time.update(time.time() - end)

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # ===================forward=====================
        with torch.no_grad():
            feat = model(x, opt.layer)

        output = classifier(feat)
        loss = criterion(output, y)

        acc1, acc5 = accuracy(output, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))
        top5.update(acc5[0], x.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if opt.local_rank == 0:
            if idx % opt.print_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Lr {lr:.3f} \t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    return top1.avg, top5.avg, losses.avg


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
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print(f'Test: [{idx}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    opt = parse_option()
    if opt.local_rank == 0:
        pprint(vars(opt))

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True
    best_acc1 = 0

    main(opt)
