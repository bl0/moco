"""
Code for MoCo pre-training

MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

"""
import argparse
import os
import time
from pprint import pprint

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

from lib.NCE import MemoryMoCo, NCESoftmaxLoss
from lib.dataset import ImageFolderInstance
from lib.models.resnet import resnet50
from lib.util import adjust_learning_rate, AverageMeter, check_dir, DistributedShufle, set_bn_train, moment_update

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    # exp name
    parser.add_argument('--exp-name', type=str, default='exp',
                        help='experiment name, used to determine checkpoint/tensorboard dir')

    # optimization
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')

    # root folders
    parser.add_argument('--data-root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'])
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'])

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50'])
    parser.add_argument('--model-width', type=int, default=1, help='width of resnet, eg, 1, 2, 4')
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # loss function
    parser.add_argument('--nce-k', type=int, default=16384)
    parser.add_argument('--nce-t', type=float, default=0.07)

    # misc
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb-freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch-size', type=int, default=32, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    # set the path according to the environment
    output_dir = check_dir(os.path.join(args.output_root, args.dataset, args.exp_name))
    args.model_folder = check_dir(os.path.join(output_dir, 'models'))
    args.tb_folder = check_dir(os.path.join(output_dir, 'tensorboard'))

    return args


def get_loader(args):
    # set the data loader
    train_folder = os.path.join(args.data_root, args.dataset, 'train')

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


def load_checkpoint(args, model, model_ema, contrast, optimizer):
    if args.local_rank == 0:
        print("=> loading checkpoint '{}'".format(args.resume))

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])
    contrast.load_state_dict(checkpoint['contrast'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if args.local_rank == 0:
        print("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()

    # make sure all process have loaded the checkpoint
    torch.distributed.barrier()


def save_checkpoint(args, epoch, model, model_ema, contrast, optimizer):
    print('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(args.model_folder, 'current.pth'))
    if epoch % args.save_freq == 0:
        torch.save(state, os.path.join(args.model_folder, f'ckpt_epoch_{epoch}.pth'))
    # help release GPU memory
    del state
    torch.cuda.empty_cache()


def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    if args.local_rank == 0:
        print(f"length of training dataset: {n_data}")

    model, model_ema = build_model(args)
    contrast = MemoryMoCo(128, args.nce_k, args.nce_t).cuda()
    criterion = NCESoftmaxLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, model_ema, contrast, optimizer)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(epoch, args, optimizer)

        tic = time.time()
        loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args)

        if args.local_rank == 0:
            print('epoch {}, total time {:.2f}'.format(epoch, time.time() - tic))

            # tensorboard logger
            logger.log_value('ins_loss', loss, epoch)
            logger.log_value('ins_prob', prob, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # save model
            save_checkpoint(args, epoch, model, model_ema, contrast, optimizer)


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args):
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
        loss.backward()
        optimizer.step()

        moment_update(model, model_ema, args.alpha)

        # update meters
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if args.local_rank == 0 and idx % args.print_freq == 0:
            print(f'Train: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                  f'prob {prob_meter.val:.3f} ({prob_meter.avg:.3f})')

    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    opt = parse_option()
    if opt.local_rank == 0:
        pprint(vars(opt))

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    main(opt)
