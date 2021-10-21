"""Train Glow on various datasets.
Train script adapted from: https://github.com/chrischute/glow
which is then adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import Glow
from tqdm import tqdm
import torch.nn.functional as F


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # No normalization applied, since Glow expects inputs in (0, 1)
    if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST' or args.dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ])
        input_shape = (1, 32, 32)
        priors = [0.1 for _ in range(10)]
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    if args.dataset == 'CIFAR10':
        input_shape = (3, 32, 32)
        priors = [0.1 for _ in range(10)]
    elif args.dataset == 'CIFAR100':
        input_shape = (3, 32, 32)
        priors = [0.01 for _ in range(100)]

    if args.dataset == 'SVHN':
        trainset = getattr(torchvision.datasets, args.dataset)(root='data', split='train', download=True,
                                                               transform=transform)
        testset = getattr(torchvision.datasets, args.dataset)(root='data', split='test', download=True,
                                                              transform=transform)
        input_shape = (3, 32, 32)
    else:
        trainset = getattr(torchvision.datasets, args.dataset)(root='data', train=True, download=True, transform=transform)
        testset = getattr(torchvision.datasets, args.dataset)(root='data', train=False, download=True, transform=transform)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps, input_shape=input_shape, q=priors)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        global global_step
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainset)


    loss_fn = util.NLLLossCond().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, scheduler,
              loss_fn, args.max_grad_norm, args.output_dir, args.alpha)
        test(epoch, net, testloader, device, loss_fn, args.num_samples, args.output_dir, args.sample_dir, args.alpha)


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm, output_dir, alpha):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    nll_meter = util.AverageMeter()
    cel_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            z, sldj, y_logits = net(x, reverse=False)
            nll = loss_fn(z, sldj, net.module.means[y], net.module.cov_diag)
            cel = F.cross_entropy(y_logits, y)
            loss = nll + alpha * cel

            nll_meter.update(nll.item(), x.size(0))
            cel_meter.update(cel.item(), x.size(0))
            loss_meter.update(loss.item(), x.size(0))

            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=nll_meter.avg,
                                     cel=cel_meter.avg,
                                     loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)

    print('Saving the latest model...')

    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'nll': nll_meter.avg,
        'cel': cel_meter.avg,
        'loss': loss_meter.avg,
        'net_config': (net.module.input_shape, net.module.q)
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, output_dir + '/latest.pth.tar')


@torch.no_grad()
def sample(net, batch_size, device, y=None):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, *net.module.input_shape), dtype=torch.float32, device=device)

    if y is None:
        q = net.module.q
        idx = np.random.choice(list(range(len(q))), batch_size, p=q)
    else:
        idx = y

    z += net.module.means.data[idx]
    z *= net.module.cov_diag.data.abs().sqrt()
    x, _, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


@torch.no_grad()
def test(epoch, net, testloader, device, loss_fn, num_samples, output_dir, sample_dir, alpha):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    nll_meter = util.AverageMeter()
    cel_meter = util.AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, y in testloader:
            x = x.to(device)
            y = y.to(device)

            z, sldj, y_logits = net(x, reverse=False)

            nll = loss_fn(z, sldj, net.module.means[y], net.module.cov_diag)
            cel = F.cross_entropy(y_logits, y)
            loss = nll + alpha * cel

            nll_meter.update(nll.item(), x.size(0))
            cel_meter.update(cel.item(), x.size(0))
            loss_meter.update(loss.item(), x.size(0))

            progress_bar.set_postfix(nll=nll_meter.avg,
                                     cel=cel_meter.avg,
                                     loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    if nll_meter.avg < best_loss:
        print('Saving the best model...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'test_nll': nll_meter.avg,
            'test_cel': cel_meter.avg,
            'epoch': epoch,
            'net_config': (net.module.input_shape, net.module.q)
        }
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state, output_dir + '/best.pth.tar')
        best_loss = nll_meter.avg

    # Save samples and data
    images = sample(net, num_samples, device)
    os.makedirs(sample_dir, exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, (sample_dir+'/epoch_{}.png').format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional Glow')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=16, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')

    parser.add_argument(
        "--output_dir",
        default="ckpts",
        help="Directory to output logs and model checkpoints",
    )

    parser.add_argument(
        "--sample_dir",
        default="samples",
        help="Directory to output samples after each epochs",
    )

    parser.add_argument('--alpha', type=float, default=0.01, help='trade off coefficient in the class prediction loss')

    parser.add_argument('--dataset', type=str, default='MNIST', help='choose from MNIST, CIFAR10, CIFAR100, FashionMNIST or SVHN')

    best_loss = 1e10
    global_step = 0

    main(parser.parse_args())
