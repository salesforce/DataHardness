import torch
import argparse

from models import Glow
import numpy as np
import os


@torch.no_grad()
def sample(net, batch_size, device, priors, y=None, temperature=1.0):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    input_shape = net.module.cov_diag.shape
    z = torch.randn((batch_size, *input_shape), dtype=torch.float32, device=device)

    if y is None:
        idx = np.random.choice(list(range(len(priors))), batch_size, p=priors)
    else:
        idx = y

    z *= net.module.cov_diag.data.abs().sqrt() * temperature
    z += net.module.means.data[idx]
    x, _, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x, torch.from_numpy(idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default='mnist-ckpts/best.pth.tar',
        help="which model you want to generate samples from",
    )

    parser.add_argument("--batch_sz", type=int, default=100, help="number of samples per class per batch")

    parser.add_argument("--n_batches", type=int, default=600, help="number of batches per class")

    parser.add_argument("--temperature", type=float, default=1.0, help="temperature used to generate samples")

    parser.add_argument("--save_fp", type=str, default="saved_datasets/mnist", help="directory you want to store the dataset in")

    args = parser.parse_args()

    checkpoint = torch.load(args.model_path)
    input_shape, priors = checkpoint['net_config']
    net = Glow(num_channels=512, num_levels=3, num_steps=16, input_shape=input_shape, q=priors)
    net = torch.nn.DataParallel(net)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.load_state_dict(checkpoint['net'])
    net = net.to(device)
    net.eval()

    images = []
    labels = []
    for batch in range(args.n_batches):
        images_t, labels_t = sample(net, args.batch_sz, device=device, priors=priors, y=None, temperature=args.temperature)
        images.append(images_t)
        labels.append(labels_t)
        print('batch # ' + str(batch) + ' processed...')

    images = torch.cat(images)
    labels = torch.cat(labels)

    print('generated dataset of size ', images.shape)
    os.makedirs(args.save_fp, exist_ok=True)
    torch.save({'images': (images * 255.999).type(torch.uint8), 'labels': labels}, args.save_fp+'/data.h')