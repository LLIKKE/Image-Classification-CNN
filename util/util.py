import numpy as np
import torch as tc
from torch import nn


def data_loader(dataset: str, path, transformss, args):
    import torchvision
    from torchvision.transforms import transforms
    from torch.utils.data import sampler
    if dataset == 'cifar100':
        train_loader = tc.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=path, download=False, train=True,
                                          transform=transforms.Compose(transformss
                                                                       +
                                                                       [
                                                                           transforms.ToTensor(),
                                                                           transforms.Normalize(
                                                                               [0.5070751592371323, 0.48654887331495095,
                                                                                0.4409178433670343],
                                                                               [0.2673342858792401, 0.2564384629170883,
                                                                                0.27615047132568404])])),
            batch_size=args.batch_size, sampler=sampler.SubsetRandomSampler(range(45000))
        )

        eval_loader = tc.utils.data.DataLoader(
            (torchvision.datasets.CIFAR100(root=path, download=False, train=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(
                                                   [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                   [0.2673342858792401, 0.2564384629170883,
                                                    0.27615047132568404])]))),
            batch_size=5000, sampler=sampler.SubsetRandomSampler(range(45000, 50000)))
        test_loader = tc.utils.data.DataLoader(
            (torchvision.datasets.CIFAR100(root=path, download=False, train=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(
                                                   [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                   [0.2673342858792401, 0.2564384629170883,
                                                    0.27615047132568404])]))),
            batch_size=2000, shuffle=False)
    elif dataset == 'cifar10':
        train_loader = tc.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=path, download=False, train=True,
                                         transform=transforms.Compose(transformss
                                                                      +
                                                                      [
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize(
                                                                              [0.5070751592371323, 0.48654887331495095,
                                                                               0.4409178433670343],
                                                                              [0.2673342858792401, 0.2564384629170883,
                                                                               0.27615047132568404])])),
            batch_size=args.batch_size, sampler=sampler.SubsetRandomSampler(range(45000))
        )

        eval_loader = tc.utils.data.DataLoader(
            (torchvision.datasets.CIFAR10(root=path, download=False, train=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                  [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                  [0.2673342858792401, 0.2564384629170883,
                                                   0.27615047132568404])]))),
            batch_size=5000, sampler=sampler.SubsetRandomSampler(range(45000, 50000)))
        test_loader = tc.utils.data.DataLoader(
            (torchvision.datasets.CIFAR10(root=path, download=False, train=False,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                  [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                  [0.2673342858792401, 0.2564384629170883,
                                                   0.27615047132568404])]))),
            batch_size=2000, shuffle=False)
    else:
        assert 1, 'dataset key error'
    return train_loader, eval_loader, test_loader



