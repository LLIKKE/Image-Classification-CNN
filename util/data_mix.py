import numpy as np
import torch as tc


def rand_bbox(size, lam):
    '''
    产生待裁剪的随机区域坐标
    '''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # clip:将数组值限定在0-w之间，大的改成w,小的改成0
    # // 求商运算
    # cx,cy是cut区域的中心像素点。cut_w,cut_h是裁剪区域大小
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, beta=1, cut_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    r = np.random.rand(1)
    if r < cut_prob:
        return x, y
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1
    batch_size = x.size()[0]

    # 得到一个将0~batcisize-1个数值随机打乱的的序号张量
    index = tc.randperm(batch_size).cuda()

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_sliced = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    x_m = x.clone()
    x_m[:, :, bbx1:bbx2, bby1:bby2] = x_sliced
    return x_m, y_a, y_b, lam


def mixup_data(x, y, beta=1, cut_prob=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    r = np.random.rand(1)
    if r < cut_prob:
        return x, None, None, None
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1
    batch_size = x.size()[0]

    index = tc.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_and_cutmix(x, y, beta=1, cut_prob=0.5):
    r = np.random.rand(1)
    if r < cut_prob:
        return x, None, None, None
    p = np.random.rand(1)
    if p > 0.5:
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, beta=beta, cut_prob=0.)
    else:
        mixed_x, y_a, y_b, lam = mixup_data(x, y, beta=beta, cut_prob=0.)
    return mixed_x, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)