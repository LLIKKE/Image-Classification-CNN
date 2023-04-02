import numpy as np
import torch as tc

from util.data_mix import cutmix_data, mixup_data, mixup_and_cutmix, mix_criterion


def find_index_10_max(eval_list, test_list, time=10):
    eval_list = eval_list.copy()
    index_list = []
    for i in range(len(eval_list)):
        p = eval_list.index(max(eval_list))
        index_list.append(p)
        eval_list[p] = -1
    sum = 0
    for i in index_list[:time]:
        sum += test_list[i]
    return sum / time


def test_or_eval(net, dataloader, cri, args):
    with tc.no_grad():
        net.eval()
        num_c = 0
        num_t = 0
        test_acc = 0
        eval_loss = 0
        eval_iter = 0
        for image, target in dataloader:
            image, target = image.to(args.device), target.to(args.device, dtype=tc.long)
            predict = net(image)
            eval_loss += cri(predict, target).item()
            eval_iter += 1
            pred = tc.argmax(predict.data, dim=1)
            num_c = num_c + tc.sum(pred == target).item()  # 正确分类
            num_t = num_t + target.numel()  # numel统计张量中元素总数
        test_acc = num_c / num_t  # 计算训练精度
    return test_acc, eval_loss / eval_iter


def stop(data_x, data_y):
    m = len(data_y)
    x_bar = np.mean(data_x)
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - m * (x_bar ** 2))

    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_delta += (y - w * x)
    b = sum_delta / m
    return w, b


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    model.train()
    iters = 0
    train_loss = 0
    num_c = 0
    num_t = 0
    for image, target in train_loader:
        if args.cutmix_prob > 0 and args.mixup_prob < 0:
            image, y_a, y_b, lam = cutmix_data(image, target, cut_prob=args.cutmix_prob)
        elif args.cutmix_prob < 0 and args.mixup_prob > 0:
            image, y_a, y_b, lam = mixup_data(image, target, cut_prob=args.mixup_prob)
        elif args.cutmix_prob > 0 and args.mixup_prob > 0:
            image, y_a, y_b, lam = mixup_and_cutmix(image, target, cut_prob=(args.cutmix_prob + args.mixup_prob) / 2)
        else:
            pass
        optimizer.zero_grad()
        image, target = image.to(args.device), target.to(args.device, dtype=tc.long)
        predict = model(image)
        if (args.cutmix_prob > 0 or args.mixup_prob > 0) and lam is not None:
            y_a, y_b = y_a.to(args.device, dtype=tc.long), y_b.to(args.device, dtype=tc.long)
            loss = mix_criterion(criterion, predict, y_a, y_b, lam)
        else:
            loss = criterion(predict, target)
        train_loss += loss.item()
        iters += 1
        loss.backward()
        optimizer.step()
        scheduler.step()
        pred = tc.argmax(predict.data, dim=1)
        num_c = num_c + tc.sum(pred == target).item()  # 正确分类
        num_t = num_t + target.numel()  # numel统计张量中元素总数
    train_acc = num_c / num_t  # 计算训练精度
    lr = optimizer.param_groups[0]["lr"]
    return train_loss / iters, train_acc, lr


def count_parameters(model, print_=True):
    num = sum([param.nelement() for param in model.parameters()])
    if print_ is True:
        print("Number of parameter: %.5fM" % (num / 1e6))
    return num


def setup_seed(seed):
    import random
    tc.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    tc.backends.cudnn.deterministic = True
    random.seed(seed)


def draw(xlist, ylist, save_path='', named='train_loss', y_limit=None, title='train_loss_decline', x_tag='iter',
         y_tag='loss', ):
    # plt.scatter(xlist, ylist, c='blue')
    from matplotlib import pyplot as plt
    plt.plot(xlist, ylist, lw=2, ls='-', c='b', alpha=0.5)
    plt.title(title)
    if y_limit is not None:
        plt.ylim(y_limit[0], y_limit[1])
    plt.xlabel(x_tag)
    plt.ylabel(y_tag)
    plt.savefig(save_path + f'{named}.png')
    plt.cla()


def draw_features(width, height, x, savepath, name):
    import cv2
    import os
    from matplotlib import pyplot as plt
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    x = x.cpu().detach().numpy()
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')

        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8

        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        # print("{}/{}".format(i, width * height))
    fig.savefig(savepath + name, dpi=100)
    fig.clf()
    plt.close()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, way="normal"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            验证集连续多少次loss上升会停止训练
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            采用GL标准，这个delta意味着刚计算的验证集loss比最好的验证集loss变差了 delta %
                            Default: 0
            way ():两种停止标准
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.way = way

    def __call__(self, val_loss, model, model_name, val_losslist=None):

        if val_losslist is None:
            val_losslist = []
        score = val_loss
        if len(val_losslist) >= 4:
            if abs(round(val_losslist[-4:-1][0], 4) - round(val_losslist[-4:-1][1], 4)) < 10e-12 and abs(
                    round(val_losslist[-4:-1][1], 4) - round(val_losslist[-4:-1][2], 4)) < 10e-12:
                self.early_stop = True
                print('损失稳定')
        if self.way == 'normal':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, model_name)
            elif score > self.best_score or abs(score - self.best_score) < 1e-12:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    print('损失上升')
            elif score < self.best_score:
                self.best_score = score
                self.counter = 0
                self.save_checkpoint(val_loss, model, model_name)
        if self.way == "PQ_alpha":
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, model_name)
            GL = 100 * (score / self.best_score - 1)
            P_K = 1000 * (sum(val_losslist[-self.patience - 1:-1]) / (
                    self.patience * min(val_losslist[-self.patience - 1:-1]) - 1))
            PQ_alpha = GL / P_K
            if PQ_alpha > self.delta:
                self.save_checkpoint(val_loss, model, model_name)

    def save_checkpoint(self, val_loss, model, model_name):
        """
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        """
        tc.save(model.state_dict(), f'vail_{model_name}')  # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss


def resume(checkpoint=False, net=None, optimizer=None):
    loss = []
    val_loss = []
    if checkpoint:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = tc.load('checkpoint/check_point.pth')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        val_loss = checkpoint['val_loss']
        return last_epoch, loss, val_loss
    else:
        last_epoch = 0
        return last_epoch, loss, val_loss


def save_checkpoint(net, optimizer, epoch, lr_scheduler, args, frequency=-1):
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             "lr_scheduler": lr_scheduler.state_dict()}
    tc.save(state, args.save_path + f'/best_model.pth')
    if frequency > 0:
        if epoch % frequency == 0:
            tc.save(state, f'checkpoint/check_point_{epoch}.pth')


def Histogram(data, save=False, named=None, path=None):
    from matplotlib import pyplot as plt
    import seaborn as sns
    if data.requires_grad:
        data = data.detach().numpy()
    else:
        data = np.array(data.cpu())
    plt.figure(figsize=(8, 4))
    plt.xlim(-10000, 10000)
    # plt.ylim(0, 100)
    sns.distplot(data, bins=10000, hist=True, kde=False, norm_hist=False,
                 rug=True, vertical=False, label='distplot',
                 axlabel='x', hist_kws={'color': 'y', 'edgecolor': 'k'})
    # sns.displot(data=data)
    # 用标准正态分布拟合
    plt.legend()
    plt.grid(linestyle='--')
    if save:
        plt.savefig(path + f'{named}.png')
    plt.show()
