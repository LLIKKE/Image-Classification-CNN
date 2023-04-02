import torch as tc


def exp_mov_avg(nets, net, alpha=0.999, global_step=999):
    """ 参数优化：指数平均数 ema_para = ema_para * alpha + param * (1 - alpha)
    nets: 做了ema的参数网络
    net: 训练中实际使用的网络
    global_step: 迭代次数（训练次数）
    使用： 网络实例化时拷贝一个nets = copy.deepcopy(net) ， 然后训练中调用exp_mov_avg(nets, net, global_step=step)
   y.add_(x,alpha) 指定x按比例alpha缩放后，加到y上，并取代y的内存y.mul_(x,other) 将x乘以other，取代y的内存存储，这里用带’_‘的函数就已经取代原参数了"""
    '''使用的时候，需额外定义一个nets'''
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(nets.parameters(), net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def create_lr_scheduler(optimizer,
                        num_step: int,  # every epoch has how much step
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,  # warmup进行多少个epoch
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha  # 对于alpha的一个线性变换，alpha是关于x的一个反比例函数变化
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
