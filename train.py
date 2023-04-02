import datetime
import os
import torch as tc
import torch.nn as nn
from model.network import choose_network
from util.optimize import create_lr_scheduler
from util.util import data_loader
from util.Func import count_parameters, setup_seed, draw, test_or_eval, find_index_10_max, train_one_epoch, \
    save_checkpoint
import argparse

if __name__ == '__main__':

    # Training Arguments Settings
    parser = argparse.ArgumentParser(description='training_setting')
    parser.add_argument('--data_root', type=str, default='E:/DATASET')
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--seed', default=43)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--cutmix_prob', type=float, default=0.5)
    parser.add_argument('--mixup_prob', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='./save_weights')
    parser.add_argument('--print_parser', default=True)
    parser.add_argument('--print_model', default=True)
    args = parser.parse_args()

    # 打印超参
    if args.print_parser:
        print('\n----------Argument Values-----------')
        for name, value in vars(args).items():
            print('%s: %s' % (str(name), str(value)))
        print('------------------------------------\n')
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    setup_seed(args.seed)

    train_loader, eval_loader, test_loader = data_loader(args.dataset, path=args.data_root,
                                                         transformss=[], args=args)

    cri = nn.CrossEntropyLoss()

    model = choose_network(model=args.model, num_classes=args.classes)
    model.to(args.device)
    optimizer = tc.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epoch, warmup_epochs=10)
    if args.print_model:
        print(model)
    count = count_parameters(model, print_=True)
    data = {'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []}
    best_acc = -1
    test_best_acc = -1
    test_acclist = []
    test_acc = -1
    for epoch in range(args.epoch):
        train_loss, train_acc, lr = train_one_epoch(train_loader, model, cri, optimizer, epoch, scheduler, args)
        data['train_acc'].append(train_acc)
        data['train_loss'].append(train_loss)
        draw(list(range(len(data['train_loss']))), data['train_loss'], named='train_loss', title='train_loss')
        draw(list(range(len(data['train_acc']))), data['train_acc'], named='train_acc', title='train_acc')

        eval_acc, _ = test_or_eval(model=model, dataloader=eval_loader, cri=cri, args=args)
        data['eval_acc'].append(eval_acc)
        draw(list(range(len(data['eval_acc']))), data['eval_acc'], named='eval_acc', title='eval_acc')
        test_acc, _ = test_or_eval(model=model, dataloader=test_loader, cri=cri, args=args)
        test_acclist.append(test_acc)
        draw(list(range(len(test_acclist))), test_acclist, named='test_acc', title='test_acc')
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(model, optimizer, epoch, scheduler, args)
            if test_best_acc < test_acc:
                test_best_acc = test_acc
        if epoch % 50 == 0:
            print(
                f'epoch = {epoch},train_acc = {train_acc} train_loss = {train_loss}, eval_acc={eval_acc}')
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_info = f'epoch = {epoch}, ' \
                             f'train_acc = {train_acc}, ' \
                             f'train_loss = {train_loss}, ' \
                             f'eval_acc={eval_acc}, ' \
                             f'test_acc={test_acc}\n'
                f.write(train_info)

    test_aver = find_index_10_max(data['eval_acc'], test_acclist, 10)
    with open(results_file, "a") as f:
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        train_info = f'para = {count},\n ' \
                     f'test_acc_best : {test_best_acc},\n ' \
                     f'test_acc_early : {test_aver}\n ' \
                     f'test_acc_last :{sum(test_acclist[-10:]) / 10}\n ' \
                     f"train_acc: {sum(data['train_acc'][-10:]) / 10}\n" \
                     f"train_loss: {sum(data['train_loss'][-10:]) / 10}\n" \
                     f"eval_acc:{sum(data['eval_acc'][-10:]) / 10}\n" \
                     f"eval_loss:{sum(data['eval_loss'][-10:]) / 10}\n\n"
        f.write(train_info)
    print("--------------------------end----------------------------")
