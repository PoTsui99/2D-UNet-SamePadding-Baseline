from argparser import parse_args
import time
from pathlib import Path
import joblib
import torch
from unet import Unet
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import iou_score
from collections import OrderedDict


class AverageMeter(object):
    """计数, 更新平均值"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = parse_args()
    folder_name = 'BraTS-2018-UNet' + '_%s' % hash(time.time())
    # args.name = '%s_%s' % (args.name, hash(time.time()))
    output_path = Path('../models/%s' % folder_name)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # save and serialize args
    with open('../models/%s/args.txt' % folder_name, 'w') as f:
        for arg in vars(args):
            f.write('%s: %s\n' % (arg, getattr(args, arg)))
    joblib.dump(args, '../models/%s/args.pkl' % folder_name)

    criterion = nn.BCEWithLogitsLoss().cuda()
    cudnn.benchmark = True

    data_paths = list(Path('./train_data_float16').iterdir())
    target_paths = list(Path('./train_ground_truth_float16').iterdir())

    # train/test split
    train_data_paths, val_data_paths, train_target_paths, val_target_paths = \
        train_test_split(data_paths, target_paths, test_size=0.2, random_state=822921)

    model = Unet()
    model = model.cuda()
    # set optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    train_dataset = Dataset(train_data_paths, train_target_paths)
    val_dataset = Dataset(val_data_paths, val_target_paths)

    # pin_memory=True while memory is ample
    train_iter = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_iter = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    best_iou = 0
    trigger_early_stop = 0  # early stop标志, 验证集上的IoU为early stop观察指标
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, args.epochs))
        # training and validation in one epoch
        # train_log_cur_epoch = train(train_iter, model, criterion, optimizer, args)
        # val_log_cur_epoch = validate(val_iter, model, criterion, args)
        train_loss_history = AverageMeter()
        train_iou_history = AverageMeter()
        val_loss_history = AverageMeter()
        val_iou_history = AverageMeter()

        model.train()  # train mode influences batch normalization, drop out

        # training loop
        for data, target in tqdm(train_iter, total=len(train_iter)):
            data = data.cuda()
            target = target.cuda()

            output = model(data)
            loss = criterion(output, target)
            iou = iou_score(output, target)

            # batch_size not essentially equals to data.size(0) when drop_last=False
            train_loss_history.update(loss.item(), data.size(0))  # loss is a tensor
            train_iou_history.update(iou, data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluation mode
        model.eval()

        # validation loop
        with torch.no_grad():  # validation dont need weights' update
            for data, target in tqdm(val_iter, total=len(val_iter)):
                data = data.cuda()
                target = target.cuda()

                output = model(data)
                loss = criterion(output, target)
                iou = iou_score(output, target)

                val_loss_history.update(loss.item(), data.size(0))
                val_iou_history.update(iou, data.size(0))

        train_log_cur_epoch = OrderedDict([('loss', train_loss_history.avg), ('iou', train_iou_history.avg)])
        val_log_cur_epoch = OrderedDict([('loss', val_loss_history.avg), ('iou', val_iou_history.avg)])

        print('train_loss: %.4f - train_iou: %.4f - val_loss: %.4f - val_iou: %.4f'
              % (train_log_cur_epoch['loss'], train_log_cur_epoch['iou'], val_log_cur_epoch['loss'],
                 val_log_cur_epoch['iou']))

        log_cur_epoch = pd.DataFrame({'epoch': epoch, 'lr': args.lr, 'loss': train_log_cur_epoch['loss'],
                                      'iou': train_log_cur_epoch['iou'], 'val_loss': val_log_cur_epoch['loss'],
                                      'val_iou': val_log_cur_epoch['iou']}, index=[0])

        log = pd.concat([log, log_cur_epoch], ignore_index=True)
        log.to_csv('../models/%s/log.csv' % folder_name, index=False)

        trigger_early_stop += 1

        cur_iou = val_log_cur_epoch['iou']
        if cur_iou > best_iou:
            # save parameters of the network
            torch.save(model.state_dict(), '../models/%s/model.pth' % folder_name)
            best_iou = cur_iou
            print("[info] Model weight saved")
            trigger_early_stop = 0

        # early stop when IoU is not improved after trigger_early_stop epoch
        if args.early_stop is not None:
            if trigger_early_stop >= args.early_stop:
                print(f"[info] Model not improved over last {trigger_early_stop} epochs, early stopped.")
                break  # 退出epoch循环
        torch.cuda.empty_cache()
