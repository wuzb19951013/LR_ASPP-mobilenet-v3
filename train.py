"""
    train

Author: Zhengwei Li
Date  : 2018/12/24
"""

import argparse
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from data import dataset
from model import network


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Fast portrait matting !')
    parser.add_argument('--dataDir', default='./DATA/',
                        help='dataset directory')
    parser.add_argument('--testDir', default='./TEST/',
                        help='evaluation directory')

    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument(
        '--trainData', default='human_matting_data', help='train dataset name')
    parser.add_argument(
        '--trainList', default='./data/train.txt', help='train img ID')
    parser.add_argument(
        '--testList', default='./test/test.txt', help='test img ID')

    parser.add_argument('--load', default='human_matting', help='save model')

    parser.add_argument('--finetuning', action='store_true',
                        default=False, help='finetuning the training')
    parser.add_argument('--without_gpu', action='store_true',
                        default=False, help='no use gpu')

    parser.add_argument('--nThreads', type=int, default=4,
                        help='number of threads for data loading')
    parser.add_argument('--train_batch', type=int, default=4,
                        help='input batch size for train')
    parser.add_argument('--patch_size', type=int,
                        default=256, help='patch size for train')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--nEpochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=1,
                        help='number of epochs to save model')

    args = parser.parse_args()
    print(args)
    return args


def set_lr(args, epoch, optimizer):

    lrDecay = args.lrDecay
    decayType = args.lrdecayType
    if decayType == 'keep':
        lr = args.lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = args.lr * math.pow((1 - epoch / args.nEpochs), 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class Train_Log():
    def __init__(self, args):
        self.args = args

        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_model(self, model, epoch):

        # epoch_out_path = "{}/ckpt_e{}.pth".format(self.save_dir_model, epoch)
        # print("Checkpoint saved to {}".format(epoch_out_path))

        # torch.save({
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        # }, epoch_out_path)

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, lastest_out_path)

        model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
        torch.save(
            model,
            model_out_path)

    def save_best_model(self, model, epoch):

        lastest_out_path = "{}/ckpt_best.pth".format(self.save_dir_model)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, lastest_out_path)

        model_out_path = "{}/best_model_obj.pth".format(self.save_dir_model)
        torch.save(
            model,
            model_out_path)

    def load_model(self, model):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            lastest_out_path, ckpt['epoch']))

        return start_epoch, model

    def save_log(self, log):
        self.logFile.write(log + '\n')


def loss_function(args, img, trimap_pre, trimap_gt):

    # -------------------------------------
    # classification loss L_t
    # ------------------------
    # Cross Entropy
    # criterion = nn.BCELoss()
    # trimap_pre = trimap_pre.contiguous().view(-1)
    # trimap_gt = trimap_gt.view(-1)
    # L_t = criterion(trimap_pre, trimap_gt)

    eps = 1e-6
    # l_alpha
    L_t = torch.sqrt(torch.pow(trimap_pre - trimap_gt, 2.) + eps).mean()

    fg = torch.cat((trimap_gt, trimap_gt, trimap_gt), 1) * img
    fg_pre = torch.cat((trimap_pre, trimap_pre, trimap_pre), 1) * img

    L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()

    loss = 0.5*L_t + 0.5*L_composition

    return loss, L_t


def eval_model(model, args, testloader, device, optimizer):
    with torch.no_grad():
        loss_eval = 0
        model.eval()

        for i, sample_batched in enumerate(testloader):
            img, trimap_gt = sample_batched['image'], sample_batched['trimap']
            img, trimap_gt = img.to(device), trimap_gt.to(device)
            optimizer.zero_grad()
            trimap_pre = model(img)
            loss, L_cross = loss_function(args, img, trimap_pre, trimap_gt)
            loss_eval += loss.item()
        loss_eval = loss_eval / (i+1)

        return loss_eval


def main():
    print("=============> Loading args")
    args = get_args()

    print("============> Environment init")
    if args.without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU is available !")

    print("============> Building model ...")
    model = network.net()
    model.to(device)

    print("============> Loading datasets ...")
    train_data = getattr(dataset, args.trainData)(root_dir=args.dataDir,
                                                  imglist=args.trainList,
                                                  patch_size=args.patch_size)
    test_data = getattr(dataset, args.trainData)(root_dir=args.testDir,
                                                 imglist=args.testList,
                                                 patch_size=args.patch_size)
    trainloader = DataLoader(train_data,
                             batch_size=args.train_batch,
                             drop_last=True,
                             shuffle=True,
                             num_workers=args.nThreads,
                             pin_memory=True)
    testloader = DataLoader(test_data,
                            batch_size=args.train_batch,
                            drop_last=True,
                            shuffle=True,
                            num_workers=args.nThreads,
                            pin_memory=True)

    print("============> Set optimizer ...")
    lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, betas=(0.9, 0.999),
                           weight_decay=0.0005)

    print("============> Start Train ! ...")
    start_epoch = 1
    trainlog = Train_Log(args)
    if args.finetuning:
        start_epoch, model = trainlog.load_model(model)

    eval_loss_ = 100
    unfit = 0
    loss_init = 100
    for epoch in range(start_epoch, args.nEpochs+1):
        model.train()
        loss_ = 0
        L_cross_ = 0

        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)

        for i, sample_batched in enumerate(trainloader):

            img, trimap_gt = sample_batched['image'], sample_batched['trimap']
            img, trimap_gt = img.to(device), trimap_gt.to(device)

            trimap_pre = model(img)
            loss, L_cross = loss_function(args, img, trimap_pre, trimap_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()
            L_cross_ += L_cross.item()

        if epoch % args.save_epoch == 0:

            # speed = (t1 - t0) / 60

            loss_ = loss_ / (i+1)
            L_cross_ = L_cross_ / (i+1)

            eval_loss = eval_model(model, args, testloader, device, optimizer)

            log = "[{} / {}] \tLr: {:.5f}\nloss: {:.5f}\tloss_t: {:.5f}\tloss_eval: {:.5f}\tloss_val: {:.5f}\t" \
                .format(epoch, args.nEpochs,
                        lr,
                        loss_,
                        L_cross_,
                        eval_loss_,
                        eval_loss)
            print(log)
            trainlog.save_log(log)

            if loss_init > loss_:
                loss_init = loss_
                trainlog.save_model(model, epoch)
                unfit = 0
            else:
                unfit += 1

            if eval_loss_ > eval_loss:
                trainlog.save_best_model(model, epoch)
                eval_loss_ = eval_loss
            #     unfit = 0
            # else:
            #     unfit += 1

        if unfit >= 8:
            break


if __name__ == "__main__":
    main()
