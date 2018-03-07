import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from stereo_loader import StereoDataset
from helper import *
from twinnet import TwinNet

parser = argparse.ArgumentParser(description='Twinnet Expiriemental Stereo Architecture')

parser.add_argument('-stereo', type=bool, default=True, help='use the stereo architecture (default true')
parser.add_argument('-batchsize', type=int, default=1, help='batch size (default 1')
parser.add_argument('-epochs', type=int, default=100, help='batch size (default 100')
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate (default 1e-4')
parser.add_argument('-stepfreq', type=int, default=15, help='how often the learning rate is adjusted (default 15')
parser.add_argument('-gamma', type=float, default=0.5, help='step size of the LR scheduler reduction (default .5)')
parser.add_argument('-numclass', type=int, default=11, help='number of classes in dataset (default 1)')
args = parser.parse_args()

ignore_classes = torch.LongTensor([0])  # list of class numbers to ignore in training and evaluation
means = np.array([0, 0, 0])  # training set means to center data
root_dir = "dataset/"
data_train_csv = os.path.join(root_dir, "train.csv")
data_val_csv = os.path.join(root_dir, "val.csv")


use_stereo = args.stereo
batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
step_freq = args.stepfreq
gamma = args.gamma
n_class = args.numclass

configuration = "Twinnet_Adam_LOSS_batch{}_epoch{}_scheduler-step{}-gamma{}_lr{}".format(batch_size, epochs, step_freq, gamma, lr)
print("Configs:", configuration)

#
# Create Models Dir
####################################################
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configuration)
# create dir for score
score_dir = os.path.join("scores", configuration)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)

#
# GPU Setup
####################################################

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

#
# Data Loaders
####################################################
train_data = StereoDataset(csv_file=data_train_csv, root_dir=root_dir, num_classes=n_class, means=means, use_stereo=use_stereo)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

val_data = StereoDataset(csv_file=data_val_csv, root_dir=root_dir, num_classes=n_class, means=means, use_stereo=use_stereo)
val_loader = DataLoader(val_data, batch_size=1, num_workers=0)

#
# Create Model
####################################################

twinnet = TwinNet(num_classes=n_class, use_stereo=use_stereo)
# twinnet = torch.load("models/Twinnet_dataset3_CE_LOSS_batch1_epoch202_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay0.0001")
print(get_n_params(twinnet))

if use_gpu:
    ts = time.time()
    twinnet = twinnet.cuda()
    twinnet = nn.DataParallel(twinnet, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

#
# Setup
####################################################
stan_weights = torch.ones(n_class)
stan_weights[ignore_classes] = 0.0

if use_gpu:
    stan_weights = stan_weights.cuda()

criterion = nn.CrossEntropyLoss(weight=stan_weights)
optimizer = optim.Adam(twinnet.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_freq, gamma=gamma)  # decay learning rate by gamma every step_size of epochs

#
# TRAIN
####################################################

def train():
    for epoch in range(epochs):
        scheduler.step()
        time_sec = time.time()
        for iterat, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs_left = Variable(batch['XL'].cuda())
                if use_stereo:
                    inputs_right = Variable(batch['XR'].cuda())
                labels = Variable(batch['Y'].cuda()).long()
            else:
                inputs_left = Variable(batch['XL'])
                if use_stereo:
                    inputs_right = Variable(batch['XR'])
                labels = Variable(batch['Y'])

            if use_stereo:
                outputs = twinnet(inputs_left, inputs_right)  # pass in left and right images
            else:
                outputs = twinnet(inputs_left)

            flat = outputs.view(1, n_class, -1)
            labels = labels.view(1, -1)
            if use_gpu:
                flat = flat.cuda()
                labels = labels.cuda()


            loss = criterion(flat, labels)
            loss.backward()
            optimizer.step()

            if iterat % 10 == 0:
                print("Epoch:", epoch, "Iteration:", iterat, "Loss:", loss.data[0])

        print("Finish epoch", epoch, "Time Elapsed", time.time() - time_sec)
        torch.save(twinnet, model_path)
        val(epoch)


#
# VALIDATION
####################################################
def val(epoch):
    twinnet.eval()  # this preps your dropout and batchnorm layers for validation
    total_ious = []

    class_accs = []

    pixel_correct = []
    pixel_total = []
    for iterat, batch in enumerate(val_loader):
        if use_gpu:
            # we need to declare volatile in training so that grads are not kept reducing memory
            inputs_left = Variable(batch['XL'].cuda(), volatile=True)
            if use_stereo:
                inputs_right = Variable(batch['XR'].cuda(), volatile=True)
        else:
            inputs_left = Variable(batch['XL'], volatile=True)
            if use_stereo:
                inputs_right = Variable(batch['XR'], volatile=True)
        if use_stereo:
            output = twinnet(inputs_left, inputs_right)
        else:
            output = twinnet(inputs_left)
        output = output.data.cpu().numpy()

        target = batch['Y'].numpy()

        bat, _, h, w = output.shape

        actmap = np.squeeze(output.transpose(0, 2, 3, 1))[:, :, 0]

        pred = np.squeeze(output.transpose(0, 2, 3, 1))

        pred = pred.reshape(-1, n_class).argmax(axis=1)

        pred = pred.reshape(bat, h, w)

        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            correct, total, class_ind = pixel_acc(p, t)
            pixel_total.append(total)
            pixel_correct.append(correct)
            class_accs.append(class_ind)

    # Calculate average IoU
    total_ious = np.array(total_ious).T
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = (np.array(pixel_correct).sum()) / (np.array(pixel_total).sum())

    np_class_accs = np.array(class_accs)
    class_accuracies = np_class_accs.mean(axis=0)
    print("Epoch:", epoch, "Pixel_acc:", pixel_accs, "MeanIoU:", np.nanmean(ious))
    print("Class Accuracies:", class_accuracies)
    print("Class IoUs:", ious)
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    y_pred = np.squeeze(pred.reshape(-1))
    y_true = np.squeeze(target.reshape(-1))

    class_accss = []
    for i in range(n_class):
        class_samples = np.where(y_true == i)
        preds = y_pred[class_samples]
        num_total = preds.shape[0]
        correct = np.where(preds == i)
        num_correct = len(correct[0])
        if num_total > 0:
            percent = num_correct / num_total
        else:
            percent = 0
        class_accss.append(percent)
    for ignore in ignore_classes:
        unclass = np.where(y_true == ignore)
        y_true = np.delete(y_true, unclass)
        y_pred = np.delete(y_pred, unclass)

    correct = (y_pred == y_true).sum()
    total = (y_true == y_true).sum()
    # class_accs = metrics.accuracy_score(y_true, y_pred)
    return correct, total, class_accss


if __name__ == "__main__":
    train()
