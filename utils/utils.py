import shutil
import torch
import os
from config import config
import numpy as np

def save_checkpoint(state, is_best, filename=config.checkpoint_save):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, config.best_models)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_balanced_accuracy(output, target, topk=1):
    _, predictions = output.topk(topk, 1, True, True)
    predictions = predictions.data.cpu().numpy()
    labels = target.cpu().numpy()[:, np.newaxis]
    
    precisions = []
    recalls = []
    for i in range(0, config.num_classes+1):
        tp = np.sum(np.logical_and(
            np.equal(labels, i), np.equal(predictions, i)
        ).astype(np.int32))
        fp = np.sum(np.logical_and(
            np.logical_not(np.equal(labels, i)), np.equal(predictions, i)
        ).astype(np.int32))
        fn = np.sum(np.logical_and(
            np.equal(labels, i), np.logical_not(np.equal(predictions, i))
        ).astype(np.int32))
        precisions = precisions + [100.0*tp/(tp+fp+1e-6)]
        recalls = recalls + [100.0*tp/(tp+fn+1e-6)]
    return np.mean(recalls), np.mean(precisions)


# def time_to_str(t, mode='min'):
#     if mode=='min':
#         t  = int(t)/60
#         hr = t//60
#         min = t%60
#         return '%2d hr %02d min'%(hr,min)
#
#     elif mode=='sec':
#         t   = int(t)
#         min = t//60
#         sec = t%60
#         return '%2d min %02d sec'%(min,sec)
#
#     else:
#         raise NotImplementedError

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    if epoch < 8:
        lr = 2e-4 + (config.lr-2e-4) * iteration / (epoch_size * 7)
    else:
        lr = config.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
