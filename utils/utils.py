import shutil
import torch
import os
from config import config
import numpy as np
from sklearn.metrics import classification_report

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
        return res,pred


def get_balanced_accuracy(output, target, topk=1):
    _, predictions = output.topk(topk, 1, True, True)
    predictions = predictions.data.cpu().numpy()
    labels = target.cpu().numpy()[:, np.newaxis]

    tp_list = []
    fp_list = []
    fn_list = []
    for i in range(0, config.num_classes):
        tp = np.sum(np.logical_and(
            np.equal(labels, i), np.equal(predictions, i)
        ).astype(np.int32))
        fp = np.sum(np.logical_and(
            np.logical_not(np.equal(labels, i)), np.equal(predictions, i)
        ).astype(np.int32))
        fn = np.sum(np.logical_and(
            np.equal(labels, i), np.logical_not(np.equal(predictions, i))
        ).astype(np.int32))

        tp_list = tp_list + [tp]
        fp_list = fp_list + [fp]
        fn_list = fn_list + [fn]
    return tp_list, fn_list


# def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
#     if epoch < 8:
#         lr = 2e-4 + (config.lr-2e-4) * iteration / (epoch_size * 7)
#     else:
#         lr = config.lr * (gamma ** (step_index))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

# def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
#     if epoch > 50:
#         lr = config.lr * 0.2
#     else:
#         lr = config.lr * (gamma ** (step_index))
#
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr






def plot_classification_report(y_true,y_pred,acc,mean_class_recall,save_path):
    target_names = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC','unk']
    
    with open(save_path,mode='a') as f:
        f.write(classification_report(y_true, y_pred,target_names=target_names))
        f.write("ACC: %f" % (acc))
        f.write("mean_class_recall: %f"%(mean_class_recall))
