#coding:utf-8
import random
import time
import warnings
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch import nn,optim
from torchvision.models.resnet import resnet50
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Data_Loader import *
from utils.utils import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
from cnn_finetune import make_model


#1. set random.seed and cudnn performance
# random.seed(config.seed)
# np.random.seed(config.seed)
# torch.manual_seed(config.seed)
# torch.cuda.manual_seed_all(config.seed)
#os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
warnings.filterwarnings('ignore')

best_acc = 0  # best test accuracy

def main():
    global best_acc
    writer = SummaryWriter()
    # load dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ISICDataset(csv_file=config.train_csv_path,image_path=config.image_path,
                                transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize
                                ]))
    val_dataset = ISICDataset(csv_file=config.val_csv_path,image_path=config.image_path,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  normalize
                              ]))

    train_dataloader = DataLoader(
        train_dataset,batch_size=config.batch_size*len(config.gpus),shuffle=True,num_workers=config.num_worker,pin_memory=True)
    val_dataloader = DataLoader(
        val_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_worker,pin_memory=True
    )

    #   create model
    print("=====> Loading model..")
    
    # model = make_model('se_resnext50_32x4d', num_classes=config.num_classes+1, pretrained=True)

    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048,config.num_classes+1)
    model = torch.nn.DataParallel(model, device_ids=config.gpus)
    model.cuda(device=config.gpus[0])

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=config.momentum,weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()

    # some parameters restart model
    resume = config.resume

    # Resume
    if resume:
        if os.path.isfile(config.resume_path):
            print("=> loading checkpoint '{}'".format(config.resume_path))
            checkpoint = torch.load(config.resume_path)
            config.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if config.gpus is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc = best_acc.to(config.gpus[0])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    # Train and val
    print("=====> Start training..")
    for epoch in range(config.start_epoch,config.epochs):
        # get Learning_rate
        Learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print("EPOCH: %i" % epoch)
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
        print("EPOCH: %i   Tranin_ACC:%f  Train_LOSS:%f  lr:%f" % (epoch, train_acc, train_loss,Learning_rate))
        val_loss, val_acc = validate(val_dataloader, model, criterion)
        print("EPOCH: %i   Val_ACC:%f  Val_LOSS:%f" % (epoch, val_acc, val_loss))

        ####  ！！！！！！此处需调整
        if epoch >= 140:
            Learning_rate = config.lr * 1.0 / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = Learning_rate

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        #   使用tensorboard
        writer.add_scalars('Loss', {'train': train_loss,
                                    'valid': val_loss}, epoch)
        writer.add_scalars('Acc', {'train': train_acc,
                                   'valid': val_acc}, epoch)
        writer.add_scalars('lr',{'lr':Learning_rate}, epoch)


def train(train_loader, model, criterion, optimizer):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()


    for i, (inputs, targets) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # one_hot to Long
        targets = np.argmax(targets,axis=1)

        inputs = Variable(inputs).cuda(device=config.gpus[0])
        targets = Variable(torch.from_numpy(np.array(targets)).long()).cuda(device=config.gpus[0])

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)



def validate(val_loader, model, criterion):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(tqdm(val_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # one_hot to Long
        targets = np.argmax(targets,axis=1)

        inputs = Variable(inputs).cuda(device=config.gpus[0])
        targets = Variable(torch.from_numpy(np.array(targets)).long()).cuda(device=config.gpus[0])

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)





if __name__ =="__main__":
    main()

