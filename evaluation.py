#coding:utf-8
import time
import warnings
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Data_Loader import *
from utils.utils import *
from tqdm import tqdm
from cnn_finetune import make_model


warnings.filterwarnings('ignore')

def list_add(a,b):
    c = []
    for i in range(len(b)):
        c.append(a[i]+b[i])
    return c

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = ISICDataset(csv_file=config.val_csv_path,image_path=config.image_path,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  normalize
                              ]))
    val_dataloader = DataLoader(
        val_dataset,batch_size=config.batch_size,shuffle=False,num_workers=config.num_worker,pin_memory=True
    )

    print("=====> Loading model..")

    model = make_model('se_resnext50_32x4d', num_classes=config.num_classes+1, pretrained=True)

    model = torch.nn.DataParallel(model, device_ids=config.gpus)
    model.cuda(device=config.gpus[0])
    
    criterion = nn.CrossEntropyLoss().cuda(device=config.gpus[0])

    cudnn.benchmark = True


    # Resume
    checkpoint = torch.load(config.resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    print("=====> Start evaluation..")

    acc,mean_recall = validate(val_dataloader, model, criterion)
    print("validate_acc: %f  validate_mean_recall: %f" % (acc,mean_recall))


def validate(val_loader, model, criterion):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc1 = AverageMeter()
    # top1 = AverageMeter()

    y_pred = []
    y_true = []

    tp_sum = [0 for i in range(config.num_classes)]
    fn_sum = [0 for i in range(config.num_classes)]

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
        
        acc,pred = accuracy(outputs.data, targets.data, topk=(1,))

        y_true.extend(targets.data.cpu().numpy().tolist())
        y_pred.extend(np.squeeze(pred.data.cpu().numpy().tolist()))
       

        # measure accuracy and record loss
        # prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        prec1 = get_balanced_accuracy(outputs.data, targets.data, topk=1)


        losses.update(loss.item(), inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))
        acc1.update(acc[0], inputs.size(0))

        tp_sum = [tp_sum[i] + prec1[0][i] for i in range(config.num_classes)]
        fn_sum = [fn_sum[i] + prec1[1][i] for i in range(config.num_classes)]


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    recall = [100.0 * tp_sum[i] / (tp_sum[i] + fn_sum[i]) for i in range(config.num_classes)]
    gt_sum = [(tp_sum[i] + fn_sum[i]) for i in range(config.num_classes)]
    mean_class_recall = float(np.mean(recall))

    plot_classification_report(y_true,y_pred,acc1.avg,mean_class_recall,'./classification_report.txt')

    return (acc1.avg, mean_class_recall)


if __name__ =="__main__":
    main()