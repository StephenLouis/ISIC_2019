class DefaultConfigs(object):
    #1.string parameters
    # train_csv_path = r"/mnt/storage/lupeng/ISIC/test.csv"
    # val_csv_path = r'/mnt/storage/lupeng/ISIC/train.csv'
    # image_path = r'/mnt/storage/lupeng/ISIC/ISIC_2019_Training_Input/'
    train_csv_path = r"/home/liangxiaoqing/downloads/ISIC_2019_Training_Input/train.csv"
    val_csv_path = r'/home/liangxiaoqing/downloads/ISIC_2019_Training_Input/test.csv'
    image_path = r'/home/liangxiaoqing/downloads/ISIC_2019_Training_Input'

    model_name = "resnet50"
    checkpoint_save = './checkpoint.pth.tar'
    best_models = './best_model.pth.tar'

    resume = r'./checkpoint.pth.tar'

    gpus = [0,1,2,3]

    #2.numeric parameters
    gamma = 0.1
    momentum = 0.9
    num_worker = 8
    epochs = 100
    batch_size = 64
    img_height = 300
    img_weight = 300
    num_classes = 8
    seed = 888
    lr = 1e-4 * len(gpus)
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
