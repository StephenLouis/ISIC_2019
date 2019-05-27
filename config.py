class DefaultConfigs(object):
    #1.string parameters
    # train_csv_path = r"/mnt/storage/lupeng/ISIC/test.csv"
    # val_csv_path = r'/mnt/storage/lupeng/ISIC/train.csv'
    # image_path = r'/mnt/storage/lupeng/ISIC/ISIC_2019_Training_Input/'
    train_csv_path = r"/mnt/storage/lupeng/ISIC/train.csv"
    val_csv_path = r'/mnt/storage/lupeng/ISIC/test.csv'
    image_path = r'/mnt/storage/lupeng/ISIC/ISIC_2019_Training_Input'

    model_name = "resnet50"
    checkpoint_save = '/home/lupeng/code_space/checkpoint/ISIC_2019/checkpoint.pth.tar'
    best_models = '/home/lupeng/code_space/checkpoint/ISIC_2019/best_model.pth.tar'

    resume = r'/home/lupeng/code_space/checkpoint/ISIC_2019/checkpoint.pth.tar'

    gpus = "1"

    #2.numeric parameters
    gamma = 0.1
    momentum = 0.9
    num_worker = 5
    epochs = 100
    batch_size = 64
    img_height = 300
    img_weight = 300
    num_classes = 8
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()