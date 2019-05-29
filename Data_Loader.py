import os
import torch
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def split_csv(file):
    data = []
    a_train_file = r'/mnt/storage/lupeng/ISIC/train.csv'
    a_test_file = r'/mnt/storage/lupeng/ISIC/test.csv'

    seed = 3
    np.random.seed(seed)
    train_indices = np.random.choice(25331, 20265, replace=False)  # 设置随机数生成从0-150中随机挑选120个随机数
    test_indices = np.array(list(set(range(25331)) - set(train_indices)))
    # test_indices = np.random.choice(len(residue), 30, replace=False)  # 如果训练集和测试集综合的数据加起来就是一整个数据集则不需要这个操作

    with open(file)as afile:
        a_reader = csv.reader(afile)  # 从原始数据集中将所有数据读取出来并保存到a_reader中
        labels = next(a_reader)  # 提取第一行设置为labels
        for row in a_reader:  # 将a_reader中每一行的数据提取出来并保存到data的列表中
            data.append(row)

    # 生成训练数据集
    if not os.path.exists(a_train_file):
        with open(a_train_file, "w", newline='') as a_trian:
            writer = csv.writer(a_trian)
            writer.writerows([labels])  # 第一行为标签行
            writer.writerows(np.array(data)[train_indices])
            a_trian.close()

    # 生成测试数据集
    if not os.path.exists(a_test_file):
        with open(a_test_file, "w", newline='')as a_test:
            writer = csv.writer(a_test)
            writer.writerows([labels])  # 第一行为标签行
            writer.writerows(np.array(data)[test_indices])
            a_test.close()


def read_labels_csv(file,header=True):
    images = []
    num_categories = 0
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class ISICDataset(Dataset):
    def __init__(self,csv_file,image_path,transform=None):
        self.images = read_labels_csv(csv_file)
        self.root_dir = image_path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name,target = self.images[index]
        # print(os.path.join(self.root_dir,image_name+'.jpg'))
        image = Image.open(os.path.join(self.root_dir,image_name+'.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image,target


if __name__ == '__main__':
    split_csv(file=r"/mnt/storage/lupeng/ISIC/ISIC_2019_Training_GroundTruth.csv")
