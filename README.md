# ISIC 2019 皮肤病变分析对黑色素瘤的检测
网址： https://challenge2019.isic-archive.com/  
>ISIC 2019的目标是在九种不同的诊断类别中对皮肤镜图像进行分类：  
>&emsp;1.黑色素瘤  
> &emsp;2.黑素细胞痣  
>&emsp;3.基底细胞癌  
>&emsp;4.光化性角化病  
>&emsp;5.良性角化病（太阳能lentigo /脂溢性角化病/扁平苔藓样角化病）  
>&emsp;6.皮肤纤维瘤  
>&emsp;7.血管病变  
>&emsp;8.鳞状细胞癌  
>&emsp;9.没有其他人  
>**共有25,332张图片可用于8个不同类别**。此外，测试数据集（计划发布的8月2日）将包含一个未在训练数据中表示的额外异常类，开发系统必须能够识别。

## 依赖
+ Python 3.6.5
+ PyTorch 1.0.1
+ Torchvision 0.2.2
+ Tqdm 4.32.1
+ TensorboardX 1.6
+ cnn-finetune 0.5.3

## 用法说明
**第一阶段**：比赛方共提供25332张图片，和相应标签文件（CSV文件）。**由于没有给测试集**，我们需手动将所给数据划分为训练集和验证集。  
1. 运行Data_Loader.py中split_csv的函数（自行修改path），生成tran.csv和val.csv。
2. 修改config的路径、gpu、batch_size等参数
3. 运行main.py即可进行训练。（自行修改超参数）
