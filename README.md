# 2D_UNet_SamePadding_Baseline
This repo stores the code of the pipeline of classical 2D U-Net baseline. The demo is run on the BraTS-2018 dataset.

本DEMO在BraTS-2018数据集(训练集)上运行

## 准备工作
首先从[BraTS-2018官方网页](https://ipp.cbica.upenn.edu/#BraTS18eval_trainingPhase)中下载数据集并进行解压.

然后将数据集放入项目文件中, 文件目录如下:

+ 2D_UNet_SampePadding_Baseline
    + Code
    + dataset
        + MICCAI_BraTS_2018_Data_Training
            + HGG
            + LGG
            + survival_data.csv
    + models

## 运行Demo
1. 运行preprocessing.py(请确保BraTS-2018数据集在正确路径上).
2. 训练: cd到Code目录, 执行命令, 如`python ./train.py --epochs 10`
3. 测试: 执行命令, 如`python ./test.py --name BraTS-2018-UNet_1439070506901286798 --batch-size 8`

训练及测试所有可选参数及默认值请自行查看`2D_UNet_SampePadding_Baseline/Code/argparser.py`.
