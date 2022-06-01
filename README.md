# 2D_UNet_SamePadding_Baseline
This repo stores the code of the pipeline of classical 2D U-Net baseline. The demo is run on the BraTS-2018 dataset.

本demo在BraTS-2018数据集(训练集)上运行. 如需使用其它数据集, 需自行修改输入输出channel、数据集路径、数据预处理等.

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
1. 运行preprocessing.py(请确保BraTS-2018数据集在正确路径上), 将会在`Code/`下生成`./train_data_float16`、`./train_ground_truth_float16`、`./test_data`、`./test_ground_truth`四个路径.
2. 训练: cd到`Code/``目录, 执行命令, 如`python ./train.py --epochs 10`
3. 测试: cd到`Code/``目录, 执行命令, 如`python ./test.py --name BraTS-2018-UNet_1439070506901286798 --batch-size 8`

训练及测试所有可选参数及默认值请自行查看`Code/argparser.py`.

如对您有帮助, 可如下引用本repo:

@misc{2d_unet_baseline,
  author = {CuiBo},
  title = {2D_UNet_SamePadding_Baseline},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PoTsui99/2D_UNet_SamePadding_Baseline}},
}
