
requirements:
tensorflow-gpu == 2.3.1
CUDA == 11.0


编写模型：
1）model.py : 定义模型
2）vgg19_loss.py : 定义VGG19损失 (即论文中的content loss)
3）Utils.py : 处理及保存图像所用到的函数
4）img_process.py : 将dataset压缩成npz文件
5）metrics.py : psnr与ssim评价指标


训练模型：
1）运行 img_process.py 将数据集进行处理生成npz文件，目的是加快训练结果
2）运行 train.py 进行训练，本模型经过 3000 epochs 的训练
3）将模型保存为gen_model3000.h5 ( ./Trained_model/gen_model3000.h5 )

通过HTML页面进行展示：
1）运行 main.py 
2）打开index.html 选取想要的图片即可

实际应用：
1）运行applicationV2.py
2) 传入所需图片文件路径
3）在application文件夹查看生成结果

