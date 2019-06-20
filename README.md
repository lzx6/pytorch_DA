# pytorch domain adaptation
# 环境要求
- pytorch 1.0
- torchvision 0.2.1
- CUDA9.1
- python3
# 数据
**MNIST-->MNIST-M**
![mnist.png](https://upload-images.jianshu.io/upload_images/16293451-90bfebadcf59a2a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![mnist_m.png](https://upload-images.jianshu.io/upload_images/16293451-d043cb413d558632.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 框架
![dann.png](https://upload-images.jianshu.io/upload_images/16293451-32c144b8f522418f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![dann.png](https://upload-images.jianshu.io/upload_images/16293451-99fdda9f32c38081.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**说明**

- 整个框架跟DANN原论文都一样，在其中一些层添加了dropout和BN(model_3_channel.py)
- 无迁移方法：除去域判别器
- mmd: 对feature extractor的特征进行多核mmd约束
- DANN



#结果


![result.png](https://upload-images.jianshu.io/upload_images/16293451-990607b88d760c5d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 参考
1. https://arxiv.org/pdf/1505.07818.pdf[DANN]