# coding:utf-8
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
import glob
import os

allimages = glob.glob('D:/PS_dataset/newpsdatav1/pstype3/output/*.jpg')  # 获取目录下所有图片路径

# print(allimages)
for path in allimages:
    #     print(path)
    name = os.path.basename(path)[:-4]
    print(name)
    images = cv2.imread(path, 1)
    images = [images]  # 数据量变成4倍
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
    seq = iaa.Sequential(
        [
            # iaa.Fliplr(0.2),  # 对50%的图像进行镜像翻转
            # iaa.Flipud(0.2),  # 对20%的图像做左右翻转
            # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            #
            # iaa.Add((-10, 10), per_channel=0.5),

            # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop的幅度为0到10%

            # sometimes(iaa.Affine(  # 部分图像做仿射变换
            #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%
            #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%
            #     # rotate=(-30, 30),  # 旋转±30度
            #     # shear=(-16, 16),  # 剪切变换±16度（矩形变平行四边形）
            #     # order=[0, 1],# 使用最邻近差值或者双线性差值
            #     cval=255,  # 全白全黑填充
            #     # mode=ia.ALL  # 定义填充图像外区域的方法
            # )),

            # 使用下面的0个到2个之间的方法增强图像
            iaa.SomeOf((0, 2),
                       [
                           # 将部分图像进行超像素的表示
                           # sometimes(
                           #     iaa.Superpixels(
                           #         p_replace=(0, 1.0),
                           #         n_segments=(20, 200)
                           #     )
                           # ),

                           # 用高斯模糊，均值模糊，中值模糊中的一种增强
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # 锐化处理
                           iaa.Sharpen(alpha=(0, 0.3), lightness=(0.9, 1.1)),

                           # #浮雕效果
                           # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                           # 边缘检测，将检测到的赋值0或者255然后叠在原图上
                           # sometimes(iaa.OneOf([
                           #     iaa.EdgeDetect(alpha=(0, 0.7)),
                           #     iaa.DirectedEdgeDetect(
                           #         alpha=(0, 0.7), direction=(0.0, 1.0)
                           #     ),
                           # ])),

                           # 加入高斯噪声
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),

                           # 将1%到10%的像素设置为黑色
                           # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                           # iaa.OneOf([
                           #     iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           #     iaa.CoarseDropout(
                           #         (0.03, 0.15), size_percent=(0.02, 0.05),
                           #         per_channel=0.2
                           #     ),
                           # ]),

                           # #5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                           # iaa.Invert(0.05, per_channel=True),

                           # 每个像素随机加减-10到10之间的数
                           # iaa.Add((-10, 10), per_channel=0.5),

                           # 像素乘上0.5或者1.5之间的数字
                           # iaa.Multiply((0.8, 1.2), per_channel=0.5),

                           # 将整个图像的对比度变为原来的一半或者二倍
                           # iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                           # 将RGB变成灰度图然后乘alpha加在原图上
                           # iaa.Grayscale(alpha=(0.0, 1.0)),

                           # 把像素移动到周围
                           # sometimes(
                           #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           # ),

                           # 扭曲图像的局部区域
                           # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                       ],

                       random_order=True  # 随机的顺序把这些操作用在图像上
                       )
        ],
        random_order=True  # 随机的顺序把这些操作用在图像上
    )

    images_aug = seq.augment_images(images)  # 应用数据增强

    # c = 1
    # for each in images_aug:
    #     cv2.imwrite('E:/AI_wenan/third-finetune/supplement2/mc-cde/%s%s.jpg' % (name, c), each)  # 增强图片保存到指定路径
    #     c += 1
    # ia.imshow(np.hstack(images_aug))# 显示增强图片

    for each in images_aug:
        cv2.imwrite('D:/PS_dataset/newpsdatav1/pstype3/output1/%s.jpg' % name, each)
