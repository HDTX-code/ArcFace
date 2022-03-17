# opencv实现图像旋转实例
import cv2
import random
import matplotlib.pylab as plt

# 定义旋转函数
import numpy as np


def ImageRotate(image):
    image = cv2.resize(image, (224, 224))
    height, width = image.shape[:2]  # 输入(H,W,C)，取 H，W 的zhi
    center = (width / 2, height / 2)  # 绕图片中心进行旋转
    angle = random.randint(-180, 180)  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
    scale = 0.8  # 将图像缩放为80%

    # 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 进行仿射变换，边界填充为255，即白色，默认为黑色
    image_rotation = cv2.warpAffine(src=image, M=M, dsize=(height, width), borderValue=(255, 255, 255))

    return image_rotation


def ImageNew(src):
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    result = usm
    return result


def Image_GaussianBlur(img):
    kernel_size = (5, 5)
    sigma = 1.5
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    return img


if __name__ == '__main__':
    image = cv2.imread(r"D:\project\humpWhale\data\humpback-whale-identification\train\00aae723d.jpg")
    image_new = ImageNew(image)
    image_gaussianBlur = Image_GaussianBlur(image)
    image_rotation = ImageRotate(image_new)

    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.imshow(image_rotation)
    cv2.imshow("image", image_rotation)
    cv2.imshow("image2", image_new)
    cv2.imshow("image2=3", image_gaussianBlur)
    cv2.waitKey()
