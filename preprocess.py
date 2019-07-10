# coding:utf-8
import cv2
import numpy as np
import random

from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
size = 1024
Image.MAX_IMAGE_PIXELS = 100000000000
# 随机窗口采样
def generate_train_dataset(image_num = 500,
                           train_image_path='dataset/train2/images/',
                           train_label_path='dataset/train2/labels/'):
    '''
    该函数用来生成训练集，切图方法为随机切图采样
    :param image_num: 生成样本的个数
    :param train_image_path: 切图保存样本的地址
    :param train_label_path: 切图保存标签的地址
    :return:
    '''

    # 用来记录所有的子图的数目
    g_count = 1

    images_path = ['dataset/origin/image_1.png','dataset/origin/image_2.png']
    labels_path = ['dataset/origin/image_1_label.png','dataset/origin/image_2_label.png']

    # 每张图片生成子图的个数
    image_each = image_num // len(images_path)
    image_path, label_path = [], []
    for i in tqdm(range(len(images_path))):
        count = 0
        image = Image.open(images_path[i])  # 注意修改img路径
        image = np.asarray(image)
        label = Image.open(labels_path[i])  # 注意修改label路径
        label = np.asarray(label)
        X_height, X_width = image.shape[0], image.shape[1]
        while count < image_each:
            random_width = random.randint(0, X_width - size - 1)
            random_height = random.randint(0, X_height - size - 1)

            image_ogi = image[random_height: random_height + size, random_width: random_width + size,:]
            if 255 not in image_ogi[:,:,-1]:
                continue
            image_ogi = image_ogi[:,:,0:3]
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]
            if 1 not in label_ogi and 2 not in label_ogi and 3 not in label_ogi:
                continue
            image_d, label_d = data_augment(image_ogi, label_ogi)

            image_path.append(train_image_path+'%05d.png' % g_count)
            label_path.append(train_label_path+'%05d.png' % g_count)
            cv2.imwrite((train_image_path+'%05d.png' % g_count), image_d)
            cv2.imwrite((train_label_path+'%05d.png' % g_count), label_d)
            count += 1
            g_count += 1
    df = pd.DataFrame({'image':image_path, 'label':label_path})
    df.to_csv('dataset/path_list2.csv', index=False)

# 以下函数都是一些数据增强的函数
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(size)]

    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)

    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)

    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((size /2, size / 2), angle, 1)

    xb = cv2.warpAffine(xb, M_rotate, (size, size))

    yb = cv2.warpAffine(yb, M_rotate, (size, size))

    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))

    return img


def add_noise(img):
    for i in range(size):  # 添加点噪声

        temp_x = np.random.randint(0, img.shape[0])

        temp_y = np.random.randint(0, img.shape[1])

        img[temp_x][temp_y] = 255

    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转

        yb = cv2.flip(yb, 1)

    # if np.random.random() < 0.25:
    #     xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    # 双边过滤
    if np.random.random() < 0.25:
        xb =cv2.bilateralFilter(xb,9,75,75)

    #  高斯滤波
    if np.random.random() < 0.25:
        xb = cv2.GaussianBlur(xb,(5,5),1.5)

    # if np.random.random() < 0.2:
    #     xb = add_noise(xb)

    return xb, yb

if __name__ == '__main__':
    if not os.path.exists('dataset/train2/images'): os.mkdir('dataset/train2/images')
    if not os.path.exists('dataset/train2/labels'): os.mkdir('dataset/train2/labels')
    generate_train_dataset()
