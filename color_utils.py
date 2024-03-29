# coding:utf-8
import numpy as np
import cv2

# 给标签图上色
def color_predicts(img):

    '''
    给class图上色

    '''
    # img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)
    color = np.ones([img.shape[0], img.shape[1], 3])
    color[img==0] = [0, 0, 0] #其他，黑色，0
    color[img==1] = [255,0, 0]#烤烟，红色，1
    color[img==2] = [0, 255, 0] #玉米，绿色，2
    color[img==3] = [0,0,255] #薏米仁，蓝色，3

    return color
def color_annotation(label_path, output_path):

    '''

    给class图上色

    '''

    img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)

    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img==0] = [0, 0, 0] #其他，黑色，0
    color[img==1] = [255,0, 0]#烤烟，红色，1
    color[img==2] = [0, 255, 0] #玉米，绿色，2
    color[img==3] = [0,0,255] #薏米仁，蓝色，3

    cv2.imwrite(output_path,color)