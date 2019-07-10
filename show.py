import numpy as np
import cv2

def color_predicts():

    '''
    给class图上色

    '''
    img = cv2.imread("image_3_predict.png",cv2.CAP_MODE_GRAY)
    color = np.ones([img.shape[0], img.shape[1], 3])
    color[img==0] = [0, 0, 0] #其他，黑色，0
    color[img==1] = [255,0, 0]#烤烟，红色，1
    color[img==2] = [0, 255, 0] #玉米，绿色，2
    color[img==3] = [0,0,255] #薏米仁，蓝色，3
    cv2.imwrite("color.png",color)
    #return color
color_predicts()