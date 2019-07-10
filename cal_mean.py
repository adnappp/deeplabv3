import cv2
import os
import numpy as np
R_sum =0
G_sum =0
B_sum =0
path = "dataset/train/images/"
images = os.listdir(path)
per_image_Rmean = []
per_image_Gmean = []
per_image_Bmean = []
for image in images:
    img = cv2.imread(os.path.join(path,image))
    per_image_Bmean.append(np.mean(img[:, :, 0]))
    per_image_Gmean.append(np.mean(img[:, :, 1]))
    per_image_Rmean.append(np.mean(img[:, :, 2]))
R_mean = np.mean(per_image_Rmean)
G_mean = np.mean(per_image_Gmean)
B_mean = np.mean(per_image_Bmean)
print(R_mean)
print(G_mean)
print(B_mean)