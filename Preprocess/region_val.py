import cv2 as cv2
import numpy as np
import os

abs_path = os.path.abspath('../Data/Val_P/Faces/')

videos = os.listdir(abs_path)

reg_dir = '..\Data\Val_Region'

try:
    os.mkdir(reg_dir)
except Exception as e:
    print(str(e))


for v in videos:
    images_path = abs_path + '/' + v 
    imgs = os.listdir(images_path)
    img = imgs[int(len(imgs)/2)]
    img_path =  images_path + '/' + img
    
    new_dir = reg_dir + '/' + v

    try:
        os.mkdir(new_dir)
    except Exception as e:
        print(str(e))

    img = cv2.imread(img_path)
    copy = cv2.resize(img, (224,224))
    cv2.imwrite(new_dir + '/copy.jpg',copy)

