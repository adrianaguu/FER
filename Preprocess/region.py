import cv2 as cv2
import numpy as np
import os

abs_path = os.path.abspath('../Data/Train_P/Faces/')

videos = os.listdir(abs_path)

reg_dir = '..\Data\Train_Region'

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
    crop_img = img[0:96,0:96]
    crop_img = cv2.resize(crop_img, (224,224))
    cv2.imwrite(new_dir + './croped1.jpg',crop_img)
    crop_img = img[0:96,32:128]
    crop_img = cv2.resize(crop_img, (224,224))
    cv2.imwrite(new_dir + './croped2.jpg',crop_img)
    crop_img = crop_img = img[32:128,16:112]
    crop_img = cv2.resize(crop_img, (224,224))
    cv2.imwrite(new_dir +'./croped3.jpg',crop_img)
    crop_img = img[6:122,6:122]
    crop_img = cv2.resize(crop_img, (224,224))
    cv2.imwrite(new_dir + './croped4.jpg',crop_img)
    crop_img = img[10:118,10:118]
    crop_img = cv2.resize(crop_img, (224,224))
    cv2.imwrite(new_dir + './croped5.jpg',crop_img)
