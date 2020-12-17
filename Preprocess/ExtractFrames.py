import cv2
import os
from os import listdir
from face_alignment import face_alignment, save_img
from helper import *
import numpy as np

abs_path = os.path.abspath('../Data/Train_NP')
paths=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

for path in paths:
  videos_path = abs_path + "/" + path
  files = listdir(videos_path)
  new_dir = '../Data/Train/'+path
  try:
    os.mkdir(new_dir)
  except Exception as e:
    print(str(e))

  for f in files:
  
    video_path = videos_path + "/" + f
    folder_name = f.split('.')[0]
    new_dir_f = new_dir+'/'+folder_name
    try:
      os.mkdir(new_dir_f)
    except Exception as e:
      print(str(e))

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    count = 0
    while success:
      # Allign face
      image = face_alignment(image)

      copy_image = image.copy()

      # Convert the image to RGB colorspace
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Convert the image to gray 
      gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

      # Detect faces in the image using pre-Trained face dectector
      faces = face_cascade.detectMultiScale(gray_image, 1.25, 6)

      print('Number of faces detected:', len(faces))

      # Get the bounding box for each detected face
      for f in faces:
        try:
          x, y, w, h = [ v for v in f ]
          perX25 = int((x+w)*0.25)
          perY25 = int((y+h)*0.25)
          face_crop = copy_image[y-perY25:y+h+perY25, x-perX25:x+w+perX25]
          face_crop = cv2.resize(face_crop, (224,224))
          save_img('../Data/Train/%s/%s/cropedframe%d.jpg'%(path,folder_name,count),face_crop)
        except Exception as e:
          print(str(e))
      
      success,image = vidcap.read()
      count += 1




