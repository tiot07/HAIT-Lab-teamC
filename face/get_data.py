#!/usr/bin/env python
import os
import sys
import cv2
import time
from datetime import datetime as dt


if len(sys.argv) < 2:
    print("引数に名前を入れて下さい。")
    print("例: python3 get_train_data.py yusuke")
    exit()

#名前指定
name = sys.argv[-1]
print(name)

#Haar-like特徴量読み込み
cascade_file = "haarcascade_frontalface_alt2.xml"
cascade = cv2.CascadeClassifier(cascade_file)

#
img_num = 20
cnt = 0
faces = []

#カメラ
DEVICE_ID = 0
cap = cv2.VideoCapture(DEVICE_ID)


while cnt <= img_num:
    end_flag, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

    for (pos_x, pos_y, w, h) in face_list:
        img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]
        img_face = cv2.resize(img_face, (100, 100))
        faces.append(img_face)

    if len(face_list) > 0:
        print("\r", "画像取得中  {}/{}".format(cnt,img_num), end="")
        cnt += 1

    time.sleep(1)
cap.release()

t = dt.now().strftime('%Y%m%d%H%M%S')


#学習用データ保存
num = 0
path='faces/train/{}'.format(name)
try:
    os.makedirs(path)
except FileExistsError:
    pass

for face in faces[:-1]:
    filename = '{}-{}-{}.jpg'.format(name,t,num)
    cv2.imwrite('{}/{}'.format(path, filename), face)
    num += 1


#テスト用データ保存
path = 'faces/test/{}'.format(name)
try:
    os.makedirs(path)
except FileExistsError:
    pass

face = faces[-1]
filename = '{}-{}-{}.jpg'.format(name, t,num)
cv2.imwrite('{}/{}'.format(path, filename), face)
