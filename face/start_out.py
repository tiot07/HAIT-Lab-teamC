#!/usr/bin/env python

import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime as dt
from keras.models import model_from_json
import h5py
import requests

# 設定
TOKEN=''
CHANNEL=''
TEXT='test'
USERNAME='test_username'
URL='https://slack.com/api/chat.postMessage'

# post
def slack(name):
    TEXT="{}さんが退出しました".format(name)
    post_json = {
        'token': TOKEN,
        'text': TEXT,
        'channel': CHANNEL,
        'username': USERNAME,
        'link_names': 1
    }
    requests.post(URL, data = post_json)

status=1 #0:macのカメラ 1:ネットワークカメラ

ip = sys.argv[-1]
print(ip)

#モデル読み込み
model = model_from_json(open('./model_tf/face-model.json').read())
model.load_weights('./model_tf/face-model.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.summary()

#正解ラベル読み込み
path = "./faces/train"
dirs = os.listdir(path)
dirs = [f for f in dirs if os.path.isdir(os.path.join(path, f))]
label_dict = {}
names = dirs


if __name__ == '__main__':

    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # ウィンドウの準備
    WINDOW_NAME = "out"
    cv2.namedWindow(WINDOW_NAME)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_size = 1.0


    #カメラ指定
    if status==0:
      DEVICE_ID = 0
      cap = cv2.VideoCapture(DEVICE_ID)
    elif status==1:
      img_url = 'http://{}/?action=snapshot'.format(ip)

    # 変換処理ループ
    # 画像取得
    i=0
    tmp="init"
    while True:
        
        if status==0:
          end_flag, img = cap.read()
          img = cv2.resize(img, (640, 360))
        else:
          req = requests.get(img_url)
          img_buf = np.fromstring(req.content, dtype='uint8')
          img = cv2.imdecode(img_buf, 1)


        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

        for (pos_x, pos_y, w, h) in face_list:
            img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]
            img_face = cv2.resize(img_face, (32, 32))

            test_images = []
            test_images.append(img_face / 255.0)
            test_images = np.asarray(test_images)

            results = model.predict(test_images)

            #text = names[np.argmax(results[0])]+"  "+str(np.max(results[0]))+"   "+str(i)
            text = names[np.argmax(results[0])]+"   "+str(i)
            name=names[np.argmax(results[0])]
            if name==tmp and np.max(results[0])>0.8:
              i+=1
            elif name!=tmp:
              i=0
              
            tmp=name
            # 10フレーム連続で同一人物
            if i>10:
              t=dt.now().strftime('%Y%m%d%H%M%S')
              try:
                df = pd.read_csv('../face_log.csv', header=None)
                if (df.tail(1).iloc[0][0]==names[np.argmax(results[0])] and df.tail(1).iloc[0][2]=='out'):
                  pass
                else:
                  df=df.append([[names[np.argmax(results[0])],t,"out","{}-{}.jpg".format(name,t)]])
              except FileNotFoundError:
                df=pd.DataFrame(data=[[names[np.argmax(results[0])],t,"out","{}-{}.jpg".format(name,t)]])
              
              #slack(name)
              df.to_csv('../face_log.csv', header=False, index=False)
              cv2.imwrite("../static/images/{}-{}.jpg".format(name,t), img)
              i=0


            color = (0, 0, 225) #赤
            pen_w = 2
            cv2.putText(img,text,(pos_x,pos_y - 10), font, font_size, color)
            cv2.rectangle(img, (pos_x, pos_y), (pos_x+w, pos_y+h), color, thickness = pen_w)


        # フレーム表示
        cv2.imshow(WINDOW_NAME, img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

    cv2.destroyAllWindows()

