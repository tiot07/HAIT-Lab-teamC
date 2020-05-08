# 紹介動画
https://www.youtube.com/watch?v=1jzbwTGvBJw&feature=youtu.be  
動作は(5:20~)

# 前準備
## OpenCVをインストール
pip3 install opencv-python

## 画像取得　([name]の部分に自分の名前を記載)
python3 get_train_data.py [name]

## 学習
python3 train.py

## app.pyのallmemberlistに全員の名前を登録

## ストリーミングサーバーに、mjpg-streamerのインストール
https://qiita.com/lobmto/items/c31e0c8136c16f75b1cd  
上記を参考にインストール(Ubuntu,Raspbianにて動作確認)

## ストリーミングサーバーのipアドレス(xx.xx.xx.xx)を調べ、index4.htmlを編集



# 実行
## ストリーミングサーバー起動
./mjpg_streamer -i "./input_uvc.so -d /dev/video0 -r 640x480 -f 10" -o "./output_http.so -w ./www -p 8080"  
./mjpg_streamer -i "./input_uvc.so -d /dev/video1 -r 640x480 -f 10" -o "./output_http.so -w ./www -p 8081"

## 認識プログラム起動
python3 start_in.py xx.xx.xx.xx:8080  
python3 start_out.py xx.xx.xx.xx:8081

## Flask起動
python3 app.py


# その他
## slackに通知
https://qiita.com/9en/items/23eb3762a9df2c29e812  
上記を参考にtokenとCHANNEL名を取得し、start_in.py start_out.pyのL17,18,127を編集

## 既知の不具合
新たに実行する時は、face_log.csvを消去してから起動プログラムを実行してください。  
顔が認識されるとface_log.csvが作成されますが、face_log.csvがない状態で「出席の確認」ページに行くとエラーが表示されます。
