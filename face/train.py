#!/usr/bin/env python

import os
import random
import cv2
import numpy as np
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.callbacks

path = "./faces/train"
dirs = os.listdir(path)
dirs = [f for f in dirs if os.path.isdir(os.path.join(path, f))]

label_dict = {}
i = 0

for dirname in dirs:
    label_dict[dirname] = i
    i += 1

def load_data(data_type):

    filenames, images, labels = [], [], []
    walk = filter(lambda _: not len(_[1]) and data_type in _[0], os.walk('faces'))

    for root, dirs, files in walk:
        filenames += ['{}/{}'.format(root, _) for _ in files if not _.startswith('.')]

    # シャッフル
    random.shuffle(filenames)

    # Read, resize, and reshape images
    images = []
    for file in filenames:
        img = cv2.imread(file)
        img = cv2.resize(img, (32,32))
        images.append(img.astype(np.float32) / 255.0)
    images = np.asarray(images)

    for filename in filenames:
        label = np.zeros(len(label_dict))
        for k, v in label_dict.items():
           if k in filename:
                label[v] = 1.
        labels.append(label)
    labels = np.asarray(labels)
    print(labels,12)

    return images, labels

def make_model(train_shape, label_shape):

    model = Sequential()

    # model.add(Conv2D(32, (3, 3), padding="same", input_shape=train_shape[1:]))
    # model.add(Activation('relu'))
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=train_shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(label_shape[1]))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    return model

accuracys = []
tb_cb = keras.callbacks.TensorBoard(log_dir="tflog/", histogram_freq=1)
cbks = [tb_cb]
train_images, train_labels = load_data('train')
test_images, test_labels = load_data('test')

print("train_images", len(train_images))
print("test_images", len(test_images))

print(train_images.shape, train_labels.shape)

model = make_model(train_images.shape, train_labels.shape)

#tensorvboard
model.fit(train_images, train_labels, batch_size=32, epochs=20,callbacks=cbks,validation_data=(test_images, test_labels))

# modelのテスト
score = model.evaluate(test_images, test_labels)
print('loss=', score[0])
print('accuracy=', score[1])

model_json_str = model.to_json()
open('./model_tf/face-model.json', 'w').write(model_json_str)

hdf5_file = "./model_tf/face-model.hdf5"
model.save_weights(hdf5_file)

gc.collect()
