"""
Walk-Assistant : Recognizing sidewalk for the visually impaired
Copyright (C) 2018 Yoongi Kim (devlifecode@outlook.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import tensorflow as tf
from keras import Sequential
from keras import layers
from keras.layers import TimeDistributed, Reshape, Conv2D, MaxPool2D, Lambda, Input, Dense, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, DepthwiseConv2D, Bidirectional
from keras.optimizers import Adam
from keras import Model
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.utils import to_categorical
from keras import backend as K
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import mobilenet_v2
import random
from keras.utils.generic_utils import CustomObjectScope
import keras
import glob
import os
from keras.models import Model, model_from_json
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import EarlyStopping
import argparse
from tqdm import tqdm
from filter import Filter


class MyModel:
    def __init__(self, load, height, width, kernel, stride):
        self.load = load
        self.height = height
        self.width = width
        self.epoch = 0
        self.loss = 0
        self.acc = 0

        if self.load:
            self.model = self.load_model()
        else:
            self.model = self.make_model(kernel=kernel, stride=stride)

        opt = Adam(lr=0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

        with open('models/model.json', 'w') as f:
            f.write(self.model.to_json())
            print('Saved Model Structure')

    def train_on_batch(self, X, Y):
        Y = self.to_categorical_custom(Y)
        self.loss, self.acc = self.model.train_on_batch(X, Y)

    def save_weights(self):
        self.model.save_weights('models/weight.h5')
        print('Saved model successfully')

    def load_model(self):
        weight_files = glob.glob('models/weight.*.h5')
        last_file = max(weight_files, key=os.path.getctime)

        file_name = last_file.split('/')[-1]
        epoch_acc = file_name.split('.')[1]
        epoch = epoch_acc.split('-')[0]
        self.epoch = int(epoch)

        print('Starting from {} - epoch: {}'.format(file_name, self.epoch))

        with CustomObjectScope({'relu6': tf.nn.relu6, 'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'tf': tf}):
            with open('models/model.json', 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(last_file)
        print('Loaded Model')

        return model

    def make_model(self, kernel=120, stride=80):
        i = Input(batch_shape=(None, self.height, self.width, 3))
        x = Lambda(lambda x: tf.extract_image_patches(
            x, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID'))(i)

        out_width = int((self.width - kernel)/stride + 1)
        out_height = int((self.height - kernel)/stride + 1)
        print(out_height, out_width)

        x = Reshape([out_height, out_width, kernel, kernel, 3])(x)
        x = Reshape([out_height*out_width, kernel, kernel, 3])(x)

        x = mobilenet_v2.MobileNetv2(x)

        x = TimeDistributed(GlobalAveragePooling2D())(x)
        x = TimeDistributed(Dense(2, activation='softmax'))(x)
        x = Reshape([out_height, out_width, 2])(x)

        model = Model(inputs=[i], outputs=[x])

        model.summary()

        return model

    @staticmethod
    def to_categorical_custom(Y):
        # softmax를 위해 실수 범위에서 one-hot 인코딩
        if len(Y.shape) != 4:
            print('model.py, to_categorical_custom(Y),'
                  ' input shape is wrong. expected 4 dimensions (batch, height, width, channels.')
        batch = Y.shape[0]
        row = Y.shape[1]
        col = Y.shape[2]
        a = np.zeros(shape=(batch, row, col, 2), dtype=np.float32)
        for b in range(batch):
            for i in range(row):
                for j in range(col):
                    a[b][i][j][0] = 1.0-Y[b][i][j]
                    a[b][i][j][1] = Y[b][i][j]

        return np.array(a)

    def prepare_train(self):
        # time = datetime.datetime.now().strftime('%m-%d, %H-%M-%S')
        # path = './logs/{}'.format(time)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        # board = TensorBoard(log_dir='./logs/{}'.format(time), write_images=True, write_graph=True)
        # checkpoint_path = 'models' + '/weight.{epoch:02d}-{acc:.2f}.h5'
        # checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', save_best_only=False, mode='auto')
        pass