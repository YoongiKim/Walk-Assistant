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
from keras.layers import Reshape, Conv2D, MaxPool2D, UpSampling2D, Lambda, Input, Dense, GlobalAveragePooling2D, Flatten
from keras.layers import TimeDistributed as Dist
from keras.layers import BatchNormalization, Activation, DepthwiseConv2D, Bidirectional, CuDNNLSTM, LSTM
from keras.layers.core import Activation, Reshape, Permute
from keras.optimizers import Adam
from keras import Model
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
from keras.callbacks import EarlyStopping
import argparse
from tqdm import tqdm
from filter import Filter
from data_loader import DataLoader
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

Graph = None

class MyModel:
    def __init__(self, load, height, width, kernel, stride, lr, model_name):
        """

        :param load: bool load model
        :param height: 720
        :param width: 1280
        :param kernel: 80
        :param stride: 80
        :param lr: learning rate
        :param model_name: 'main' or 'road'
        """
        self.load = load
        self.height = height
        self.width = width
        self.lr = lr
        self.model_name = model_name
        self.epoch = 0
        self.kernel = kernel
        self.stride = stride

        DataLoader.mkdir('models/{}'.format(self.model_name))

        if self.load:
            self.model = self.load_model()
        else:
            self.model = self.build_simple_model(self.kernel, self.stride)

        opt = Adam(lr=lr)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    def save_model(self, epoch, acc):
        Model(self.model).save('models/{}/model.{}-{:.3f}.h5'.format(self.model_name, int(epoch), acc))
        print('Saved model')

    def load_model(self):
        global Graph  # multiprocess-able

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))

        # model.99-0.98.h5
        files = glob.glob('models/{}/model.*.h5'.format(self.model_name))

        if len(files) == 0:
            print('Trained model not found from "models/{}/model.*.h5"'.format(self.model_name))
            print('Building new model because model file not found...')

            return self.build_simple_model(self.kernel, self.stride)

        last_file = max(files, key=os.path.getctime)

        file_name = last_file.replace('\\', '/').split('/')[-1].replace('model.', '').replace('.h5', '')
        self.epoch = int(file_name.split('-')[0])
        acc = float(file_name.split('-')[1])

        with CustomObjectScope({'relu6': tf.nn.relu6, 'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'tf': tf}):
            model = load_model(last_file)

        Graph = tf.get_default_graph()

        print('Loaded last model - {}, epoch: {}, acc: {}'.format(last_file, self.epoch, acc))

        return model

    def predict(self, X):  # multiprocess-able
        global Graph

        with Graph.as_default():
            Y = self.model.predict(X)

        return Y

    def build_simple_model(self, kernel=80, stride=80):
        i = Input(batch_shape=(None, self.height, self.width, 3))
        x = Lambda(lambda x: tf.extract_image_patches(
            x, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID'))(i)

        out_width = int((self.width - kernel)/stride + 1)
        out_height = int((self.height - kernel)/stride + 1)
        print(out_height, out_width)

        x = Reshape([out_height, out_width, kernel, kernel, 3])(x)
        x = Reshape([out_height*out_width, kernel, kernel, 3])(x)  # [144, 80, 80, 3]

        x = Dist(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))(x)
        x = Dist(BatchNormalization())(x)
        x = Dist(Activation('relu'))(x)
        x = Dist(MaxPool2D())(x)

        x = Dist(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))(x)
        x = Dist(BatchNormalization())(x)
        x = Dist(Activation('relu'))(x)
        x = Dist(MaxPool2D())(x)

        x = Dist(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))(x)
        x = Dist(BatchNormalization())(x)
        x = Dist(Activation('relu'))(x)
        x = Dist(MaxPool2D())(x)

        x = Dist(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))(x)
        x = Dist(BatchNormalization())(x)
        x = Dist(Activation('relu'))(x)

        x = Dist(Flatten())(x)
        x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)
        x = Dist(Dense(2, activation='softmax'))(x)
        x = Reshape([out_height, out_width, 2])(x)

        model = Model(inputs=[i], outputs=[x])

        model.summary()

        return model

    def build_model(self, kernel=80, stride=80):
        i = Input(batch_shape=(None, self.height, self.width, 3))
        x = Lambda(lambda x: tf.extract_image_patches(
            x, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID'))(i)

        out_width = int((self.width - kernel) / stride + 1)
        out_height = int((self.height - kernel) / stride + 1)
        print(out_height, out_width)

        x = Reshape([out_height, out_width, kernel, kernel, 3])(x)
        x = Reshape([out_height * out_width, kernel, kernel, 3])(x)

        x = mobilenet_v2.MobileNetv2(x)

        x = Dist(GlobalAveragePooling2D())(x)
        x = Dist(Dense(2, activation='softmax'))(x)
        x = Reshape([out_height, out_width, 2])(x)

        model = Model(inputs=[i], outputs=[x])

        model.summary()

        return model
