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
print("""
    Walk-Assistant Copyright (C) 2018 Yoongi Kim
    This program comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it
    under certain conditions.
""")

import tensorflow as tf
from keras import Sequential
from keras import layers
from keras.layers import TimeDistributed, Reshape, Conv2D, MaxPool2D, Lambda, Input, Dense, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, DepthwiseConv2D, Bidirectional
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

HEIGHT = 720
WIDTH = 1280

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='data/test.mp4', help='input video')
parser.add_argument('--load', type=bool, default=True, help='Load last weight')
parser.add_argument('--steps', type=int, default=200, help='Desired train steps, skips frames if smaller than video')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
args = parser.parse_args()

LOAD_MODEL = args.load
print('Load model = {}'.format(LOAD_MODEL))

# 시작시 카메라 흔들림 때문에 200프레임을 스킵합니다.
INIT_SKIP_FRAME = 200

print('Training video: {}, you can set manually "--video PATH"'.format(args.video))
vidcap = cv2.VideoCapture(args.video)

total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
skip = int(total / args.steps+1)

print('Total frames: {}, Desired steps: {}, Skip frames: {}, Init skip: {}'.format(total, args.steps, skip, INIT_SKIP_FRAME))
print('Decoding video and loading to memory...')
for i in range(INIT_SKIP_FRAME):
    success, image = vidcap.read()

success, image = vidcap.read()

X = []
for i in tqdm(range(0, int((total - INIT_SKIP_FRAME)/skip))):
    if success:
      img = cv2.resize(image, (WIDTH, HEIGHT))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      X.append(img)
      for j in range(skip+1):
        success, image = vidcap.read()

print('Loaded input images')

X = np.array(X)
print(len(X))


def to_categorical_custom(Y):
    # softmax를 위해 실수 범위에서 one-hot 인코딩
    row = len(Y)
    col = len(Y[-1])
    a = np.zeros(shape=(row, col, 2), dtype=np.float32)
    for i in range(row):
        for j in range(col):
            a[i][j][0] = 1.0-Y[i][j]
            a[i][j][1] = Y[i][j]

    return np.array(a)

y = []
with open('data/safe_zone.csv', newline='') as f:
    rdr = csv.reader(f)
    for row in rdr:
        y.append(row)

y = np.array(y, dtype=np.float32)
print('Safe zone:')
print(y)
y = to_categorical_custom(y)
# print(y[:,:,1])

Y = []
for i in range(len(X)):
    Y.append(y)
Y = np.array(Y)


start_epoch = 0
file_name = "no_file"

if LOAD_MODEL:
    weight_files = glob.glob('models/weight.*.h5')
    last_file = max(weight_files, key=os.path.getctime)

    file_name = last_file.split('/')[-1]
    epoch_acc = file_name.split('.')[1]
    epoch = epoch_acc.split('-')[0]
    start_epoch = int(epoch)

    print('Starting from {} - epoch: {}'.format(file_name, start_epoch))

    with CustomObjectScope({'relu6': tf.nn.relu6, 'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'tf': tf}):
        with open('models/model.json', 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(last_file)
    print('Loaded Model')

else:
    i = Input(batch_shape=(None, HEIGHT, WIDTH, 3))  # (None, 720, 1280, 3)
    x = Lambda(lambda x: tf.extract_image_patches(
        x, ksizes=[1, 120, 120, 1], strides=[1, 80, 80, 1], rates=[1,1,1,1], padding='VALID'))(i)
    # (None, 8, 15, 43200)

    x = Reshape([8, 15, 120, 120, 3])(x)  # (None, 8, 15, 120, 120, 3)
    x = Reshape([8*15, 120, 120, 3])(x)  # (None, 120, 120, 120, 3)

    x = mobilenet_v2.MobileNetv2(x)  # (None, 120, 4, 4, 1280)

    x = TimeDistributed(GlobalAveragePooling2D())(x)  # (None, 120, 1280)
    x = TimeDistributed(Dense(2, activation='softmax'))(x)  # (None, 120, 2)
    x = Reshape([8, 15, 2])(x)  # (None, 8, 15, 2)

    model = Model(inputs=[i], outputs=[x])

    model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

with open('models/model.json', 'w') as f:
    f.write(model.to_json())
    print('Saved Model Structure')

time = datetime.datetime.now().strftime('%m-%d, %H-%M-%S')
path = './logs/{}'.format(time)
if not os.path.exists(path):
    os.makedirs(path)


print("Start train? (y/n)")
answer = input()

if answer != 'y':
    exit(0)

print('Start Training - {}'.format(time))
print('Starting from {} - epoch: {}'.format(file_name, start_epoch))

board = TensorBoard(log_dir='./logs/{}'.format(time), write_images=True, write_graph=True)
checkpoint_path = 'models' + '/weight.{epoch:02d}-{acc:.2f}.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', save_best_only=False, mode='auto')
earlystop = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='auto')

for epoch in range(args.epochs):
    random.shuffle(X)
    #  actual batch size = batch_size * 144
    model.fit(X, Y, epochs=start_epoch+epoch+1, batch_size=1, callbacks=[earlystop, board, checkpoint], initial_epoch=start_epoch+epoch)

model.save_weights('models/weight.h5')
model.save('model.h5')

print('Saved model successfully')

