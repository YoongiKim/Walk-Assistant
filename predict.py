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

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import CustomObjectScope
import matplotlib.pyplot as plt
import glob
import os
from keras.models import Model, model_from_json
import argparse
from tqdm import tqdm
from filter import Filter

HEIGHT = 720
WIDTH = 1280

MODEL_NAME = 'main'

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='input video path')
parser.add_argument('--out', type=str, default='output/output.avi', help='output video save path')
parser.add_argument('--show', type=str, default='True', help='show real time video')
parser.add_argument('--skip', type=int, default=1, help='skip frame to speed up')
parser.add_argument('--filter', type=str, default='False', help='Filter sparse area')
parser.add_argument('--model_name', type=str, default='main', help='model directory name under models.')
args = parser.parse_args()

SHOW = True if str(args.show).upper() == 'TRUE' else False
print('show=',SHOW)
FILTER = True if str(args.filter).upper() == 'TRUE' else False
print('filter=',FILTER)
MODEL_NAME = args.model_name

weight_files = glob.glob('models/{}/weight*.h5'.format(MODEL_NAME))
last_file = max(weight_files, key=os.path.getctime)
file_name = last_file.split('/')[-1]
print('Starting from {}'.format(file_name))

with CustomObjectScope({'relu6': tf.nn.relu6, 'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'tf': tf}):
    with open('models/{}/model.json'.format(MODEL_NAME), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(last_file)
print('Loaded Model')

vidcap = cv2.VideoCapture(args.video)
total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / (args.skip+1))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(args.out, fourcc, int(60 / (args.skip + 1)), (1280, 720))

success, image = vidcap.read()
for i in tqdm(range(0, total)):
    if success:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = np.array([img])/255.0
        
        res = model.predict(img)[0]  # (batch, 8, 15, 2)
        res = res[:, :, 1]  # (8, 15, 1)
        res *= 255  # to image 8 bit scale
        res = res.astype(np.uint8)
        if FILTER:
            _, res = cv2.threshold(res, 128, 255, cv2.THRESH_TOZERO)
            res = Filter.remove_small_objects(res, 3)
        
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)  # to white color
        res[:, :, 0] = 0  # remove blue channel
        res[:, :, 2] = 0  # remove red channel
        res = cv2.resize(res, (1280, 720), interpolation=cv2.INTER_LINEAR)  # resize 15x8 to 1280x720
        org = cv2.resize(image, (1280, 720))
        added = cv2.add(org, res)  # combine input, output

        out.write(added)  # save video frame
        if SHOW:
            cv2.imshow('result', added)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                exit(1)

        for j in range(args.skip+1):
            success, image = vidcap.read()

print('Saved to {}'.format(args.out))
