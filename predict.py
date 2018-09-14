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

HEIGHT = 288
WIDTH = 512

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='input video path')
parser.add_argument('--out', type=str, default='output/output.avi', help='output video save path')
parser.add_argument('--show', type=str, default='False', help='show real time video')
parser.add_argument('--skip', type=int, default=1, help='skip frame to speed up')
args = parser.parse_args()

SHOW = True if str(args.show).upper() == 'TRUE' else False

weight_files = glob.glob('models/weight*.h5')
last_file = max(weight_files, key=os.path.getctime)
file_name = last_file.split('/')[-1]
print('Starting from {}'.format(file_name))

with CustomObjectScope({'relu6': tf.nn.relu6, 'DepthwiseConv2D': keras.layers.DepthwiseConv2D, 'tf': tf}):
    with open('models/model.json', 'r') as f:
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
        img = np.array([img])
        
        result = model.predict(img)[0]  # (batch, 8, 15, 2)
        result = result[:, :, 1]  # (8, 15, 1)
        result *= 255  # to image 8 bit scale
        result = result.astype(np.uint8)
        _, result = cv2.threshold(result, 200, 255, cv2.THRESH_TOZERO)
        result = Filter.remove_small_objects(result, 5)
        
        vis = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # to white color
        vis[:, :, 0] = 0  # remove blue channel
        vis[:, :, 2] = 0  # remove red channel
        vis = cv2.resize(vis, (1280, 720), interpolation=cv2.INTER_LINEAR)  # resize 15x8 to 1280x720
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
