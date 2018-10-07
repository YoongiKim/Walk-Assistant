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
import numpy as np
import argparse
from tqdm import tqdm, trange
from filter import Filter
from model import MyModel
from data_loader import DataLoader
from keras.callbacks import TensorBoard, ModelCheckpoint
from road_detector_generator import Generator

HEIGHT = 720
WIDTH = 1280
KERNEL = 80
STRIDE = 80

parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, default='True', help='Load last weight')
parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
args = parser.parse_args()

LOADMODEL = True if str(args.load).upper() == 'TRUE' else False
print('Load model = {}'.format(LOADMODEL))

model = MyModel(LOADMODEL, HEIGHT, WIDTH, KERNEL, STRIDE, 0.01, model_name='road')
model.prepare_train()

rows = int((HEIGHT-KERNEL)/STRIDE+1)
cols = int((WIDTH-KERNEL)/STRIDE+1)

generator = Generator()

if model.epoch >= args.epochs:
    print("Loaded model's epochs is already bigger than desired epochs. Set --epochs {}".format(model.epoch + 1))
    exit(1)


checkpoint_path = 'models/road/weight.{epoch:02d}-{acc:.2f}.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', save_best_only=False, mode='auto')

model.model.fit_generator(generator.generator(),
                          steps_per_epoch=len(generator.files),
                          epochs=args.epochs,
                          callbacks=[checkpoint],
                          initial_epoch=model.epoch)