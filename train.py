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
from model import MyModel
import glob
from generator import Generator
from keras.callbacks import TensorBoard, ModelCheckpoint

TRAIN = "data/frames"

BATCH_SIZE = 1

HEIGHT = 720
WIDTH = 1280
KERNEL = 80
STRIDE = 80

TILE_ROW = int((HEIGHT-KERNEL)/STRIDE+1)
TILE_COL = int((WIDTH-KERNEL)/STRIDE+1)

print('Load last trained model? (y/n) Default= y')
answer = input()

load = True

if answer == 'n':
    load = False
    print('Building new model...')
else:
    load = True
    print('Loading model...')

my_model = MyModel(load, HEIGHT, WIDTH, KERNEL, STRIDE, lr=1e-3, model_name='main')

gen = Generator(TRAIN, tile_row=TILE_ROW, tile_col=TILE_COL, batch_size=BATCH_SIZE)

checkpoint_path = 'models/main/model.{epoch:02d}-{acc:.2f}.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', save_best_only=False, mode='auto', save_weights_only=False)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)

my_model.model.fit_generator(gen.generator(), steps_per_epoch=len(gen.files)//BATCH_SIZE, epochs=10000, callbacks=[checkpoint, tb], initial_epoch=my_model.epoch)
