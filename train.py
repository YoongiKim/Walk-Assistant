import tensorflow as tf
from model import MyModel
import glob
from generator import Generator
from keras.callbacks import TensorBoard, ModelCheckpoint

TRAIN = "H:/Workspaces/Walk-Assistant/data/frames"

BATCH_SIZE = 1

HEIGHT = 720
WIDTH = 1280
KERNEL = 80
STRIDE = 80

TILE_ROW = int((HEIGHT-KERNEL)/STRIDE+1)
TILE_COL = int((WIDTH-KERNEL)/STRIDE+1)

print('Load Model? (y/n)')
answer = input()

load = False

if answer == 'y':
    load = True
    print('Loading model...')
else:
    print('Building new model...')

my_model = MyModel(load, HEIGHT, WIDTH, KERNEL, STRIDE, lr=1e-3, model_name='main')

gen = Generator(TRAIN, tile_row=TILE_ROW, tile_col=TILE_COL, batch_size=BATCH_SIZE)

checkpoint_path = 'models/main/model.{epoch:02d}-{acc:.2f}.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='acc', save_best_only=False, mode='auto', save_weights_only=False)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)

my_model.model.fit_generator(gen.generator(), steps_per_epoch=len(gen.files)//BATCH_SIZE, epochs=10000, callbacks=[checkpoint, tb], initial_epoch=my_model.epoch)
