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


# HEIGHT = 288
# WIDTH = 512
# KERNEL = 40
# STRIDE = 40

HEIGHT = 720
WIDTH = 1280
KERNEL = 80
STRIDE = 80

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='data/test.mp4', help='input video')
parser.add_argument('--load', type=str, default='True', help='Load last weight')
parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
parser.add_argument('--show', type=bool, default=False, help='Show filtering task')
parser.add_argument('--init_skip', type=int, default=200, help='Skip frames on start')
parser.add_argument('--skip', type=int, default=1, help='Skip frames per loop')
args = parser.parse_args()

print('Training video: {}, you can set manually "--video PATH"'.format(args.video))
LOADMODEL = True if str(args.load).upper() == 'TRUE' else False
print('Load model = {}'.format(LOADMODEL))

model = MyModel(LOADMODEL, HEIGHT, WIDTH, KERNEL, STRIDE)
model.prepare_train()

zone_h = int((HEIGHT-KERNEL)/STRIDE+1)
zone_w = int((WIDTH-KERNEL)/STRIDE+1)
filter = Filter(n_cluster=32, zone_h=zone_h, zone_w=zone_w)

for i in range(model.epoch+1, args.epochs+1):
    vidcap = cv2.VideoCapture(args.video)
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / (args.skip + 1))
    print('Skipping {} frames'.format(args.init_skip))
    for j in range(args.init_skip):
        success, image = vidcap.read()

    success, image = vidcap.read()

    with trange(0, total-args.init_skip) as t:
        t.set_description('Epoch {}'.format(i))
        for step in t:
            if success:
                t.write('step:{}, loss:{}, acc:{}'.format(step, model.loss, model.acc))
                img = cv2.resize(image, (WIDTH, HEIGHT))
                y = filter.filter_sidewalk(img, args.show)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                model.train_on_batch(np.array([img]), np.array([y]))
                
                if step % 50 == 0:
                    model.save_weights(i)

            for skip in range(0, args.skip+1):
                success, image = vidcap.read()

    model.save_weights(i)
