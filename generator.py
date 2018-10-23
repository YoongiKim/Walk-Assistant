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


import glob
import random
from data_loader import DataLoader
import numpy as np
from keras.utils import to_categorical
import cv2


class Generator:
    def __init__(self, data_dir, tile_row=9, tile_col=16, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.files, self.labels = self.read_label_file('{}/annotation.txt'.format(data_dir), h=tile_row, w=tile_col)

    def read_label_file(self, path, h, w):
        with open(path, 'r') as f:
            lines = f.readlines()

        files = []
        labels = []

        for line in lines:
            line = line.replace('\n', '')
            file_name, label_encode = line.split(',')

            label = []
            for b in label_encode:
                label.append(int(b))

            files.append('{}/{}'.format(self.data_dir, file_name))
            labels.append(np.array(label).reshape((h, w)))

        return np.array(files), np.array(labels)

    def get_XY(self, indexes):
        X = []
        Y = []

        for index in indexes:
            X.append(DataLoader.read_image(self.files[index]))
            Y.append(self.labels[index])

        return np.array(X, dtype=np.float32)/255.0, to_categorical(np.array(Y, dtype=np.float32), 2)

    def generator(self):
        while True:
            pos = 0
            indexes = [i for i in range(len(self.files))]
            random.shuffle(indexes)

            while pos + self.batch_size <= len(self.files):
                X, Y = self.get_XY(indexes[pos:pos + self.batch_size])
                pos += self.batch_size

                yield (X, Y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gen = Generator('D:/Walk-Assistant/frames', batch_size=1)

    for x, y in gen.generator():
        print(x)
        plt.imshow(x[0])
        plt.show()
        print(y)
        input()