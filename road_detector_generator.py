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


from data_loader import DataLoader
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
import random

BDD100K_PATH = 'data/bdd100k'
DRIVABLE_PATH = BDD100K_PATH + '/drivable_maps/labels/train'
IMAGE_PATH = BDD100K_PATH + '/images/100k/train'
OUTPUT_PATH = 'data/preprocess'
BOX_SIZE = 80


class Generator:
    def __init__(self, batch_size=1):
        self.files = DataLoader.get_files_list(DRIVABLE_PATH + '/*.png')
        print('%d Drivable Maps Files Found' % len(self.files))
        self.batch_size = batch_size

    @staticmethod
    def get_id_from_path(path):
        file_name = path.replace('\\', '/').split('/')[-1].replace('.png', '').replace('.jpg', '')
        image_id = file_name.replace('_drivable_id', '')
        return image_id

    @staticmethod
    def extract_labels(files):
        """
        :param files: list of png file paths
        :return: original_image, labels(0:not_road, 1:road)
        """
        images = []
        labels = []

        for file in files:
            # print(file)
            img = DataLoader.read_image(file)
            resize = cv2.resize(img, (int(1280/BOX_SIZE), int(720/BOX_SIZE)))
            label = np.array(resize).astype(np.bool).astype(np.uint8)

            image_id = Generator.get_id_from_path(file)

            matching_image = DataLoader.read_image('{}/{}.jpg'.format(IMAGE_PATH, image_id))

            images.append(matching_image)
            labels.append(label)

        return np.array(images), np.array(labels)

    #@staticmethod
    # def tile(imgs, patch_size=BOX_SIZE, stride=BOX_SIZE):
    #     if np.ndim(imgs) != 4:
    #         raise ValueError('Input must be (batch, height, width, channels).')
    #
    #     batch = []
    #
    #     for img in imgs:
    #         h, w, c = img.shape
    #         x = 0; y = 0
    #
    #         patches = []
    #         rows = 0; cols = 0
    #
    #         while (y + patch_size <= h):
    #             x=0
    #             cols=0
    #
    #             while (x + patch_size <= w):
    #                 patches.append(img[y:y+patch_size, x:x+patch_size])
    #                 x+=stride
    #                 cols+=1
    #             y+=stride
    #             rows+=1
    #
    #         tiles = np.reshape(patches, (rows, cols, patch_size, patch_size, 3))
    #         batch.append(tiles)
    #
    #     return np.array(batch)

    @staticmethod
    def get_XY(files):
        imgs, labels = Generator.extract_labels(files)
        # tiles = tile(imgs)

        # batches, rows, cols, size, size, channel = tiles.shape
        batches, rows, cols = labels.shape

        one_hot = np.zeros((batches, rows, cols, 2), dtype=np.float32)

        for b in range(batches):
            for r in range(rows):
                for c in range(cols):
                    one_hot[b][r][c][1] = labels[b][r][c]
                    one_hot[b][r][c][0] = 1.0 - labels[b][r][c]

        return imgs, one_hot

    def generator(self):
        while True:
            pos = 0
            random.shuffle(self.files)
            while pos+self.batch_size <= len(self.files):
                imgs, labels = Generator.get_XY(self.files[pos:pos+self.batch_size])

                pos += self.batch_size

                yield (imgs, labels)

