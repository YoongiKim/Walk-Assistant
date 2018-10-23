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
from skimage.io import imread, imsave
import os

class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def get_files_list(pattern):
        files = glob.glob(pattern)
        return files

    @staticmethod
    def read_image(path):
        img = imread(path)  # Read as RGB
        return img

    @staticmethod
    def write_image(img, path):
        imsave(path, img)

    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    loader = DataLoader()

    files = loader.get_files_list('data/bdd100k/drivable_maps/labels/train/*.png')
    print(files)
    print(len(files))

    for i in range(10):
        print(files[i])
        print(loader.read_image(files[i]))
