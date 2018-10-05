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
        img = imread(path)
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
