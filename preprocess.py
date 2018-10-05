from data_loader import DataLoader
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings

BDD100K_PATH = 'data/bdd100k'
DRIVABLE_PATH = BDD100K_PATH + '/drivable_maps/labels/train'
IMAGE_PATH = BDD100K_PATH + '/images/100k/train'
OUTPUT_PATH = 'data/preprocess'
BOX_SIZE = 80

def get_id_from_path(path):
    file_name = path.replace('\\', '/').split('/')[-1].replace('.png', '').replace('.jpg', '')
    image_id = file_name.replace('_drivable_id', '')
    return image_id

def extract_labels(files):
    """
    :param files: list of png file paths
    :return: original_image, labels(0:not_road, 1:main_road, 2:alternative_road)
    """
    images = []
    labels = []

    for file in files:
        print(file)
        img = DataLoader.read_image(file)
        resize = cv2.resize(img, (int(1280/BOX_SIZE), int(720/BOX_SIZE)))
        label = np.array(resize).astype(np.uint8)

        image_id = get_id_from_path(file)

        matching_image = DataLoader.read_image('{}/{}.jpg'.format(IMAGE_PATH, image_id))

        images.append(matching_image)
        labels.append(label)

    return np.array(images), np.array(labels)

def tile(imgs, patch_size=BOX_SIZE, stride=BOX_SIZE):
    if np.ndim(imgs) != 4:
        raise ValueError('Input must be (batch, height, width, channels).')

    batch = []

    for img in imgs:
        h, w, c = img.shape
        x = 0; y = 0

        patches = []
        rows = 0; cols = 0

        while (y + patch_size <= h):
            x=0
            cols=0

            while (x + patch_size <= w):
                patches.append(img[y:y+patch_size, x:x+patch_size])
                x+=stride
                cols+=1
            y+=stride
            rows+=1

        tiles = np.reshape(patches, (rows, cols, patch_size, patch_size, 3))
        batch.append(tiles)

    return np.array(batch)

def make_data(files):
    imgs, labels = extract_labels(files)
    tiles = tile(imgs)

    DataLoader.mkdir(OUTPUT_PATH + '/0')
    DataLoader.mkdir(OUTPUT_PATH + '/1')
    DataLoader.mkdir(OUTPUT_PATH + '/2')

    batches, rows, cols, size, size, chaneel = tiles.shape

    for b in range(batches):
        for r in range(rows):
            for c in range(cols):
                label = labels[b][r][c]
                DataLoader.write_image(tiles[b][r][c], '{}/{}/{}_{}_{}.jpg'.format(OUTPUT_PATH, int(label), get_id_from_path(files[b]), r, c))

if __name__ == '__main__':
    drivable_maps_files = DataLoader.get_files_list(DRIVABLE_PATH + '/*.png')
    print('%d Files Found' % len(drivable_maps_files))

    warnings.filterwarnings('ignore')

    pool = Pool(8)

    batches = [drivable_maps_files[i:i + 10] for i in range(0, len(drivable_maps_files), 10)]

    for _ in tqdm(pool.imap_unordered(make_data, batches), total=len(batches)):
        pass

    pool.close()
    pool.join()