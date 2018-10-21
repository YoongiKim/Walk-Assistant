import cv2
from data_loader import DataLoader
import glob
import numpy as np
import random
import os
from model import MyModel

PATH = "H:/Workspaces/Walk-Assistant/data/frames"
ANNOTATION_FILE = "H:/Workspaces/Walk-Assistant/data/frames/annotation.txt"
KERNEL = 80
SHAPE = (9, 16)
HEIGHT = 720
WIDTH = 1280

labels = []

grid = np.zeros(SHAPE, dtype=np.int)
l_down = False
r_down = False

def click(event, x, y, flags, param):
    global clicks, l_down, r_down

    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and l_down):
        l_down = True

        x = max(0, min(x, WIDTH-1))
        y = max(0, min(y, HEIGHT-1))

        r = y // KERNEL
        c = x // KERNEL

        grid[r][c] = 1

    elif event == cv2.EVENT_LBUTTONUP:
        l_down = False

    if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and r_down):
        r_down = True

        r = y // KERNEL
        c = x // KERNEL

        grid[r][c] = 0

    elif event == cv2.EVENT_RBUTTONUP:
        r_down = False

def visualize(img):
    global grid

    grid = np.round(grid)

    h, w, c = img.shape
    box = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            box[i*KERNEL : (i+1)*KERNEL, j*KERNEL : (j+1)*KERNEL] = grid[i][j] * 255

    box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
    box[:,:,0]=0
    box[:,:,2]=0

    visual = cv2.add(img, box)

    for i in range(0, HEIGHT, KERNEL):  # 가로선
        cv2.line(visual, (0, i), (WIDTH, i), color=(255,255,255))
    for i in range(0, WIDTH, KERNEL):  # 세로선
        cv2.line(visual, (i, 0), (i, HEIGHT), color=(255,255,255))

    return visual

def read_label():
    files = []

    if not os.path.exists(ANNOTATION_FILE):
        return [], []

    with open(ANNOTATION_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = str(line).replace('\n', '')
            file, encode = line.split(',')
            files.append(file)

    return files, lines

def write_label(file_name, grid):
    grid = np.round(grid).astype(np.uint8)
    flat = np.array(grid).flatten()
    encode = ''
    label = ''
    for b in flat:
        encode += str(b)
        label = '%s,%s\n' % (file_name, encode)

    print(label)
    labels.append(label)

    with open(ANNOTATION_FILE, 'w') as f:
        f.writelines(labels)
        print('Saved label file')

def predict(img, my_model):
    img = np.array([img])/255.0
    res = my_model.model.predict(img)[0]
    res = np.squeeze(res[:,:,1])  # (9, 16)
    return res

if __name__ == '__main__':
    model = MyModel(True, HEIGHT, WIDTH, KERNEL, KERNEL, 1e-6, 'main')

    files = glob.glob('{}/*.jpg'.format(PATH))

    prev_files, prev_lines = read_label()
    labels = prev_lines
    print('Previous annotations: {}'.format(len(prev_files)))
    print(prev_files)

    random.shuffle(files)

    for index, file in enumerate(files):
        file_name = str(file).replace('\\', '/').split('/')[-1]

        if file_name in prev_files:
            continue

        img = cv2.imread(file)

        grid = predict(img, model)

        cv2.namedWindow('img')
        cv2.setMouseCallback('img', click)

        while True:
            cv2.imshow('img', visualize(img))

            k = cv2.waitKey(2)

            if k == -1:  # No Key
                pass
            elif k == 27:  # ESC
                print('Exit')
                exit(0)
            elif k == 32 or k == 13:  # ESC, Space, Enter
                write_label(file_name, grid)
                break
            else:
                print('Key pressed: ', k)

print('Done')
