import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import MyModel

img = cv2.imread('data/test1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img)

X = np.array([img / 255.0])

model = MyModel(True, 720, 1280, 80, 80, 1e-6, 'main')
res = model.model.predict(X, 1)[0]*2.0
res = np.squeeze(res[:,:,1]).flatten()
print(res)

tiles = []

for i in range(0, 720, 80):
    for j in range(0, 1280, 80):
        tiles.append(img[i:i+80, j:j+80])

for index, tile in enumerate(tiles):
    # plt.imshow(tile)
    # plt.show()
    plt.imsave('data/tiles/%d_%.2f.jpg'%(index, min(res[index], 1.0)), tile)
