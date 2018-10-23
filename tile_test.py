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
import matplotlib.pyplot as plt
from model import MyModel


def brighter(img, multiply, max_value):
    for i in range(len(img)):
        for j in range(len(img[i])):
            img[i][j] = min(img[i][j]*multiply, max_value)
    return img


bgr = cv2.imread('data/test1.jpg')
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = np.array(rgb)

X = np.array([rgb / 255.0])

model = MyModel(True, 720, 1280, 80, 80, 1e-6, 'main')
res = model.model.predict(X, 1)[0]
res = np.squeeze(res[:,:,1])
res = brighter(res, 2.0, 1.0)
probs = res.flatten()
print(res)

tiles = []

for i in range(0, 720, 80):
    for j in range(0, 1280, 80):
        tiles.append(rgb[i:i+80, j:j+80])

for index, tile in enumerate(tiles):
    # plt.imshow(tile)
    # plt.show()
    plt.imsave('data/tiles/%d_%.2f.jpg'%(index, probs[index]), tile)
visual = res*255.0
visual = np.array(cv2.resize(visual, (1280, 720))).astype(np.uint8)
visual = cv2.cvtColor(visual, cv2.COLOR_GRAY2BGR)
visual[:,:,0] = 0
visual[:,:,2] = 0
visual = cv2.add(bgr, visual)

cv2.imshow('visual', visual)
cv2.imwrite('data/result_test1.jpg', visual)
cv2.waitKey(0)
