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

import cv2
import numpy as np

class Filter:
    def __init__(self, n_cluster=16, zone_h=17, zone_w=31):
        self.n_cluster = n_cluster
        self.colors = None
        self.zone_h = zone_h
        self.zone_w = zone_w

    def filter_sidewalk(self, img, show=False, roi_w=100, roi_h=50):
        # 이미지를 작게 해서 처리속도 향상
        img = cv2.resize(img, (480, 270))
        blur = Filter.blur(img)
        
        labels = Filter.color_quantization(blur, self.n_cluster, 1)
        crop = Filter.roi(labels, roi_w, roi_h, img, show=show)
        # cv2.imshow('crop', crop)

        self.colors = set(crop.flatten())

        main_colors = self.get_main_colors(crop)
        # print(main_colors)

        match = Filter.binary_match(labels, main_colors)

        match = Filter.remove_small_objects(match, 3000)
        if show:
            cv2.imshow('match', match)

        activation = cv2.resize(match, (self.zone_w, self.zone_h))
        if show:
            cv2.imshow('result', cv2.resize(activation, (480, 270)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User Interrupted')
            exit(1)
        # cv2.waitKey(0)

        np_arr = np.array(activation).reshape(self.zone_h, self.zone_w, 1)
        np_arr = np_arr / 255.0

        return np_arr

    def get_main_colors(self, img):
        count = [0 for i in range(self.n_cluster)]

        for i in img.flatten():
            count[i] += 1

        avg = np.average(count)

        main_colors = []
        for i in range(len(count)):
            if count[i] > avg:
                main_colors.append(i)

        return main_colors

    @staticmethod
    def remove_small_objects(img, min_size=150):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # your answer image
        img2 = img
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                img2[output == i + 1] = 0

        return img2

    @staticmethod
    def binary_match(img, search_list, mask=255):
        h, w, c = img.shape
        match = np.zeros(img.shape, dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                if img[row][col][0] in search_list:
                    match[row][col] = mask

        return match

    @staticmethod
    def color_quantization(img, n_cluster, iteration, epsilon=1.0):
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iteration, epsilon)
        ret, label, center = cv2.kmeans(Z, n_cluster, None, criteria, iteration, cv2.KMEANS_PP_CENTERS)

        labels = label.reshape((img.shape[0], img.shape[1], 1))
        # center = np.uint(center)
        # visual = center[label.flatten()]
        # visual = visual.reshape(img.shape)
        # visual = np.uint8(visual)

        return labels

    @staticmethod
    def blur(img):
        return cv2.bilateralFilter(img, 9, 75, 75)

    @staticmethod
    def roi(labels, width, height, original_img, show=False):
        h, w, c = labels.shape
        x_center = int(w/2)
        roi_size = width
        x_start = int(x_center - roi_size/2)
        x_end = int(x_center + roi_size/2)
        if show:
            rect = cv2.rectangle(original_img.copy(), (x_start, h - height), (x_end, h), (0, 255, 255), 2)
            cv2.imshow('roi', rect)
        crop = labels[h - height:h, x_start:x_end]
        return crop

if __name__ == '__main__':
    img = cv2.imread('data/test.png')
    filter = Filter()
    activation = filter.filter_sidewalk(img)
    cv2.imshow('result', cv2.resize(activation, (480, 270)))
    cv2.waitKey(0)