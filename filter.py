import cv2
import numpy as np

class Filter:
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
    def roi(img):
        h, w, c = img.shape
        x_center = int(w/2)
        roi_size = 300
        x_start = int(x_center - roi_size/2)
        x_end = int(x_center + roi_size/2)
        crop = labels[x_start:x_end, h - 100:h]
        return crop


if __name__ == '__main__':
    img = cv2.imread('data/test.png')
    img = cv2.resize(img, (1280, 720))
    img = Filter.blur(img)

    labels = Filter.color_quantization(img, 32, 10)
    crop = Filter.roi(labels)
    colors = np.array(crop).flatten()
    set_colors = set(colors)
    print(set_colors)
    print(len(set_colors))

    match = np.zeros(labels.shape, dtype=np.uint8)
    for row in range(len(labels)):
        for col in range(len(labels[row])):
            if labels[row][col][0] in set_colors:
                match[row][col] = 255


    cv2.imshow('res', match)
    cv2.waitKey(0)