import cv2
import numpy as np

class Filter:
    @staticmethod
    def color_quantization(img, n_cluster, iteration, epsilon=1.0):
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iteration, epsilon)
        ret, label, center = cv2.kmeans(Z, n_cluster, None, criteria, iteration, cv2.KMEANS_PP_CENTERS)

        center = np.uint(center)
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)
        res2 = np.uint8(res2)
        return res2

    @staticmethod
    def blur(img):
        return cv2.bilateralFilter(img, 9, 75, 75)


if __name__ == '__main__':
    img = cv2.imread('data/test.png')
    img = Filter.blur(img)

    result = Filter.color_quantization(img, 16, 10)
    cv2.imshow('res', result)
    cv2.waitKey(0)