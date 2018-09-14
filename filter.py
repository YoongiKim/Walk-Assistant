import cv2
import numpy as np

class Filter:
    def __init__(self, n_cluster=16):
        self.n_cluster = n_cluster
        self.colors = None

    def filter_sidewalk(self, img):
        # 이미지를 작게 해서 처리속도 향상
        img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_LINEAR)
        img = Filter.blur(img)

        labels = Filter.color_quantization(img, self.n_cluster, 10)
        crop = Filter.roi(labels)

        self.colors = set(crop.flatten())

        main_colors = self.get_main_colors(crop)
        print(main_colors)

        match = Filter.binary_match(labels, main_colors)

        match = Filter.remove_small_objects(match, 2000)

        return match

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
        img2 = np.zeros((output.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255

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
    def roi(img):
        h, w, c = img.shape
        x_center = int(w/2)
        roi_size = 300
        x_start = int(x_center - roi_size/2)
        x_end = int(x_center + roi_size/2)
        crop = img[x_start:x_end, h - 100:h]
        return crop

if __name__ == '__main__':
    img = cv2.imread('data/test.png')
    filter = Filter()
    match = filter.filter_sidewalk(img)
    cv2.imshow('res', match)
    cv2.waitKey(0)