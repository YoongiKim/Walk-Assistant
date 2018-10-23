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

class OptFlow:
    def __init__(self, resize_width=320, resize_height=180, height_start=0.2, height_end=0.5):
        self.width = resize_width
        self.height = resize_height

        self.height_start = int(self.height * height_start)
        self.height_end = int(self.height * height_end)

    def get_direction(self, frame1, frame2, show=False):
        frame1 = cv2.resize(frame1, (self.width, self.height))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2, (self.width, self.height))
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame1[self.height_start:self.height_end],
                                            frame2[self.height_start:self.height_end], None, 0.5, 3, 15, 1, 5, 1.2, 0)
        flow_avg = np.median(flow, axis=(0, 1))  # [x, y]

        move_x = -1 * flow_avg[0]
        move_y = -1 * flow_avg[1]

        if show:
            hsv = np.zeros((self.height_end - self.height_start, self.width, 3))
            hsv[...,1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(np.array(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

            cv2.imshow('opt_flow', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                exit(1)

        return move_x, move_y

    @staticmethod
    def draw_arrow(img, x, y, multiply=25):
        h, w, c = img.shape
        arrow = cv2.arrowedLine(img, (int(w / 2), int(h / 2)), (int(w / 2 + x * multiply), int(h / 2 + y * multiply)),
                                color=(0, 255, 255), thickness=15)
        return arrow

    @staticmethod
    def draw_way(img, move_queue, x_multiply=3, y_multiply=4, size=200):
        h, w, c = img.shape
        way = np.zeros((h, w, 1))
        sum_x = float(w/2)
        sum_y = float(h)

        for index, (x, y) in enumerate(move_queue):
            way = cv2.line(way, (int(sum_x), int(sum_y)), (int(sum_x+x), int(sum_y+y)), 255, size)
            sum_x += x * x_multiply
            sum_y += y * y_multiply

        return way


if __name__ == '__main__':
    flow = OptFlow()

    cap = cv2.VideoCapture("C:/Users/Yoongi Kim/Videos/Captures/Grand Theft Auto V 2018-10-07 오후 10_31_38.mp4")
    _, img1 = cap.read()
    _, img2 = cap.read()

    while True:
        x, y = flow.get_direction(img1, img2)
        print('X: {}, Y: {}'.format(x, y))

        arrow = flow.draw_arrow(img1, x ,y)
        cv2.imshow('arrow', arrow)
        cv2.moveWindow('arrow', 0, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User Interrupted')
            break

        img1 = img2
        _, img2 = cap.read()
