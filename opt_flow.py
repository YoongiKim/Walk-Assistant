import cv2
import numpy as np

class OptFlow:
    def __init__(self, resize_width=320, resize_height=180, height_start=0.2, height_end=0.5):
        self.width = resize_width
        self.height = resize_height

        self.height_start = int(self.height * height_start)
        self.height_end = int(self.height * height_end)

    def get_direction(self, frame1, frame2):
        frame1 = cv2.resize(frame1, (self.width, self.height))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.resize(frame2, (self.width, self.height))
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame1[self.height_start:self.height_end],
                                            frame2[self.height_start:self.height_end], None, 0.5, 3, 15, 1, 5, 1.2, 0)
        flow_avg = np.median(flow, axis=(0, 1))  # [x, y]

        move_x = -1 * flow_avg[0]
        move_y = -1 * flow_avg[1]

        return move_x, move_y

if __name__ == '__main__':
    flow = OptFlow()

    cap = cv2.VideoCapture("C:/Users/Yoongi Kim/Videos/Captures/Grand Theft Auto V 2018-10-07 오후 10_31_38.mp4")
    _, img1 = cap.read()
    _, img2 = cap.read()

    while True:
        x, y = flow.get_direction(img1, img2)
        print('X: {}, Y: {}'.format(x, y))

        h, w, c = img2.shape
        arrow = cv2.arrowedLine(img2, (int(w / 2), int(h / 2)), (int(w / 2 + x * 25), int(h / 2 + -1*abs(y * 50))),
                                color=(0, 255, 255), thickness=15)

        cv2.imshow('arrow', arrow)
        cv2.moveWindow('arrow', 0, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User Interrupted')
            break

        img1 = img2
        _, img2 = cap.read()
