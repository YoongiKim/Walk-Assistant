from opt_flow import OptFlow
import glob
import cv2
import os
import numpy as np

QUEUE_SIZE = 50
DATA_PATH = "data/gopro/*.mp4"
# 영상이 중간에 끊기면 코덱 문제이므로 ffmpeg로 mute 해야 합니다.
# ffmpeg -i test.mp4 -c copy -an test_mute.mp4

OUTPUT_PATH = "data/frames"
os.makedirs(OUTPUT_PATH, exist_ok=True)

files = glob.glob(DATA_PATH)

flow = OptFlow(height_start=0.5, height_end=1.0)

for file in files:
    file_name = str(file).replace('\\', '/').split('/')[-1].replace('.mp4', '')

    cap = cv2.VideoCapture(file)

    img_queue = []
    flow_queue = []  # (x, y) moved position of previous frame

    _, img1 = cap.read()
    img_queue.append(img1)
    succeed, img2 = cap.read()
    img_queue.append(img2)

    count = 0

    while True:
        x, y = flow.get_direction(img1, img2)
        flow_queue.append((x, y))
        print('X: {}, Y: {}'.format(x, y))

        if len(img_queue) >= QUEUE_SIZE:
            pop_img = img_queue.pop(0)
            pop_flow = flow_queue.pop(0)
            # cv2.imwrite('%s/%s_%d_%.2f_%.2f.jpg'%(OUTPUT_PATH, file_name, count, x, y), pop_img)

            way = flow.draw_way(pop_img, flow_queue)
            cv2.imshow('way', way)
            cv2.moveWindow('way', 0, 0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                break

        img1 = img2
        succeed, img2 = cap.read()
        img_queue.append(img2)
        count+=1