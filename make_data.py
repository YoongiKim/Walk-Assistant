from opt_flow import OptFlow
import glob
import cv2
import os
import numpy as np
import threading

QUEUE_SIZE = 50
DATA_PATH = "data/gopro/*.mp4"
# 영상이 중간에 끊기면 코덱 문제이므로 ffmpeg로 mute 해야 합니다.
# ffmpeg -i test.mp4 -c copy -an test_mute.mp4

OUTPUT_PATH = "D:/Walk-Assistant/frames"
LABEL_PATH = "D:/Walk-Assistant"
os.makedirs(OUTPUT_PATH, exist_ok=True)

files = glob.glob(DATA_PATH)

flow = OptFlow(height_start=0.5, height_end=1.0)

label_lines = []

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

    while succeed:
        x, y = flow.get_direction(img1, img2)
        flow_queue.append((x, y))

        if len(img_queue) >= QUEUE_SIZE:
            pop_img = img_queue.pop(0)
            pop_flow = flow_queue.pop(0)

            way = flow.draw_way(pop_img, flow_queue)
            way = cv2.resize(way, (16, 9))

            way_flatten = np.reshape(np.array(np.array(way)/255).astype(np.int), 144)
            way_encode = ''
            for b in way_flatten:
                way_encode += str(b)
            label_lines.append('%s_%d.jpg,%s\n' % (file_name, count, way_encode))
            threading.Thread(target=cv2.imwrite, args=('%s/%s_%d.jpg' % (OUTPUT_PATH, file_name, count), pop_img)).start()
            print('%s_%d.jpg,%s\n' % (file_name, count, way_encode))

            visual = cv2.resize(way, (1280, 720))
            visual = cv2.cvtColor(np.array(visual).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            visual = cv2.add(pop_img, visual)
            cv2.imshow('visual', visual)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User Interrupted')
                break

        img1 = img2
        succeed, img2 = cap.read()
        img_queue.append(img2)
        count+=1

with open('{}/label.txt'.format(OUTPUT_PATH), 'w') as f:
    f.writelines(label_lines)

print('Completed %d frames' % len(label_lines))
