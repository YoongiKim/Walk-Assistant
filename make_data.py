from opt_flow import OptFlow
import glob
import cv2
import os
import numpy as np
import threading
from tqdm import tqdm, trange

QUEUE_SIZE = 50
DATA_PATH = "H:/Workspaces/Walk-Assistant/data/videos/*.mp4"

# 영상이 중간에 끊기면 코덱 문제이므로 ffmpeg로 mute 해야 합니다.
# Linux: ffmpeg -i data/videos/test.mp4 -c copy -an data/videos/test_mute.mp4
# Windows: ffmpeg.exe -i data/videos/test.mp4 -c copy -an data/videos/test_mute.mp4

START_SKIP = 500  # 10초 생략
END_SKIP = 500  # 10초 생략

OUTPUT_PATH = "H:/Workspaces/Walk-Assistant/data/frames"
os.makedirs(OUTPUT_PATH, exist_ok=True)

files = glob.glob(DATA_PATH)

flow = OptFlow(height_start=0.5, height_end=1.0)

label_lines = []

for file in files:
    print(file)
    file_name = str(file).replace('\\', '/').split('/')[-1].replace('.mp4', '')

    cap = cv2.VideoCapture(file)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(START_SKIP):
        cap.read()

    img_queue = []
    flow_queue = []  # (x, y) moved position of previous frame

    _, img1 = cap.read()
    img_queue.append(img1)
    succeed, img2 = cap.read()
    img_queue.append(img2)

    with trange(START_SKIP, total-END_SKIP-1) as t:
        for i in t:
            x, y = flow.get_direction(img1, img2, show=False)
            flow_queue.append((x, y))

            if len(img_queue) >= QUEUE_SIZE:
                pop_img = cv2.resize(img_queue.pop(0), (1280, 720))
                pop_flow = flow_queue.pop(0)

                # way_visual = flow.draw_way(pop_img, flow_queue, size=40)
                way = flow.draw_way(pop_img, flow_queue, size=200)
                way = cv2.resize(way, (16, 9))

                way_flatten = np.reshape(np.array(np.array(way)/255).astype(np.int), 144)

                if np.sum(way_flatten) > 2:  # 횡단보도 대기 무시
                    way_encode = ''
                    for b in way_flatten:
                        way_encode += str(b)
                    label_lines.append('%s_%d.jpg,%s\n' % (file_name, i, way_encode))
                    threading.Thread(target=cv2.imwrite, args=('%s/%s_%d.jpg' % (OUTPUT_PATH, file_name, i), pop_img)).start()
                    # t.write('%s_%d.jpg,%s\n' % (file_name, i, way_encode))

                    visual = cv2.resize(way, (1280, 720))
                    visual = cv2.cvtColor(np.array(visual).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    visual = cv2.add(pop_img, visual)

                    # way_visual = cv2.cvtColor(np.array(way_visual).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    # way_visual[:,:,0]=0
                    # visual = cv2.add(visual, way_visual)

                    cv2.imshow('visual', visual)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print('User Interrupted')
                        break

            img1 = img2
            succeed, img2 = cap.read()
            img_queue.append(img2)

print('Processed %d frames' % len(label_lines))

last_labels = []
if os.path.exists('{}/label.txt'.format(OUTPUT_PATH)):
    print('Found existing label file. Appending labels...')
    with open('{}/label.txt'.format(OUTPUT_PATH), 'r') as p:
        last_labels = p.readlines()

total_labels = last_labels + label_lines
set_labels = set(total_labels)
print('Found %d duplicated labels' % (len(total_labels) - len(set_labels)))

with open('{}/label.txt'.format(OUTPUT_PATH), 'w') as f:
    f.writelines(set_labels)

print('Saved %d labels' % len(set_labels))
