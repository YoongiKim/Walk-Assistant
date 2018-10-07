from matplotlib import pyplot as plt
import cv2
import numpy as np
from deeplab_v3.model import Deeplabv3
from tqdm import tqdm

deeplab_model = Deeplabv3()

vidcap = cv2.VideoCapture('data/test.mp4')
total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

success, image = vidcap.read()
for i in tqdm(range(0, total)):
    if success:
        img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('org', img)
        img = img / 127.5 - 1

        res = deeplab_model.predict(np.array(img).reshape((1, 512, 512, 3)))
        labels = np.argmax(res.squeeze(), -1)
        labels = labels * 12.75
        labels = labels.astype(np.uint8)
        labels = cv2.cvtColor(labels, cv2.COLOR_GRAY2BGR)

        cv2.imshow('result', labels)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('User Interrupted')
            exit(1)

        success, image = vidcap.read()