import cv2
import numpy as np

RESIZE_WIDTH = 320
RESIZE_HEIGHT = 180

half_height = int(RESIZE_HEIGHT / 2)

cap = cv2.VideoCapture("data/test.mp4")
ret, img1 = cap.read()
frame1 = cv2.resize(img1, (RESIZE_WIDTH, RESIZE_HEIGHT))
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

while(1):
    _, img2 = cap.read()
    frame2 = cv2.resize(img2, (RESIZE_WIDTH, RESIZE_HEIGHT))
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs[half_height:],next[half_height:], None, 0.5, 3, 15, 1, 5, 1.2, 0)
    flow_avg = np.median(flow, axis=(0,1)) # [x, y]
    print('X:%d, Y:%d' % (-1*flow_avg[0], flow_avg[1]))
    move_x = -1*flow_avg[0]
    move_y = -1*flow_avg[1]

    h, w, c = img2.shape
    arrow = cv2.arrowedLine(img2, (int(w/2), int(h/2)), (int(w/2+move_x*15), int(h/2+move_y*15)), color=(0,255,255), thickness=15)
    cv2.imshow('arrow', arrow)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((half_height, RESIZE_WIDTH, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('map',bgr)
    cv2.moveWindow('arrow', 0, 0)
    cv2.moveWindow('map', 0, 420)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('User Interrupted')
        break
    prvs = next
cap.release()
cv2.destroyAllWindows()