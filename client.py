import requests
import json
import cv2
import base64
import numpy as np

addr = 'http://127.0.0.1'
test_url = addr + '/predict'

WIDTH = 1280
HEIGHT = 720

def decode_result(s):
    arr = []

    rows = str(s).split('\n')

    for row in rows:
        cols = row.split(',')

        cols_parse = []
        for col in cols:
            cols_parse.append(float(col))

        arr.append(cols_parse)

    return np.array(arr, dtype=np.float32)

def render(img, res):
    res *= 255
    res = res.astype(np.uint8)

    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)  # to white color
    res[:, :, 0] = 0  # remove blue channel
    res[:, :, 2] = 0  # remove red channel

    res = cv2.resize(res, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)  # resize 15x8 to 1280x720
    org = cv2.resize(img, (WIDTH, HEIGHT))
    added = cv2.add(org, res)  # combine input, output

    cv2.imshow('result', added)
    cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print('User Interrupted')
    #     exit(1)

if __name__ == '__main__':
    img = cv2.imread('data/test1.jpg')
    _, img_encoded = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(img_encoded)

    response = requests.post(test_url, data={"img": b64})

    result = json.loads(response.text)['result']
    print(result)

    result = decode_result(result)

    render(img, result)
