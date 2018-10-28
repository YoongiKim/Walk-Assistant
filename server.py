from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import cv2
import base64
import numpy as np
from model import MyModel

app = Flask(__name__)
api = Api(app)

my_model = MyModel(True, 720, 1280, 80, 80, 1e-3, 'main')  # load model

def encode_result(arr):
    arr = np.array(arr)

    encode = ''
    for i in arr:
        for index, j in enumerate(i):
            encode += '%.2f' % j
            if index != len(i) - 1:
                encode += ','

        encode += '\n'

    return encode[:-1]

class Predict(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('img', type=str)
        args = parser.parse_args()

        img = args['img']

        jpg = base64.b64decode(img)
        jpg = np.fromstring(jpg, np.uint8)
        rgb = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        rgb = cv2.resize(rgb, (1280, 720))

        X = np.array([rgb]) / 255.0
        result = my_model.predict(X)[0, :, :, 1]  # (batch, 9, 16, 2)
        encode = encode_result(result)

        return jsonify({'result': encode, 'status': 'OK'})


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)
