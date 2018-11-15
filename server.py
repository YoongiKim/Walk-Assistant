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
print("""
    Walk-Assistant Copyright (C) 2018 Yoongi Kim
    This program comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it
    under certain conditions.
""")


from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import cv2
import base64
import numpy as np
from model import MyModel
from flask_cors import CORS
from flask import request
import json
import matplotlib.pyplot as plt
import time
import threading

app = Flask(__name__)
CORS(app)  # Cross Origin Resource Sharing (CORS)
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
    """
    post (base64 encoded): { "img":"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wB..." }
    receive:{ "result": "0.97,0.99,1.00,0.07,0.35,1.00,1.00,1.00,1.00,1.00,0.99,0.01,0.00,0.00,0.02,0.32\n0.01,..." ,
    "status": "OK" }
    """
    def post(self):
        print(request.data)

        parser = reqparse.RequestParser()
        parser.add_argument('img', type=str)
        parser.add_argument('save', type=str)
        args = parser.parse_args()

        img = args['img']
        save = args['save']
        img = str(img).replace('data:image/jpeg;base64,', '')

        jpg = base64.b64decode(img)
        jpg = np.fromstring(jpg, np.uint8)
        bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (1280, 720))

        if str(save).lower() == 'true':
            threading.Thread(target=plt.imsave, args=['history/{}.jpg'.format(time.time()), rgb]).start()

        X = np.array([rgb]) / 255.0
        result = my_model.predict(X)[0, :, :, 1]  # (batch, 9, 16, 2)
        encode = encode_result(result)

        data = {'result': encode, 'status': 'OK'}
        js = json.dumps(data)

        return js


api.add_resource(Predict, '/walk')

if __name__ == '__main__':
    context = ('ssl/cert.pem', 'ssl/privkey.pem')
    app.run(host='0.0.0.0', port=8080, ssl_context=context)  # 0.0.0.0 is for external connection
