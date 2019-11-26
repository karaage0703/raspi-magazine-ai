#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

from time import sleep
import random
from pyroombaadapter import PyRoombaAdapter

if __name__ == '__main__':
    # parse options
    parser = argparse.ArgumentParser(description='keras-pi.')
    parser.add_argument('-m', '--model', default='./model/mnist_deep_model.json')
    parser.add_argument('-w', '--weights', default='./model/weights.20.hdf5')
    parser.add_argument('-l', '--labels', default='./model/labels.txt')

    args = parser.parse_args()

    PORT = '/dev/ttyUSB0'
    roomba = PyRoombaAdapter(PORT)

    camera = PiCamera()
    camera.resolution = (640, 480)

    labels = []
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())
    print(labels)

    model_pred = model_from_json(open(args.model).read())
    model_pred.load_weights(args.weights)

    # model_pred.summary()

    max_count = 0
    count = 0
    stream = PiRGBArray(camera)
    try:
        while True:
            roomba.move(0.0, np.deg2rad(0.0)) # stop
            sleep(1.0)

            camera.capture(stream, 'bgr', use_video_port=True)

            count += 1
            if count > max_count:
                X = []
                img_org = stream.array
                img = cv2.resize(img_org, (64, 64))
                img = img_to_array(img)
                X.append(img)
                X = np.asarray(X)
                X = X/255.0
                start = time.time()
                preds = model_pred.predict(X)
                elapsed_time = time.time() - start

                pred_label = ""

                label_num = 0
                tmp_max_pred = 0
                # print(preds)
                for i in preds[0]:
                    if i > tmp_max_pred:
                        pred_label = labels[label_num]
                        tmp_max_pred = i
                    label_num += 1

                count = 0

                # Control roomba
                if pred_label == 'go':
                    print('go straight')
                    roomba.move(0.2, np.deg2rad(0.0)) # go straight
                    sleep(1.0)

                else:
                    if random.randint(0,1):
                        print('turn right')
                        roomba.move(0, np.deg2rad(-20)) # turn right
                        sleep(1.0)
                    else:
                        print('turn left')
                        roomba.move(0, np.deg2rad(20)) # turn left
                        sleep(1.0)

            stream.seek(0)
            stream.truncate()

    except KeyboardInterrupt:
        print('Ctrl+C interrupted')
        camera.close()
