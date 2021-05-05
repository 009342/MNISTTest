import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
from flask import Flask, flash, request, redirect, url_for



model = tf.keras.models.load_model('model/default.h5')
model.summary()


app = Flask(__name__)

@app.route('/api/image', methods=['POST'])
def doImageProcessing():
    r = request
    nparr = np.fromstring(r.files['image'].read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    dst = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    dst = 1 - dst / 255 
    dst = np.reshape(dst,(1, 784)) 
    a = model.predict(dst)
    return json.dumps(a[0].tolist())

app.run(host="0.0.0.0", port=5000)
