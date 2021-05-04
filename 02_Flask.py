import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, flash, request, redirect, url_for



model = tf.keras.models.load_model('model/default.h5')
model.summary()

#asdf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model.evaluate(x_test,  y_test, verbose=1)
#asdf


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
    return "{"+str(model.predict_classes(dst)[0])+":"+str(model.predict(dst)[0])+"}" 

app.run(host="0.0.0.0", port=5000)