import flask
from flask import Flask
from tensorflow import keras
import cv2
import numpy as np
import os.path
# ALL CREDIT FOR FER MODEL GOES TO: ivadym on github
# https://github.com/ivadym/FER
app = Flask(__name__)

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
model = keras.models.load_model(os.path.dirname(__file__)+"model/ResNet-50.h5", compile=False)

def preprocess_input(image):
    image = cv2.resize(image, (197, 197))  # Resizing images for the trained model
    ret = np.empty((197, 197, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis=0)  # (1, XXX, XXX, 3)
    x -= 128.8006  # np.mean(train_dataset)
    x /= 64.6497  # np.std(train_dataset)
    return x

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            npimg = np.fromstring(image, np.uint8)
            # convert numpy array to image
            open_cv_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            processed_input = preprocess_input(open_cv_image)
            global model
            global emotions
            prediction = model.predict(processed_input)
            data["prediction_array"] = {emotions[i] : float(prediction[0][i]) for i in range(len(emotions))}

            data["success"] = True
    return flask.jsonify(data)



@app.route('/')
def greeter():
    return 'mental_health_api'


if __name__ == '__main__':
    load_model()
    app.run()
