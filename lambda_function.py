import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify, request
from flask import request as requ
from keras.preprocessing.image import load_img
import numpy as np
import flask
import requests
import os
from io import BytesIO
from urllib import request
from PIL import Image
import json
import waitress
TARGET_SIZE = (80, 80)
classes = ['dog', 'cat']

interpreter = tflite.Interpreter(model_path='dog_cat_classifier.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def preprocess(X):
    if isinstance(X, np.ndarray):
        # Already a NumPy array, just normalize
        X = X.astype(np.float32) / 255.0
        return X
    else:
        # Convert PIL image to NumPy array
        X = np.array(X)
        # Assuming RGB image, convert to grayscale (adjust if needed)
        X = X[:, :, 0]
        # Reshape and normalize
        X = X.astype(np.float32).reshape(-1, 80, 80, 1) / 255.0
        return X


def download_image(url):
    with request.urlopen(url) as resp:
        img_data = resp.read()
    return Image.open(BytesIO(img_data))

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
app = Flask('pred')

from flask import jsonify


@app.route('/lambda_function', methods=['POST'])
def predict():
    data = requ.get_json()
    url = data.get('url', '')

    img = download_image(url)
    img = prepare_image(img, TARGET_SIZE)
    X = preprocess(img)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    # Flatten the nested list
    flat_predictions = [item for sublist in preds for item in sublist]
    flat_predictions.append(1 - preds[0][0])  # Assuming you want to append the complement of the first value

    # Convert NumPy arrays to Python lists
    float_predictions = [float(value) for value in flat_predictions]

    # Directly create a dictionary with string keys and list values
    json_predictions = {class_name: value for class_name, value in zip(classes, float_predictions)}

    return jsonify(json_predictions)







def lambda_handler(event, context):
    print("parameters: ",event)
    url = event['url']
    results = predict(url)
    return results

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8080)