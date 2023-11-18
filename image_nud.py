import json
from os import listdir
from os.path import isfile, join, exists, isdir, abspath

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import time

import cv2

# Add these import statements for video processing
from tqdm import tqdm

IMAGE_DIM = 224   # required/default image dimensionality

from flask import Flask, request, jsonify

app = Flask(__name__)




def load_images(image_paths, image_size, verbose=True):
    '''
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        image_paths: list of image paths to load
        image_size: size into which images should be resized
        verbose: show all of the image path and sizes loaded
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
        loaded_image_indexes: paths of images which the function is able to process
    
    '''
    print(image_paths, image_size, "!!!!!!!!!!!!!!!!!!!!!!!!!")
    loaded_images = []
    loaded_image_paths = []

    if isdir(image_paths):
        parent = abspath(image_paths)
        image_paths = [join(parent, f) for f in listdir(image_paths) if isfile(join(parent, f))]
    elif isfile(image_paths):
        image_paths = [image_paths]

    for img_path in image_paths:
        try:
            if verbose:
                print(img_path, "size:", image_size)
            image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("Image Load Failure: ", img_path, ex)
    
    return np.asarray(loaded_images), loaded_image_paths


def load_model(model_path):
    if model_path is None or not exists(model_path):
    	raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
    
    # model = tf.keras.models.load_model(model_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    # model.summary()
    print(model.summary())
    return model


def classify(model, input_paths, image_dim=IMAGE_DIM):
    """ Classify given a model, input paths (could be single string), and image dimensionality...."""
    imagess, image_paths = load_images(input_paths, (image_dim, image_dim))
    print(imagess, "^^^^^^^^^^^^^^^^^^^^^^^^^")
    
    # print(model, "///////////////////////////")
    probs = classify_nd(model, imagess)
    return dict(zip(image_paths, probs))


def classify_nd(model, nd_images):
    """ Classify given a model, image array (numpy)...."""

    model_preds = model.predict(nd_images)

    
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy', '']

    probs = []
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        probs.append(single_probs)
    return probs

import os

@app.route('/nude/', methods=['POST'])
def main():
    file = request.files['file']
    image_path = os.path.join('images', file.filename)
    file.save(image_path)

    model = load_model("Nudity-Detection-Model.h5")

    start_time = time.time()
    # Classify the image
    image_preds = classify(model, image_path, IMAGE_DIM)
    print("Image Predictions:")
    print(json.dumps(image_preds, indent=2))

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    return image_preds


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
