import json
from os import listdir
from os.path import isfile, join, exists, isdir, abspath

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import time
import cv2

from tqdm import tqdm

IMAGE_DIM = 224   # required/default image dimensionality

# from flask import Flask, request, jsonify

# app = Flask(__name__)

# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     # Get the image from the request's files
#     if 'image' not in request.files:
#         return jsonify({"error": "No image found in the request"}), 400

#     image = request.files['image']
    

def load_images(image_paths, image_size, verbose=True):

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


def classify_nd(model, nd_images):

    model_preds = model.predict(nd_images)

    
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    probs = []
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        probs.append(single_probs)
    return probs



def process_video(video_path, model, batch_size=32, image_dim=IMAGE_DIM, frame_skip=5):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the video position to the beginning
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    output_data = []
    frames_batch = []

    # Calculate the progress per frame
    progress_per_frame = 100 / total_frames

    # Keep track of the current frame index
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if required
        current_frame += 1
        if current_frame % frame_skip != 0:
            continue

        # Convert the frame to RGB and resize it to the desired dimension
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_dim, image_dim))

        # Convert the frame to a format suitable for the model
        image = keras.preprocessing.image.img_to_array(frame)
        image /= 255
        frames_batch.append(image)

        # If the batch is full, process it through the model
        if len(frames_batch) == batch_size:
            frames_batch = np.array(frames_batch)
            predictions = classify_nd(model, frames_batch)
            output_data.extend(predictions)
            frames_batch = []

        # Calculate the current progress and display it
        current_progress = current_frame * progress_per_frame
        print(f"Progress: {current_progress:.2f}%", end='\r')

    # Process any remaining frames in the last batch
    if len(frames_batch) > 0:
        frames_batch = np.array(frames_batch)
        predictions = classify_nd(model, frames_batch)
        output_data.extend(predictions)

    print()  # Newline to clear the progress display
    cap.release()

    return output_data


def main():
    video_path = "videos/abx.mp4"
    model = load_model("Nudity-Detection-Model.h5")

    start_time = time.time()

    # Process the video, reducing the frame processing and increasing the batch size
    video_preds = process_video(video_path, model, batch_size=64, image_dim=IMAGE_DIM, frame_skip=10)
    print("Video Predictions:")
    print(json.dumps(video_preds, indent=2))


    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
