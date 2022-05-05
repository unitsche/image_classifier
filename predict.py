#!/usr/bin/env python

import argparse
import json
#import utility # my utility package
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import Image
import logging
tf.get_logger().setLevel(logging.ERROR)
from pathlib import Path
# this is to get rid of cumbersome verbose GPU apple M1 chip warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def process_image(img):
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, [224, 224])
    img /= 255
    return img.numpy()

def predict(image_path, keras_model, top_k=3):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    img = np.expand_dims(img, axis=0)
    pred = keras_model.predict(img)
    # tf.math.top_k function
    probs, classes = tf.math.top_k(pred, top_k)
    probs = probs.numpy().squeeze()
    classes = classes.numpy().squeeze()

    return probs, classes

def __main__():

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "path-to-img",
    help = "Please specify a path to your image",
    )

  parser.add_argument(
    "path-to-model",
    help = "Please specify a path to your model",
    )

  parser.add_argument(
    "--top_k",
    help = "Please specify the number of results",
    default = 3,
    type = int,
    required = False 
  )

  parser.add_argument(
    "--category_names",
    help = "Please specify a path to a JSON file with the category names",
    required = False
  )

  args = parser.parse_args()
  arguments = args.__dict__

  image_path     = arguments['path-to-img']
  model_path     = arguments['path-to-model']
  top_k          = arguments['top_k']
  category_names = arguments['category_names']


  # some sanity checks
  if Path(image_path).is_file() == False:
      print("Error:the path to the foto does not exist")
      return

  if Path(model_path).is_file() == False:
      print("Error: the path to the model does not exist")
      return 
 
  if top_k and top_k > 102:
    print("Error: max top_k value is 102")
    return

  if category_names:
    path = Path(category_names)
    if path.is_file() == False:
      print("Error: the path to the category file not exist")
      return


  try:
    keras_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})

    probs, classes = predict(image_path, keras_model, top_k)

    # correct the mapping by adding +1 to each key in the json file
    classes = classes+1

    # if top_k quals 1: special treatment - otherwise un error is thrown:
    # 'numpy.int64' object is not iterable'
    if top_k == 1:
      probs = [probs.tolist()]
      classes = [classes.tolist()]

    # if category_name specified:
    if category_names:
      with open(category_names) as f:
        d = json.load(f)
        class_names = []
        for class_name in classes:
          class_names.append(d[str(class_name)])

      zip_iterator = zip(class_names, probs)
    # if no category_name file specified
    else:
      zip_iterator = zip(classes, probs)

    res = dict(zip_iterator)
    print(res)
  except BaseException as err:
    print(err)
    raise

if __name__ == __main__():
  main()
