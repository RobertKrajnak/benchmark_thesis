from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

model = EfficientNetB0(weights='imagenet')

# test - no more memory
# img_path = 'n01443537_11099_goldfish.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('{} - Predicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))


# Save the entire model as a SavedModel.
model.save('../default_efficientnet_models_and_labels/efficientnetb0_saved_model')

print("Done saving!")
