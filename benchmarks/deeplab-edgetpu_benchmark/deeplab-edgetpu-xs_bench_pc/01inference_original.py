# https://tfhub.dev/google/lite-model/edgetpu/vision/deeplab-edgetpu/default_argmax/xs/1

from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
from sklearn.metrics import jaccard_score


MODEL = "../default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite"
MODEL_IMAGE_WIDTH = 512
MODEL_IMAGE_HEIGHT = 512
elapsed_time = []

img = Image.open('../../../data/ADE20K_samples/ADE_val_00000001.jpg').convert('RGB')
im = img.resize((512, 512))
plt.imshow(im)
plt.show()
input_data = np.expand_dims(im, axis=0)
print("input_data.shape: ", input_data.shape)
preprocessed_data = (input_data - 128).astype(np.int8)

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], preprocessed_data)

print("Start time")
start_time = time.time()
interpreter.invoke()
end_time = time.time()
print("Stop time")
elapsed_time = np.append(elapsed_time, end_time - start_time)
print("elapsed time for run is: ", elapsed_time)

output_data = interpreter.get_tensor(output_details[0]['index'])
predict_mask = output_data.reshape((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH))

print("output_data.shape: ", predict_mask.shape)
plt.imshow(predict_mask)
plt.show()

GT_mask = Image.open('../default_deeplab-edgetpu_models_and_labels/groud_truth_mask/ADE_val_00000001.png')
GT_mask = GT_mask.resize((512, 512))
GT_mask = np.array(GT_mask)
print("img.shape: ", GT_mask.shape)
plt.imshow(GT_mask)
plt.show()


jac_sum = []

for y in range(MODEL_IMAGE_WIDTH):
    jac = jaccard_score(GT_mask[y], predict_mask[y], average='weighted')
    jac_sum = np.append(jac_sum, jac)

print(len(jac_sum))
print(jac_sum.sum()/len(jac_sum))