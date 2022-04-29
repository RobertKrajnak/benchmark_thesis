# little memory

import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
from PIL import Image
import numpy as np

# Convert model to ONNX format.
model = Net(weights='imagenet')

# model test - non optimalized model
BATCH_SIZE = 4
img = Image.open('n01443537_11099_goldfish.jpg')
img = img.resize((224, 224))

input_batch = 255*np.array(np.repeat(np.expand_dims(np.array(img, dtype=np.float32), axis=0), BATCH_SIZE, axis=0), dtype=np.float32)
predictions = model.predict(input_batch) # warm up
indices = (-predictions[0]).argsort()[:5]

print("Class | Probability (out of 1)")
print(list(zip(indices, predictions[0][indices])))
