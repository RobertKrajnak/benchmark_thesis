import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
import onnx
import tf2onnx
import keras2onnx
from PIL import Image
import numpy as np

# Convert model to ONNX format.
model = Net(weights='imagenet')
# model.save("./Desktop/mobilenet.h5")
onnx_model = keras2onnx.convert_keras(model, model.name)
# os.system('python3 -m tf2onnx.convert --saved-model mobilenet.h5 --output temp.onnx')
# onnx_model = onnx.load_model('temp.onnx')

# Set an explicit batch size in the ONNX file.
# By default, TensorFlow doesn’t set an explicit batch size.
BATCH_SIZE = 32
inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

#Save the ONNX file.
model_name = "mobilenet_onnx_model.onnx"
onnx.save_model(onnx_model, model_name)

print("Done saving!")




