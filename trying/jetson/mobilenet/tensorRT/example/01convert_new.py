import os
import tensorflow as tf
# import keras
# from tf.keras.applications.mobilenet import MobileNet
import onnx
import tf2onnx
import keras2onnx
import numpy as np


net = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')

# Export the model
# tf.saved_model.save(net, "saved_model")
onnx_model = keras2onnx.convert_keras(net, net.name)


Invalid Node - out_relu/Relu6:0_reshape
Attribute not found: allowzero
#Save the ONNX file.
model_name = "mobilenet_onnx_model_new.onnx"
onnx.save_model(onnx_model, model_name)

print("Done saving!")



