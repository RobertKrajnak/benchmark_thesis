import tensorflow as tf
import os
import sys
os.chdir(os.path.dirname(sys.argv[0]))

# -------------- load efficientnet - convert efficientnet to tf lite - save mobilenet model -------------

resnet50_model = tf.keras.applications.ResNet50(weights="imagenet")

# converter = tf.lite.TFLiteConverter.from_keras_model(resnet50_model)
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
# with open('../default_resnet_models/resnet50_mode.tflite', 'wb') as f:
#     f.write(tflite_model)

# tflite quant model float16
converter = tf.lite.TFLiteConverter.from_keras_model(resnet50_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('../default_resnet_models_and_labels/resnet50_model_FP16.tflite', 'wb') as f:
    f.write(tflite_quant_model)
