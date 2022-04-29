import tensorflow as tf

# -------------- load efficientnet - convert efficientnet to tf lite - save mobilenet model -------------

# ------------------------------------------------------------------
import sys
# for main script run path specific
import os
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------

efficientnet_model = tf.keras.applications.EfficientNetB0(weights="imagenet")

# converter = tf.lite.TFLiteConverter.from_keras_model(efficientnet_model)
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
# with open('../default_efficientnet_models_and_labels/efficientnetb0_model.tflite', 'wb') as f:
#     f.write(tflite_model)

# tflite quant model float16
converter = tf.lite.TFLiteConverter.from_keras_model(efficientnet_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('../default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite', 'wb') as f:
    f.write(tflite_quant_model)
