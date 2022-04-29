import tensorflow as tf

# -------------- load mobilenet - convert mobilenet to tf lite - save mobilenet model -------------

# ------------------------------------------------------------------
import sys
# for main script run path specific
import os
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------

mobilenet_model = tf.keras.applications.MobileNetV2()
# mobilenet_model.save('mobilenet_model.h5')

# converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
# with open('../default_mobilenet_models/mobilenetv2_model.tflite', 'wb') as f:
#     f.write(tflite_model)

# tflite quant model float16
converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('../default_mobilenet_models_and_labels/mobilenetv2_model_FP16.tflite', 'wb') as f:
    f.write(tflite_quant_model)
