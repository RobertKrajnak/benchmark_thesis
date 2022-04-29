import tensorflow as tf

# -------------- load mobilenet - convert mobilenet to tf lite - save mobilenet model -------------

# converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)
# converter.experimental_new_converter = True
# tflite_model = converter.convert()
# with open('../default_mobilenet_models/mobilenetv2_model.tflite', 'wb') as f:
#     f.write(tflite_model)

# tflite quant model float16
converter = tf.lite.TFLiteConverter.from_saved_model(
    "../default_efficientdet_models_and_labels/efficientdet_lite1_saved_model")
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('../default_efficientdet_models_and_labels/test_batch.tflite', 'wb') as f:
    f.write(tflite_quant_model)
