import tensorflow as tf

# -------------- load mobilenet - convert mobilenet to tf lite - save mobilenet model -------------

mobilenet_model = tf.keras.applications.MobileNetV2()
# mobilenet_model.save('mobilenet_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)
converter.experimental_new_converter = True
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
