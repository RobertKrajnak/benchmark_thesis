import tensorflow as tf

mobilenet_model = tf.keras.applications.MobileNetV2()
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile('model_new.tflite', 'wb') as f:
    f.write(tflite_model)
