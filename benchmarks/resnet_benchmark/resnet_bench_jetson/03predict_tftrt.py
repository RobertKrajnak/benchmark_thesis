from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import glob
import sys

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(gpu_devices[0], [
    tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=1024)])  ## Crucial value, set lower than available GPU memory (note that Jetson shares GPU memory with CPU)
# 2048 neslo - jedine 1024

original_srdout = sys.stdout
with open('print_out_resnet50.txt', 'w') as f:
    sys.stdout = f

    num_pic = [1, 2, 4, 8, 16, 32]
    image_path = r'../../../data/ImageNet_32samples/*.JPEG'
    input_saved_model = "../default_resnet_models_and_labels/resnet50_saved_model_TFTRT_FP16"

    # inicialize - saved model load..

    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)


    # predict run function
    def get_infertime(int_numpic):
        iter = 0
        elapsed_time = []

        """Runs prediction on a single image and shows the result.
        input_saved_model (string): Name of the input model stored in the current dir
        """

        for filename in glob.iglob(image_path):
            if iter == num_pic[int_numpic]:
                break

            print("Predict picture number: ", iter + 1)
            print("Filename predict image: ", filename)

            img_path = filename
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = tf.constant(x)

            print("Start time")
            start_time = time.time()
            labeling = infer(x)
            end_time = time.time()
            print("Stop time")
            elapsed_time = np.append(elapsed_time, end_time - start_time)
            print("elapsed time for run is: ", elapsed_time)

            preds = labeling['predictions'].numpy()
            print('File Path: {} \nPredicted: {}'.format(img_path, decode_predictions(preds, top=3)[0]))

            print(" ")
            iter += 1

        return elapsed_time.sum()


    print("Run 4 Test iter - time is: ", get_infertime(2), " sec")
    print("-----------------------------------------------------------")

    for int_iter in range(len(num_pic)):
        print("Total time inference for " + str(num_pic[int_iter]) + " images is: ", get_infertime(int_iter), " sec")
        print("-----------------------------------------------------------")

    print("Done predict!")

    sys.stdout = original_srdout
