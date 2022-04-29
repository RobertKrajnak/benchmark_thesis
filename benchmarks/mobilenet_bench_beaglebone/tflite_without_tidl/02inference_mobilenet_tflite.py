# import tensorflow as tf
import tflite_runtime.interpreter as tflite

# import cv2
from PIL import Image
import numpy as np
import glob
import time

# # ------------ load tf lite model ------------

interpreter = tflite.Interpreter("model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# # --------------- Read the image preprocess and decode to a tensor -----------------

num_pic = [1, 2, 4, 8, 16, 32]
image_path = r'../../data/ImageNet_32samples/*.jpg'


def get_infertime(int_numpic):
    iter = 0
    elapsed_time = []

    for filename in glob.iglob(image_path):
        if iter == num_pic[int_numpic]:
            break

        # print("Predict picture number: ", iter+1)
        # print("Filename predict image: ", filename)

        img = Image.open(filename)
        img = img.resize((224, 224))
        # img = tf.cast(img, tf.float32)
        # img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.array(img, dtype="float32")
        img = img / 255

        # # -----------------Preprocess the image to required size and cast ------------------

        input_tensor = np.array(np.expand_dims(img, 0), dtype=np.float32)

        # #----------------- set the tensor to point to the input data to be inferred ------------------

        # print("Start time")
        start_time = time.time()

        input_index = interpreter.get_input_details()[0]["index"]
        interpreter.set_tensor(input_index, input_tensor)

        # #----------------- Run the inference ------------------

        interpreter.invoke()
        output_details = interpreter.get_output_details()

        end_time = time.time()
        # print("Stop time")
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        # print("elapsed time for run is: ", elapsed_time)

        # # -------------------- Interpreting output tensor for our Image classification --------------

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        top_k = results.argsort()

        # _, top_k_indices = tf.math.top_k(output_data, k=2)
        #  = np.array(top_k_indices)[0]
        # print("top 2 prediction classes: ", top_k_indices)
        #
        # # -------------- Search file labels --------------
        #
        # searchfile = open("imagenet1000_clsidx_to_labels.txt", "r")
        # for line in searchfile:
        #     for i in range(len(top_k_indices)):
        #         if " "+str(top_k_indices[i])+":" in line:
        #             print(line, end='', flush=True)
        # searchfile.close()

        # print(" ")
        iter += 1
    
    elapsed_time = np.array(elapsed_time, dtype=np.int8)

    return elapsed_time.sum()


print("Run 2 Test iter - time is: ", get_infertime(1), " sec")
print("-----------------------------------------------------------")

for int_iter in range(len(num_pic)):
    print("Total time inference for " + str(num_pic[int_iter]) + " images is: ", get_infertime(int_iter), " sec")
    print("-----------------------------------------------------------")
