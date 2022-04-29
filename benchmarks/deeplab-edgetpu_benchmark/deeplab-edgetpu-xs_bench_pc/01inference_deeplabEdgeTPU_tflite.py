# https://tfhub.dev/google/lite-model/edgetpu/vision/deeplab-edgetpu/default_argmax/xs/1

from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np
# import matplotlib.pyplot as plt
import os
import time
import glob
from sklearn.metrics import jaccard_score
import sys

# ------------------------------------------------------------------
# for main script run path specific
import os
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------

N_run = 64
batch_size = 1  # batch dont work on this model, set only 1
# --- N_warmup_run < N_run!
N_warmup_run = 5

modelImage_wxh = 512

# ----------------- Define Model interpreter and input output tensor --------------------

interpreter = tflite.Interpreter("../default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite")

# https://github.com/EliSchwartz/imagenet-sample-images - jedna trieda jeden test image
image_path = r'../../../data/ADE20K_samples/*.jpg'

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

batched_input = np.zeros((batch_size, modelImage_wxh, modelImage_wxh, 3), dtype=np.int8)

# --------------------------------------------------------------
elapsed_time = []
jac_list = []

# ----------- load filename images-----------------
i = 0
filename_list_first = []
filename_list = []

for filename in sorted(glob.iglob(image_path)):
    if i == batch_size * N_run:
        break
    filename_list_first = np.append(filename_list_first, filename)
    i += 1

filename_list_split = np.split(filename_list_first, N_run)
print(len(filename_list_split))


def get_infertime(run, jac_list, batched_input, input_details, output_details, elapsed_time, elapse_time_on):
    print("-------- N_run number: ", run, "------------")

    for batch in range(batch_size):
        img = Image.open(filename_list_split[run][batch]).convert('RGB')
        img = img.resize((512, 512))
        # plt.imshow(img)
        # plt.show()
        input_data = np.expand_dims(img, axis=0)
        preprocessed_data = (input_data - 128).astype(np.int8)
        batched_input[batch, :] = preprocessed_data

    print('batched_input shape: ', batched_input.shape)
    interpreter.resize_tensor_input(input_details[0]['index'], batched_input.shape)
    interpreter.resize_tensor_input(output_details[0]['index'], batched_input.shape)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, batched_input)

    if elapse_time_on == True:
        print("Start time")
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        print("Stop time")
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        print("elapsed time for run is: ", elapsed_time)
    else:
        print("Start warmup infer")
        interpreter.invoke()
        print("End warmup infer")

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predict_mask = output_data.reshape((modelImage_wxh, modelImage_wxh))
    # plt.imshow(predict_mask)
    # plt.show()

    if elapse_time_on == True:
        mask_path = os.path.basename(filename_list_split[run][batch])
        GT_mask = Image.open("../default_deeplab-edgetpu_models_and_labels/groud_truth_mask/" + mask_path[:16] + ".png")
        GT_mask = GT_mask.resize((512, 512))
        GT_mask = np.array(GT_mask)
        # plt.imshow(GT_mask)
        # plt.show()

        jac_img = []

        for y in range(modelImage_wxh):
            jac = jaccard_score(GT_mask[y], predict_mask[y], average='weighted')
            jac_img = np.append(jac_img, jac)

        jac_list = np.append(jac_list, jac_img.sum() / len(jac_img))
        print("Jaccard list: ", jac_list)

    print("")

    return elapsed_time, jac_list


print(" ----------- I running warmup run!... ----------- ")
for run in range(N_warmup_run):
    elapsed_time, jac_list = get_infertime(run, jac_list, batched_input, input_details, output_details, elapsed_time,
                                           elapse_time_on=False)
print(" ")
print(" ----------- I running benchmark run!... ----------- ")
for run in range(N_run):
    elapsed_time, jac_list = get_infertime(run, jac_list, batched_input, input_details, output_details, elapsed_time,
                                           elapse_time_on=True)
original_srdout = sys.stdout
with open("print_out_tflite_deeplab_batch"+str(batch_size)+"_nrun"+str(N_run)+".txt", 'w') as f:
    sys.stdout = f

    print("Total elapsed time sum for", N_run * batch_size,
          "images (", N_run, "runs x", batch_size, "batch_size) is {:.2f} sec".format(elapsed_time.sum()))
    print("Mean Jaccard for", N_run * batch_size, "images is: ", jac_list.sum() / N_run * batch_size)
    print('Throughput: {:.3f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    print("Mean Latency for 1 image is: ", elapsed_time.sum() / (N_run * batch_size))

    sys.stdout = original_srdout
