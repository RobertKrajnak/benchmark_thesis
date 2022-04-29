from PIL import Image
import numpy as np
import time
import os
import tensorflow as tf
from PIL import Image
import glob
import json
import re
import sys

# ------------------------------------------------------------------
# for main script run path specific
import os
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Image Classification')
parser.add_argument('--N_run', type=int, help='Specify N_run Benchmark', default=125, required=True)
parser.add_argument('--N_warmup_run', type=int, help='Specify N_warmup_run Benchmark', default=20, required=True)
parser.add_argument('--batch_size', type=int, help='Specify batch_size Benchmark', default=8, required=True)
parser.add_argument('--top_k', type=int, help='Specify top_k Benchmark', default=5, required=True)
args = parser.parse_args()

N_run = args.N_run
N_warmup_run = args.N_warmup_run
batch_size = args.batch_size
k = args.top_k
# -------------------------------------------------
# N_run = 125
# batch_size = 8
# #   N_warmup_run < N_run! - do metrik sa nerata
# N_warmup_run = 20
# k = 5
# ----------------- Define Model interpreter and input output tensor --------------------

interpreter = tf.lite.Interpreter("../default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite")

# https://github.com/EliSchwartz/imagenet-sample-images - jedna trieda jeden test image
image_path = r'../../../data/ImageNet_1000samples/*.JPEG'

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

# --------------------------------------------------------------
elapsed_time = []
top_acc_temp = 0

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


def get_infertime(run, top_acc_temp, batched_input, input_details, output_details, elapsed_time, elapse_time_on,
                  N_warmup_run_on):

    print("-------- N_run number: ", run, "------------")

    for batch in range(batch_size):

    # -------------- original preprocessing from tensorflow ---------------------------
    #     # s keras loadom trosku vyssia topk:
    #     # img = tf.keras.preprocessing.image.load_img(filename_list_split[run][batch], target_size=(224, 224))
    #     img = Image.open(filename_list_split[run][batch]).convert('RGB')
    #     img = img.resize((224, 224))
    #     x = tf.keras.preprocessing.image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = tf.keras.applications.efficientnet.preprocess_input(x)
    #     batched_input[batch, :] = x
    #     print(filename_list_split[run][batch])
    # batched_input = tf.constant(batched_input)

        #  --------------------- my preprocessing ---------------------------
        img = Image.open(filename_list_split[run][batch]).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img, dtype="float32")
        img = np.expand_dims(img, axis=0)
        batched_input[batch, :] = img
        # print(filename_list_split[run][batch])

    print('batched_input shape: ', batched_input.shape)
    print("len batched_input: ", len(batched_input))

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
        interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)

    for j in range(batch_size):
        results = np.squeeze(output_data[j])
        idx = np.argpartition(results, -k)[-k:]
        top_k_indices = idx[np.argsort((-results)[idx])]
        print("top", k, "prediction classes: ", results[top_k_indices])
        print("top", k, "prediction classes: ", top_k_indices)
        filename_list = filename_list_split[run][j].replace("_", " ")
        print("filepath: ", filename_list)

        # -------------- Search file labels json or txt--------------

        # --------------- for no synset imagenet-simple-labels --------------
        # find = False
        # file = open("../default_efficientnet_models_and_labels/imagenet-simple-labels.json")
        # label_json = json.load(file)
        # for l in range(len(top_k_indices)):
        #     print("class:", top_k_indices[l], " class name:", label_json[top_k_indices[l]])
        #     result = filename_list[j].find(label_json[top_k_indices[l]])
        #     if result > -1 and N_warmup_run_on == False and find == False and N_warmup_run_on == False:
        #         print("I find class in top", k, "acc")
        #         top_acc_temp += 1
        #         find = True
        #     else:
        #         split = label_json[top_k_indices[l]].split(" ")
        #         for spl in range(len(split)):
        #             result_split = filename_list[j].find(split[spl])
        #             if result_split > -1 and find == False and N_warmup_run_on == False:
        #                 print("I find class split in top", k, "acc")
        #                 top_acc_temp += 1
        #                 find = True
        # file.close()

        # ------------- for synset imagenet1000_clsidx_to_labels.txt ----------------
        find = False
        searchfile = open("../default_efficientnet_models_and_labels/imagenet1000_clsidx_to_labels.txt", "r")
        for line in searchfile:
            for i in range(len(top_k_indices)):
                if " " + str(top_k_indices[i]) + ":" in line:
                    print(line, end='', flush=True)
                    line = line.replace("'", "/")
                    result = re.search('/(.+?)/', line)
                    split = result.group(1).split(",")
                    for spl in range(len(split)):
                        result_split = filename_list.find(split[spl])
                        if result_split > -1 and find == False and N_warmup_run_on == False:
                            print("I find class split in top", k, "acc")
                            top_acc_temp += 1
                            find = True
        searchfile.close()

        print("")

    return elapsed_time, top_acc_temp


print(" ----------- I running warmup run!... ----------- ")
for run in range(N_warmup_run):
    elapsed_time, top_acc_temp = get_infertime(run, top_acc_temp, batched_input, input_details, output_details, elapsed_time,
                                               elapse_time_on=False, N_warmup_run_on=True)
print(" ")
print(" ----------- I running benchmark run!... ----------- ")
for run in range(N_run):
    elapsed_time, top_acc_temp = get_infertime(run, top_acc_temp, batched_input, input_details, output_details, elapsed_time,
                                               elapse_time_on=True, N_warmup_run_on=False)

original_srdout = sys.stdout
with open("print_out_tf.lite_efficientnet_batch"+str(batch_size)+"_topk"+str(k)+"_nrun"+str(N_run)+".txt", 'w') as f:
    sys.stdout = f

    print("Total find", top_acc_temp, "class in top", k, "accuracy")
    print("Top", k, "accuracy is", top_acc_temp / (batch_size * N_run), "for", top_acc_temp, "/", batch_size * N_run,
          "images")
    print("Total elapsed time sum for", N_run * batch_size,
          "images (", N_run, "runs x", batch_size, "batch_size) is {:.2f} sec".format(elapsed_time.sum()))
    print('Throughput: {:.3f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    print("Mean Latency for 1 image is: ", elapsed_time.sum() / (N_run * batch_size))

    sys.stdout = original_srdout
