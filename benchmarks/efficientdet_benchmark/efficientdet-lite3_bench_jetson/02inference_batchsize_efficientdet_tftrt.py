# save model:
# https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import os
import tensorflow as tf
from PIL import Image
import glob
import json
# import cv2
import sys

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(gpu_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
## Crucial value, set lower than available GPU memory (note that Jetson shares GPU memory with CPU)
# 2048 neslo - jedine 1024

# ------------------------------------------------------------------
# for main script run path specific
import os
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='Image Detect')
parser.add_argument('--N_run', type=int, help='Specify N_run Benchmark', default=64, required=True)
parser.add_argument('--N_warmup_run', type=int, help='Specify N_warmup_run Benchmark', default=5, required=True)
args = parser.parse_args()
N_run = args.N_run
N_warmup_run = args.N_warmup_run
# -------------------------------------------------
predict_threshold = 0.6
iou_threshold = 0.6
# -------------------------------------
# N_run = 10
batch_size = 1  # batchsize pri tomto modely neefunguje => nechat 1
### N_warmup_run < N_run!
# N_warmup_run = 2
# ----------------------------------------------------

# --------------- metric IoU function ----------
def get_iou(bbox_GT, bbox_predict):
    xA = max(bbox_GT[0], bbox_predict[0])
    yA = max(bbox_GT[1], bbox_predict[1])
    xB = min(bbox_GT[2], bbox_predict[2])
    yB = min(bbox_GT[3], bbox_predict[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (bbox_GT[2] - bbox_GT[0] + 1) * (bbox_GT[3] - bbox_GT[1] + 1)
    boxBArea = (bbox_predict[2] - bbox_predict[0] + 1) * (bbox_predict[3] - bbox_predict[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    print("IoU: ", iou)
    return iou


# ----------------- Define Model interpreter and input output tensor --------------------
input_saved_model = "../default_efficientdet_models_and_labels/efficientdet_lite3_saved_model_TFTRT_FP16"
image_path = r'../../../data/Coco2017_val_samples/*.jpg'

# inicialize - saved model load..

saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
signature_keys = list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer = saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

# ----------------- define array for batch ---------------------------
batched_input = np.zeros((batch_size, 512, 512, 3), dtype=np.uint8)

# --------------------------------------------------------------
elapsed_time = []
mean_iou_count = 0

# ----------- load filename images-----------------
i = 0
filename_list_first = []
filename_list = []
mean_iou_list = []

for filename in sorted(glob.iglob(image_path)):
    if i == batch_size * N_run:
        break
    filename_list_first = np.append(filename_list_first, filename)
    i += 1
filename_list_split = np.split(filename_list_first, N_run)
print(len(filename_list_split))


def get_infertime(mean_iou_list, run, mean_iou_count, batched_input, elapsed_time, elapse_time_on):
    print("-------- N_run number: ", run, "------------")

    for batch in range(batch_size):
        # -------------- original preprocessing from tensorflow ---------------------------
        image = Image.open(filename_list_split[run][batch]).convert('RGB')
        img = image.resize((512, 512))
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        img = np.expand_dims(img, axis=0)
        # img = tf.cast(img, tf.uint8)
        batched_input[batch, :] = img
        # print(filename_list_split[run][batch])
    batched_input = tf.constant(batched_input)

        #  --------------------- my preprocessing ---------------------------
        # image = Image.open(filename_list_split[run][batch]).convert('RGB')
        # img = image.resize((512, 512))
        # # img = np.array(img)
        # # img = img[:, :, ::-1]
        # b, g, r = img.split()
        # img = Image.merge("RGB", (r, g, b))
        # img = np.expand_dims(img, axis=0)
        # batched_input[batch, :] = img
        # print(filename_list_split[run][batch])

    print('batched_input shape: ', batched_input.shape)
    print("len batched_input: ", len(batched_input))

    if elapse_time_on == True:
        print("Start time")
        start_time = time.time()
        labeling = infer(batched_input)
        end_time = time.time()
        print("Stop time")
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        print("elapsed time for run is: ", elapsed_time, "\n")
    else:
        print("Start Warm Up infer...")
        labeling = infer(batched_input)
        print("End Warm Up infer...\n")

    # # -------------------- Interpreting output tensor for object detect --------------

    detection_boxes = labeling['output_0'].numpy()
    detection_scores = labeling['output_1'].numpy()
    detection_clases = labeling['output_2'].numpy()
    num_detection = labeling['output_3'].numpy()

    for j in range(batch_size):
        for i in range(int(num_detection[0])):
            if detection_scores[0, i] > predict_threshold and elapse_time_on == True:
                classes = detection_clases[0, i]
                boxes = detection_boxes[0, i]
                id_image = []
              
                imW, imH = image.size
                print("image.size: ", image.size)
                scale_imH, scale_imW = imH / 512, imW / 512

                file_bbox = open("../default_efficientdet_models_and_labels/instances_val2017.json")
                bbox_json = json.load(file_bbox)

                for imag in bbox_json['images']:
                    if imag['file_name'] == os.path.basename(filename_list_split[run][batch]):
                        id_image = imag["id"]
                        print("id_image: ", id_image)
                        print("I looking for new object on image...\n")
                        break

                for ann in bbox_json['annotations']:
                    if ann['image_id'] == id_image:
                        bbox_GT = ann["bbox"]
                        print("I am looking for GT bbox with matching IoU > ", iou_threshold, "....")

                        bbox_GT = [bbox_GT[0], bbox_GT[1], bbox_GT[2] + bbox_GT[0], bbox_GT[3] + bbox_GT[1]]
                        bbox_predict = [round(boxes[1] * scale_imW, 2), round(boxes[0] * scale_imH, 2), round(boxes[3] * scale_imW, 2), round(boxes[2] * scale_imH, 2)]

                        print("bbox ground truth: ", bbox_GT)
                        print("bbox predict: ", bbox_predict)

                        file_class = open("../default_efficientdet_models_and_labels/image_info_test2017.json")
                        label_json = json.load(file_class)
                        for j in label_json['categories']:
                            if j['id'] == int(classes):
                                print("class:", classes, " class name:", j['name'], " score:", detection_scores[0, i],
                                      end='',
                                      flush=True)
                                file_class.close()
                                print(" ")
                                break

                        iou_score = get_iou(bbox_GT, bbox_predict)

                        if iou_score > iou_threshold:
                            print("I find GT bbox for predicted bbox")
                            mean_iou_list = np.append(mean_iou_list, iou_score)
                            mean_iou_count += 1
                            print("mean_iou_list: ", mean_iou_list, "\n")
                            break

                        print(" ")

                        continue

    return elapsed_time, mean_iou_count, mean_iou_list


print(" ----------- I running warmup run!... ----------- ")
for run in range(N_warmup_run):
    elapsed_time, mean_iou_count, mean_iou_list = get_infertime(mean_iou_list, run, mean_iou_count, batched_input,
                                                                elapsed_time, elapse_time_on=False)
print(" ")
print(" ----------- I running benchmark run!... ----------- ")
for run in range(N_run):
    elapsed_time, mean_iou_count, mean_iou_list = get_infertime(mean_iou_list, run, mean_iou_count, batched_input,

                                                                elapsed_time, elapse_time_on=True)
original_srdout = sys.stdout
with open("print_out_tftrt_efficientdet_batch"+str(batch_size)+"_nrun"+str(N_run)+".txt", 'w') as f:
    sys.stdout = f

    print("Total object detect: ", mean_iou_count, "in", batch_size * N_run, "images")
    print("IoU threshold is:", iou_threshold, "and Predict threshold is:", predict_threshold)
    print("Mean IoU is: {:.3f} ".format(mean_iou_list.sum() / mean_iou_count), "for", batch_size * N_run, "images")
    print("Total elapsed time sum for", N_run * batch_size,
          "images (", N_run, "runs x", batch_size, "batch_size) is {:.2f} sec".format(elapsed_time.sum()))
    print('Throughput: {:.3f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    print("Mean Latency for 1 image is: ", elapsed_time.sum() / (N_run * batch_size))

    sys.stdout = original_srdout
