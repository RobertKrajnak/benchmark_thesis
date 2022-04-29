# save model:
# https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1

from PIL import Image
import numpy as np
import time
import os
import tflite_runtime.interpreter as tflite
from PIL import Image
import glob
import json
# import cv2
import re
import sys

# ------------------------------------------------------------------
# for main script run path specific
import os
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------

# -----------------------------------
predict_threshold = 0.6
iou_threshold = 0.6
# -------------------------------------
N_run = 64
batch_size = 1  # batchsize pri tomto modely neefunguje => nechat 1
#   N_warmup_run < N_run!
N_warmup_run = 5


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
interpreter = tflite.Interpreter(
    model_path="../default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite")
image_path = r'../../../data/Coco2017_val_samples/*.jpg'

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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


def get_infertime(mean_iou_list, run, mean_iou_count, batched_input, input_details, output_details, elapsed_time,
                  elapse_time_on):
    print("-------- N_run number: ", run, "------------")

    for batch in range(batch_size):

        #  --------------------- my preprocessing ---------------------------
        image = Image.open(filename_list_split[run][batch]).convert('RGB')
        img = image.resize((512, 512))
        # img = np.array(img)
        # img = img[:, :, ::-1]
        b, g, r = img.split()
        img = Image.merge("RGB", (r, g, b))
        img = np.expand_dims(img, axis=0)
        batched_input[batch, :] = img
        print(filename_list_split[run][batch])

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
        print("elapsed time for run is: ", elapsed_time, "\n")
    else:
        print("Start Warm Up infer...")
        interpreter.invoke()
        print("End Warm Up infer...\n")

    # # -------------------- Interpreting output tensor for object detect --------------
    output_details = interpreter.get_output_details()

    detection_boxes = interpreter.get_tensor(output_details[0]['index'])
    detection_classes = interpreter.get_tensor(output_details[1]['index'])
    detection_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = interpreter.get_tensor(output_details[3]['index'])

    for j in range(batch_size):
        for i in range(int(num_boxes)):
            if detection_scores[0, i] > predict_threshold and elapse_time_on == True:
                classes = detection_classes[0, i]
                boxes = detection_boxes[0, i]
                id_image = []

                # ak chceme plotovat
                # image = cv2.imread(filename_list_split[run][batch])
                # imH, imW, _ = image.shape
                # print("image.shape: ", image.shape)

                # ak nechceme plotovat
                imW, imH = image.size
                print("image.size: ", image.size)

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

                        # # modry bbox je GT
                        # cv2.rectangle(image, (int(bbox_GT[0]), int(bbox_GT[1])),
                        #               (int(bbox_GT[2] + bbox_GT[0]), int(bbox_GT[3] + bbox_GT[1])),
                        #               (255, 0, 0), 2)
                        # # zlty bbox je predict
                        # cv2.rectangle(image, (int(max(1, (boxes[1] * imW))), int(max(1, (boxes[0] * imH)))),
                        #               (int(min(imW, (boxes[3] * imW))), int(min(imH, (boxes[2] * imH)))),
                        #               (10, 255, 255), 2)
                        # cv2.imshow('image', image)
                        # cv2.waitKey(0)

                        bbox_GT = [bbox_GT[0], bbox_GT[1], bbox_GT[2] + bbox_GT[0], bbox_GT[3] + bbox_GT[1]]
                        bbox_predict = [round(boxes[1] * imW, 2), round(boxes[0] * imH, 2), round(boxes[3] * imW, 2),
                                        round(boxes[2] * imH, 2)]

                        print("bbox ground truth: ", bbox_GT)
                        print("bbox predict: ", bbox_predict)

                        file_class = open("../default_efficientdet_models_and_labels/image_info_test2017.json")
                        label_json = json.load(file_class)
                        for j in label_json['categories']:
                            if j['id'] == int(classes) + 1:
                                print("class:", classes + 1, " class name:", j['name'], " score:",
                                      detection_scores[0, i],
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
                                                                input_details,
                                                                output_details,
                                                                elapsed_time, elapse_time_on=False)
print(" ")
print(" ----------- I running benchmark run!... ----------- ")
for run in range(N_run):
    elapsed_time, mean_iou_count, mean_iou_list = get_infertime(mean_iou_list, run, mean_iou_count, batched_input,
                                                                input_details,
                                                                output_details,
                                                                elapsed_time, elapse_time_on=True)

original_srdout = sys.stdout
with open("print_out_tflite_efficientdet_batch"+str(batch_size)+"_nrun"+str(N_run)+".txt", 'w') as f:
    sys.stdout = f

    print("Total object detect: ", mean_iou_count, "in", batch_size * N_run, "images")
    print("IoU threshold is:", iou_threshold, "and Predict threshold is:", predict_threshold)
    print("Mean IoU is: {:.3f} ".format(mean_iou_list.sum() / mean_iou_count), "for", batch_size * N_run, "images")
    print("Total elapsed time sum for", N_run * batch_size,
          "images (", N_run, "runs x", batch_size, "batch_size) is {:.2f} sec".format(elapsed_time.sum()))
    print('Throughput: {:.3f} images/s'.format(N_run * batch_size / elapsed_time.sum()))
    print("Mean Latency for 1 image is: ", elapsed_time.sum() / (N_run * batch_size))

    sys.stdout = original_srdout
