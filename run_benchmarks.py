#!/bin/bash
import socket
import platform
import re
import os
import subprocess
import time
import usb.core

# TODO: PC DL 12min XXNNPACK, 6min modelsaved
# TODO: CORAL OK
# TODO: RPI OK
# TODO: JETSON 170min
# TODO: BBAI 280min

uname = platform.uname()
print(f"System: {uname.system}")
print(f"Node Name: {uname.node}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

# -----------------------------------------------------------
# Image Segmentation parameters (160 samples ADE20K):
# models: DeepLab-edgetpu-xs, DeepLab-edgetpu-m
N_run_seg = 64  # max 160
N_warmup_run_seg = 5
# batch_size is default = 1 / model does not support batch
# -----------------------------------------------------------
# Image Detection parameters (160 samples COCO)
# models: EfficientDet-lite1, EfficientDet-lite3
N_run_detect = 64  # max 160
N_warmup_run_detect = 5
# batch_size is default = 1 / model does not support batch
# -----------------------------------------------------------
# Image Classification parameters (1000 samples ImageNet)
# models: ResNet50, MobileNetV2, EfficientNetb0
N_run_classify = 100  # max 1000
N_warmup_run_classify = 10
batch_size_classify = 10
k_classify = 5
# -----------------------------------------------------------

# check coral is connect
coral_connect = False
dev = usb.core.find(find_all=True)
for cfg in dev:
    if cfg.idVendor == 6766 and cfg.idProduct == 2202:
        coral_connect = True

# -------------------------- Decimal VendorID=6766 & ProductID=2202 for connect USB Coral ------------------------------
if coral_connect == True:
    start_time = time.time()
    print("I DETECT USB CORAL.. I RUNNING TEST FOR USB CORAL..")
    # installation sh file for CORAL
    os.system("bash devices/coral/install.sh")

    # search tflite model ----------------------------- RESNET -------------------------------
    print("1/7 I RUNNING RESNET TEST FOR USB CORAL..")
    if os.path.isfile('benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        subprocess.call("python benchmarks/resnet_benchmark/resnet_bench_coral/01convert_resnet_tflite.py")
    subprocess.call(
        "python benchmarks/resnet_benchmark/resnet_bench_coral/02inference_batchsize_resnet_tflite_loadDelegate.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- MOBILENET -------------------------------
    print("2/7 I RUNNING MOBILENET TEST FOR USB CORAL..")
    if os.path.isfile(
            'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenetv2_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        subprocess.call("python benchmarks/mobilenet_benchmark/mobilenet_bench_coral/01convert_mobilenet_tflite.py")
    subprocess.call(
        "python benchmarks/mobilenet_benchmark/mobilenet_bench_coral/02inference_batchsize_mobilenet_tflite_loadDelegate.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTNET -------------------------------
    print("3/7 I RUNNING EFFICIENTNET TEST FOR USB CORAL..")
    if os.path.isfile(
            'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        subprocess.call(
            "python benchmarks/efficientnet_benchmark/efficientnet_bench_coral/01convert_efficientnet_tflite.py")
    subprocess.call(
        "python benchmarks/efficientnet_benchmark/efficientnet_bench_coral/02inference_bachsize_efficientnet_tflite_loadDelegate.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE3 -------------------------------
    print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR USB CORAL..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_coral/01inference_bachsize_efficientdet_tflite_loadDelegate.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE1 -------------------------------
    print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR USB CORAL..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_coral/01inference_bachsize_efficientdet_tflite_loadDelegate.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- DEEPLAB XS -------------------------------
    print("6/7 I RUNNING DEEPLAB XS TEST FOR USB CORAL..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_coral/01inference_deeplabEdgeTPU_tflite_loadDelegate.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    # search tflite model ----------------------------- DEEPLAB M -------------------------------
    print("7/7 I RUNNING DEEPLAB M TEST FOR USB CORAL..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_m_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_coral/01inference_deeplabEdgeTPU_tflite_loadDelegate.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    end_time = time.time()
    print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")

# --------------------------- for desktop PC specification running ------------------------------------
if uname.processor == "Intel64 Family 6 Model 158 Stepping 10, GenuineIntel" and coral_connect == False:
    start_time = time.time()
    print("I DETECT DESKTOP.. I RUNNING TEST FOR DESKTOP..")
    # installation sh file for PC
    # os.system("bash devices/pc/install.sh")

    # search tflite model ----------------------------- RESNET -------------------------------
    print("1/7 I RUNNING RESNET TEST FOR DESKTOP..")
    if os.path.isfile('benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        subprocess.call("python benchmarks/resnet_benchmark/resnet_bench_pc/01convert_resnet_tflite.py")
    subprocess.call(
        "python benchmarks/resnet_benchmark/resnet_bench_pc/02inference_resnet_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- MOBILENET -------------------------------
    print("2/7 I RUNNING MOBILENET TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenetv2_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        subprocess.call("python benchmarks/mobilenet_benchmark/mobilenet_bench_pc/01convert_mobilenet_tflite.py")
    subprocess.call(
        "python benchmarks/mobilenet_benchmark/mobilenet_bench_pc/02inference_batchsize_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTNET -------------------------------
    print("3/7 I RUNNING EFFICIENTNET TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        subprocess.call(
            "python benchmarks/efficientnet_benchmark/efficientnet_bench_pc/01convert_efficientnet_tflite.py")
    subprocess.call(
        "python benchmarks/efficientnet_benchmark/efficientnet_bench_pc/02inference_efficientnet_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE3 -------------------------------
    print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_pc/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE1 -------------------------------
    print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_pc/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- DEEPLAB XS -------------------------------
    print("6/7 I RUNNING DEEPLAB XS TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_pc/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    # search tflite model ----------------------------- DEEPLAB M -------------------------------
    print("7/7 I RUNNING DEEPLAB M TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_m_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    subprocess.call(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_pc/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    end_time = time.time()
    print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")

# --------------------------------- for RASPBERRY specification running ------------------------------------
if uname.node == "raspberrypi":
    start_time = time.time()
    print("I DETECT RASPBERRY.. I RUNNING TEST FOR RASPBERRY..")
    # installation sh file for RASPBERRY
    # os.system("bash devices/rpi/install_tflite.sh")

    # search tflite model ----------------------------- RESNET -------------------------------
    print("1/7 I RUNNING RESNET TEST FOR RASPBERRY..")
    if os.path.isfile('benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system("python benchmarks/resnet_benchmark/resnet_bench_raspberry/01convert_resnet_tflite.py")
    os.system(
        "python benchmarks/resnet_benchmark/resnet_bench_raspberry/02inference_resnet_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- MOBILENET -------------------------------
    print("2/7 I RUNNING MOBILENET TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenetv2_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system("python benchmarks/mobilenet_benchmark/mobilenet_bench_raspberry/01convert_mobilenet_tflite.py")
    os.system(
        "python benchmarks/mobilenet_benchmark/mobilenet_bench_raspberry/02inference_batchsize_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTNET -------------------------------
    print("3/7 I RUNNING EFFICIENTNET TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system(
            "python benchmarks/efficientnet_benchmark/efficientnet_bench_raspberry/01convert_efficientnet_tflite.py")
    os.system(
        "python benchmarks/efficientnet_benchmark/efficientnet_bench_raspberry/02inference_efficientnet_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE3 -------------------------------
    print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_raspberry/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE1 -------------------------------
    print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_raspberry/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- DEEPLAB XS -------------------------------
    print("6/7 I RUNNING DEEPLAB XS TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_raspberry/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    # search tflite model ----------------------------- DEEPLAB M -------------------------------
    print("7/7 I RUNNING DEEPLAB M TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_m_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_raspberry/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    end_time = time.time()
    print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")

# --------------------------------- for JETSON specification running ------------------------------------
if platform.platform() == "Linux-4.9.253-tegra-aarch64-with-Ubuntu-18.04-bionic":
    start_time = time.time()
    print("I DETECT JETSON.. I RUNNING TEST FOR JETSON..")
    # installation sh file for JETSON
    # os.system("bash devices/jetson/install_tftrt.sh")

    # search TFTRT model ----------------------------- RESNET -------------------------------
    print("1/7 I RUNNING RESNET TEST FOR JETSON..")
    if os.path.isfile(
            'benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_saved_model_TFTRT_FP16/saved_model.pb'):
        print("TFTRT convert model exist")
    else:
        print("TFTRT convert model not exist, I converting model..")
        os.system("python benchmarks/resnet_benchmark/resnet_bench_jetson/02convert_to_tftrt.py")
    os.system(
        "python benchmarks/resnet_benchmark/resnet_bench_jetson/03inference_batchsize_resnet_trt.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search TFTRT model ----------------------------- MOBILENET -------------------------------
    print("2/7 I RUNNING MOBILENET TEST FOR JETSON..")
    if os.path.isfile(
            'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenet_v2_saved_model_TFTRT_FP16/saved_model.pb'):
        print("TFTRT convert model exist")
    else:
        print("TFTRT convert model not exist, I converting model..")
        os.system("python benchmarks/mobilenet_benchmark/mobilenet_bench_jetson/02convert_to_tftrt.py")
    os.system(
        "python benchmarks/mobilenet_benchmark/mobilenet_bench_jetson/03inference_batchsize_mobilenet.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search TFTRT model ----------------------------- EFFICIENTNET -------------------------------
    print("3/7 I RUNNING EFFICIENTNET TEST FOR JETSON..")
    if os.path.isfile(
            'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_saved_model_TFTRT_FP16/saved_model.pb'):
        print("TFTRT convert model exist")
    else:
        print("TFTRT convert model not exist, I converting model..")
        os.system(
            "python benchmarks/efficientnet_benchmark/efficientnet_bench_jetson/02convert_to_tftrt.py")
    os.system(
        "python benchmarks/efficientnet_benchmark/efficientnet_bench_jetson/03inference_batchsize_efficientnet.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search TFTRT model ----------------------------- EFFICIENTDET LITE3 -------------------------------
    print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR JETSON..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_saved_model_TFTRT_FP16/saved_model.pb'):
        print("TFTRT convert model exist")
    else:
        print("I convert model to TFTRT now...")
        os.system(
            "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_jetson/01convert_to_tftrt.py")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_jetson/02inference_batchsize_efficientdet_tftrt.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search TFTRT model ----------------------------- EFFICIENTDET LITE1 -------------------------------
    print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR JETSON..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_saved_model_TFTRT_FP16/saved_model.pb'):
        print("TFTRT convert model exist")
    else:
        print("I convert model to TFTRT now...")
        os.system(
            "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_jetson/01convert_to_tftrt.py")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_jetson/02inference_batchsize_efficientdet_tftrt.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- DEEPLAB XS -------------------------------
    print("6/7 I RUNNING DEEPLAB XS TEST FOR JETSON.. RUN TFLITE MODEL!")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite'):
        print("TFLITE convert model exist")
    else:
        print("TFLITE convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_jetson/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    # search tflite model ----------------------------- DEEPLAB M -------------------------------
    print("7/7 I RUNNING DEEPLAB M TEST FOR JETSON.. RUN TFLITE MODEL!")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_m_model.tflite'):
        print("TFLITE convert model exist")
    else:
        print("TFLITE convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_jetson/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    end_time = time.time()
    print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")

# --------------------------------- for RASPBERRY specification running ------------------------------------
if uname.node == "raspberrypi":
    start_time = time.time()
    print("I DETECT RASPBERRY.. I RUNNING TEST FOR RASPBERRY..")
    # installation sh file for RASPBERRY
    # os.system("bash devices/rpi/install_tflite.sh")

    # search tflite model ----------------------------- RESNET -------------------------------
    print("1/7 I RUNNING RESNET TEST FOR RASPBERRY..")
    if os.path.isfile('benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system("python benchmarks/resnet_benchmark/resnet_bench_raspberry/01convert_resnet_tflite.py")
    os.system(
        "python benchmarks/resnet_benchmark/resnet_bench_raspberry/02inference_resnet_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- MOBILENET -------------------------------
    print("2/7 I RUNNING MOBILENET TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenetv2_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system("python benchmarks/mobilenet_benchmark/mobilenet_bench_raspberry/01convert_mobilenet_tflite.py")
    os.system(
        "python benchmarks/mobilenet_benchmark/mobilenet_bench_raspberry/02inference_batchsize_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTNET -------------------------------
    print("3/7 I RUNNING EFFICIENTNET TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system(
            "python benchmarks/efficientnet_benchmark/efficientnet_bench_raspberry/01convert_efficientnet_tflite.py")
    os.system(
        "python benchmarks/efficientnet_benchmark/efficientnet_bench_raspberry/02inference_efficientnet_tf.lite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE3 -------------------------------
    print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_raspberry/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE1 -------------------------------
    print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_raspberry/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- DEEPLAB XS -------------------------------
    print("6/7 I RUNNING DEEPLAB XS TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_raspberry/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    # search tflite model ----------------------------- DEEPLAB M -------------------------------
    print("7/7 I RUNNING DEEPLAB M TEST FOR RASPBERRY..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_m_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_raspberry/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    end_time = time.time()
    print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")

# --------------------------------- for BEAGLEBONEAI specification running ------------------------------------
if uname.node == "beaglebone":
    start_time = time.time()
    print("I DETECT BEAGLEBONEAI.. I RUNNING TEST FOR BEAGLEBONEAI..")
    # installation sh file for BEAGLEBONEAI
    # os.system("bash devices/beaglebone/install_tflite.sh")

    # search tflite model ----------------------------- RESNET -------------------------------
    print("1/7 I RUNNING RESNET TEST FOR BEAGLEBONEAI..")
    if os.path.isfile('benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system("python3 benchmarks/resnet_benchmark/resnet_bench_beaglebone/01convert_resnet_tflite.py")
    os.system(
        "python3 benchmarks/resnet_benchmark/resnet_bench_beaglebone/02inference_resnet_tflite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- MOBILENET -------------------------------
    print("2/7 I RUNNING MOBILENET TEST FOR BEAGLEBONEAI..")
    if os.path.isfile(
            'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenetv2_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system("python3 benchmarks/mobilenet_benchmark/mobilenet_bench_beaglebone/01convert_mobilenet_tflite.py")
    os.system(
        "python3 benchmarks/mobilenet_benchmark/mobilenet_bench_beaglebone/02inference_batchsize_tflite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTNET -------------------------------
    print("3/7 I RUNNING EFFICIENTNET TEST FOR BEAGLEBONEAI..")
    if os.path.isfile(
            'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, I converting model..")
        os.system(
            "python3 benchmarks/efficientnet_benchmark/efficientnet_bench_beaglebone/01convert_efficientnet_tflite.py")
    os.system(
        "python3 benchmarks/efficientnet_benchmark/efficientnet_bench_beaglebone/02inference_efficientnet_tflite.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE3 -------------------------------
    print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR BEAGLEBONEAI..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python3 benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_beaglebone/01inference_bachsize_efficientdet_tflite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE1 -------------------------------
    print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR BEAGLEBONEAI..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python3 benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_baglebone/01inference_bachsize_efficientdet_tflite.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- DEEPLAB XS -------------------------------
    print("6/7 I RUNNING DEEPLAB XS TEST FOR BEAGLEBONEAI..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python3 benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_beaglebone/01inference_deeplabEdgeTPU_tflite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    # search tflite model ----------------------------- DEEPLAB M -------------------------------
    print("7/7 I RUNNING DEEPLAB M TEST FOR BEAGLEBONEAI..")
    if os.path.isfile(
            'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_m_model.tflite'):
        print("Tflite convert model exist")
    else:
        print("Tflite convert model not exist, You must download from tf hub..")
    os.system(
        "python3 benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_beaglebone/01inference_deeplabEdgeTPU_tflite.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    end_time = time.time()
    print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")

# --------------------------- for desktop PC specification running ------------------------------------
# if uname.node == "deeplearningtitan-MS-7C02" and coral_connect == False:
#     start_time = time.time()
#     print("I DETECT DEEP LEARNING DESKTOP.. I RUNNING TEST FOR DEEP LEARNING DESKTOP..")
#     # installation sh file for PC
#     # os.system("bash devices/pc/install.sh")
#
#     # search tflite model ----------------------------- RESNET -------------------------------
#     print("1/7 I RUNNING RESNET TEST FOR DESKTOP..")
#     if os.path.isfile('benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_model_FP16.tflite'):
#         print("Tflite convert model exist")
#     else:
#         print("Tflite convert model not exist, I converting model..")
#         os.system("python benchmarks/resnet_benchmark/resnet_bench_pc/01convert_resnet_tflite.py")
#     os.system(
#         "python benchmarks/resnet_benchmark/resnet_bench_pc/02inference_resnet_tf.lite.py --N_run " + str(
#             N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
#             batch_size_classify) + " --top_k " + str(
#             k_classify) + "")
#
#     # search tflite model ----------------------------- MOBILENET -------------------------------
#     print("2/7 I RUNNING MOBILENET TEST FOR DESKTOP..")
#     if os.path.isfile(
#             'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenetv2_model_FP16.tflite'):
#         print("Tflite convert model exist")
#     else:
#         print("Tflite convert model not exist, I converting model..")
#         os.system("python benchmarks/mobilenet_benchmark/mobilenet_bench_pc/01convert_mobilenet_tflite.py")
#     os.system(
#         "python benchmarks/mobilenet_benchmark/mobilenet_bench_pc/02inference_batchsize_tf.lite.py --N_run " + str(
#             N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
#             batch_size_classify) + " --top_k " + str(
#             k_classify) + "")
#
#     # search tflite model ----------------------------- EFFICIENTNET -------------------------------
#     print("3/7 I RUNNING EFFICIENTNET TEST FOR DESKTOP..")
#     if os.path.isfile(
#             'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_model_FP16.tflite'):
#         print("Tflite convert model exist")
#     else:
#         print("Tflite convert model not exist, I converting model..")
#         os.system(
#             "python benchmarks/efficientnet_benchmark/efficientnet_bench_pc/01convert_efficientnet_tflite.py")
#     os.system(
#         "python benchmarks/efficientnet_benchmark/efficientnet_bench_pc/02inference_efficientnet_tf.lite.py --N_run " + str(
#             N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
#             batch_size_classify) + " --top_k " + str(
#             k_classify) + "")
#
#     # search tflite model ----------------------------- EFFICIENTDET LITE3 -------------------------------
#     print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR DESKTOP..")
#     if os.path.isfile(
#             'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_model.tflite'):
#         print("Tflite convert model exist")
#     else:
#         print("Tflite convert model not exist, You must download from tf hub..")
#     os.system(
#         "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_pc/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
#             N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")
#
#     # search tflite model ----------------------------- EFFICIENTDET LITE1 -------------------------------
#     print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR DESKTOP..")
#     if os.path.isfile(
#             'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_model.tflite'):
#         print("Tflite convert model exist")
#     else:
#         print("Tflite convert model not exist, You must download from tf hub..")
#     os.system(
#         "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_pc/01inference_bachsize_efficientdet_tf.lite.py --N_run " + str(
#             N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")
#
#     # search tflite model ----------------------------- DEEPLAB XS -------------------------------
#     print("6/7 I RUNNING DEEPLAB XS TEST FOR DESKTOP..")
#     if os.path.isfile(
#             'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_model.tflite'):
#         print("Tflite convert model exist")
#     else:
#         print("Tflite convert model not exist, You must download from tf hub..")
#     os.system(
#         "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_pc/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
#             N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")
#
#     # search tflite model ----------------------------- DEEPLAB M -------------------------------
#     print("7/7 I RUNNING DEEPLAB M TEST FOR DESKTOP..")
#     if os.path.isfile(
#             'benchmarks/deeplab-edgetpu_benchmark/default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_m_model.tflite'):
#         print("Tflite convert model exist")
#     else:
#         print("Tflite convert model not exist, You must download from tf hub..")
#     os.system(
#         "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_pc/01inference_deeplabEdgeTPU_tf.lite.py --N_run " + str(
#             N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")
#
#     end_time = time.time()
#     print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")


if uname.node == "deeplearningtitan-MS-7C02" and coral_connect == False:
    start_time = time.time()
    print("I DETECT DEEP LEARNING DESKTOP.. I RUNNING TEST FOR DEEP LEARNING DESKTOP SavedModel..")
    # installation sh file for PC
    # os.system("bash devices/pc/install.sh")

    # search tflite model ----------------------------- RESNET -------------------------------
    print("1/7 I RUNNING RESNET TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/resnet_benchmark/default_resnet_models_and_labels/resnet50_saved_model/saved_model.pb'):
        print("SavedModel convert model exist")
    else:
        print("SavedModel convert model not exist, You must download from hub..")
    os.system(
        "python benchmarks/resnet_benchmark/resnet_bench_pc/02inference_model_predict.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- MOBILENET -------------------------------
    print("2/7 I RUNNING MOBILENET TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/mobilenet_benchmark/default_mobilenet_models_and_labels/mobilenet_v2_saved_model/saved_model.pb'):
        print("SavedModel convert model exist")
    else:
        print("SavedModel convert model not exist, You must download from hub..")
    os.system(
        "python benchmarks/mobilenet_benchmark/mobilenet_bench_pc/02inference_model_predict.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTNET -------------------------------
    print("3/7 I RUNNING EFFICIENTNET TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/efficientnet_benchmark/default_efficientnet_models_and_labels/efficientnetb0_saved_model/saved_model.pb'):
        print("SavedModel convert model exist")
    else:
        print("SavedModel convert model not exist, You must download from hub..")
    os.system(
        "python benchmarks/efficientnet_benchmark/efficientnet_bench_pc/02inference_model_predict.py --N_run " + str(
            N_run_classify) + " --N_warmup_run " + str(N_warmup_run_classify) + " --batch_size " + str(
            batch_size_classify) + " --top_k " + str(
            k_classify) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE3 -------------------------------
    print("4/7 I RUNNING EFFICIENTDET LITE3 TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite3_saved_model/saved_model.pb'):
        print("SavedModel convert model exist")
    else:
        print("SavedModel convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite3_bench_pc/01inference_model_predict.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- EFFICIENTDET LITE1 -------------------------------
    print("5/7 I RUNNING EFFICIENTDET LITE1 TEST FOR DESKTOP..")
    if os.path.isfile(
            'benchmarks/efficientdet_benchmark/default_efficientdet_models_and_labels/efficientdet_lite1_saved_model/saved_model.pb'):
        print("SavedModel convert model exist")
    else:
        print("SavedModel convert model not exist, You must download from tf hub..")
    os.system(
        "python benchmarks/efficientdet_benchmark/efficientdet-lite1_bench_pc/01inference_model_predict.py --N_run " + str(
            N_run_detect) + " --N_warmup_run " + str(N_warmup_run_detect) + "")

    # search tflite model ----------------------------- DEEPLAB XS -------------------------------
    print("6/7 I RUNNING DEEPLAB XS TEST FOR DESKTOP..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-xs_bench_pc/01inference_model_predict.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    # search tflite model ----------------------------- DEEPLAB M -------------------------------
    print("7/7 I RUNNING DEEPLAB M TEST FOR DESKTOP..")
    os.system(
        "python benchmarks/deeplab-edgetpu_benchmark/deeplab-edgetpu-m_bench_pc/01inference_model_predict.py --N_run " + str(
            N_run_seg) + " --N_warmup_run " + str(N_warmup_run_seg) + "")

    end_time = time.time()
    print("Time for all test with print, preprocess, initial... is : ", (end_time - start_time) / 60, "min")
