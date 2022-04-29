import numpy as np
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32
print(target_dtype)
from PIL import Image
import os, time
import tensorrt as trt
import onnx

BATCH_SIZE = 32 # inference batch

# Preprocessing: load the ONNX model
model_path = '/home/deep/Documents/benchmark_thesis/benchmarks/mobilenet_bench_jetson_tensorRT_noOK/resnet50_v1.onnx'
onnx_model = onnx.load(model_path)

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')


if USE_FP16:
	os.system('/usr/src/tensorrt/bin/trtexec --onnx=resnet50_v1.onnx --saveEngine=resnet50_v1_engine.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16')
else:
	os.system('/usr/src/tensorrt/bin/trtexec --onnx=resnet50_v1.onnx --saveEngine=resnet50_v1_engine.trt  --explicitBatch')

os.system('ls -la')
print("trt file done!")

