import numpy as np
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32
print(target_dtype)
import os, time
import tensorrt as trt
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
from skimage.transform import resize
from skimage import io

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

BATCH_SIZE = 16 # inference batch

img='n01443537_11099_goldfish.jpg'
# img = resize(io.imread(img), (224, 224))
# input_batch = 255*np.array(np.repeat(np.expand_dims(np.array(img, dtype=np.float16), axis=0), BATCH_SIZE, axis=0), dtype=np.float16)
# input_batch = input_batch.astype(target_dtype)

img = resize(io.imread(img), (224, 224))
input_batch = image.img_to_array(img)
input_batch = np.expand_dims(input_batch, axis=0)
input_batch = preprocess_input(input_batch)
input_batch = input_batch.astype(target_dtype)

f = open("mobilenetv2-7_engine.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) # Need to set output dtype to FP16 to enable FP16
# Allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    return output


print("Warming up...")
trt_predictions = predict(input_batch).astype(np.float32)
print("Done warming up!")


indices = (-trt_predictions[0]).argsort()[:5]
print("Class | Probability (out of 1)")
print(list(zip(indices, trt_predictions[0][indices])))


# %%timeit
print(predict(input_batch))

