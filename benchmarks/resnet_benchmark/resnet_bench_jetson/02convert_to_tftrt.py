from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf

# ------------------------------------------------------------------
# for main script run path specific
import os
import sys
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------


# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# tf.config.experimental.set_virtual_device_configuration( gpu_devices[0],
#            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=128)]) ## Crucial value, set lower than available GPU memory (note that Jetson shares GPU memory with CPU)

print('Converting to TF-TRT FP16...')
# conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
#     precision_mode=trt.TrtPrecisionMode.FP16,
#     max_workspace_size_bytes=(11<32)) # 1<<32 

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 36))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(maximum_cached_engines=200)

converter = trt.TrtGraphConverterV2(input_saved_model_dir='../default_resnet_models_and_labels/resnet50_saved_model',
                                    conversion_params=conversion_params)
print('Converting now...')
converter.convert()
print('Done Converting to TF-TRT FP16')

converter.save(output_saved_model_dir='../default_resnet_models_and_labels/resnet50_saved_model_TFTRT_FP16')
print('Done all')
