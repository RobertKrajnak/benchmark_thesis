from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import tensorflow as tf

batched_input = np.zeros((1 , 512, 512, 3), dtype=np.float32)
batched_input = tf.constant(batched_input)

print('Converting to TF-TRT INT8...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.INT8, 
    max_workspace_size_bytes=(1 << 28), 
    use_calibration=True)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='../default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_saved_model', 
    conversion_params=conversion_params)

# def calibration_input_fn():
#     yield (1, 224, 224, 3)
# converter.convert(calibration_input_fn=calibration_input_fn)

def calibration_input_fn():
    yield (batched_input, )
converter.convert(calibration_input_fn=calibration_input_fn)

converter.save(output_saved_model_dir='../default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_saved_model_TFTRT_INT8')
print('Done Converting to TF-TRT INT8')
