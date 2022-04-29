from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import numpy as np

batched_input = np.zeros((1 , 512, 512, 3), dtype=np.float32)

print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
    max_workspace_size_bytes=(1 << 30))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(
    maximum_cached_engines=20)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='../default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_saved_model',
    conversion_params=conversion_params)

converter.convert()

def input_fn():
    yield (batched_input, )
converter.build(input_fn=input_fn)

converter.save(output_saved_model_dir='../default_deeplab-edgetpu_models_and_labels/deeplab-edgetpu_xs_saved_model_TFTRT_FP16_1batch')
print('Done Converting to TF-TRT FP16')
