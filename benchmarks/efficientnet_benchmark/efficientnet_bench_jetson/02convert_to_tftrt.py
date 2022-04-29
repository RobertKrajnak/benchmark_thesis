from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# ------------------------------------------------------------------
# for main script run path specific
import sys
import os
os.chdir(os.path.dirname(sys.argv[0]))
# -------------------------------------------------


print('Converting to TF-TRT FP16...')
# conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
#     precision_mode=trt.TrtPrecisionMode.FP16,
#     max_workspace_size_bytes=(11<32))

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
    max_workspace_size_bytes=(1 << 32))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(
    maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='../default_efficientnet_models_and_labels/efficientnetb0_saved_model',
    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='../default_efficientnet_models_and_labels/efficientnetb0_saved_model_TFTRT_FP16')
print('Done Converting to TF-TRT FP16')
