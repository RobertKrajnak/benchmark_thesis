conda create --name benchmark_thesis python=3.7
conda activate benchmark_thesis
# pre linux - cuda + tf: -------------------
conda install -c conda-forge tensorflow-gpu
# -------------------------------------------
# pre win - cuda:----------------------------
# https://youtu.be/OEFKlRSd8Ic
# https://www.tensorflow.org/install/gpu
pip install tensorflow
# -------------------------------------------
pip install tflite_runtime
pip install pillow
pip install opencv-python
pip install -U scikit-learn
pip install pyusb