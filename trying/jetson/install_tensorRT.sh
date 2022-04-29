# Jetson nano

# tensort rt sa nainstaluje uz ako sucast jetpacku teda nachadza sa v userovi defaulte

# sudo apt-get install virtualenv
# virtualenv jetson_benchmark_tensor_venv -p python3
# source ./jetson_benchmark_tensor_venv/bin/activate

# ------- tensorflow ----------

sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools==49.6.0 

sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0

# sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
python -m pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 'tensorflow<2'

python3 -m pip install pillow
python3 -m pip install scikit-image
python3 -m pip install Pillow

# --------------------------- TesnorRT - ONNX optimalization ---------------------------------
#sudo apt-get install protobuf-compiler libprotoc-dev # inak neslo zinstalovat onnx
#python3 -m pip install tf2onnx
#python3 -m pip install keras2onnx
#
## instalacia kvoli pycude
#sudo apt-get install build-essential autoconf libtool pkg-config python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev
#sudo easy_install greenlet
#sudo easy_install gevent
#python3 -m pip install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
#python3 -m pip install pycuda
## ------------- nepomohlo











