# Jetson nano

sudo apt-get install virtualenv
virtualenv jetson_benchmark_venv -p python3
virtualenv jetson_benchmark_tf2_venv -p python3
source ./jetson_benchmark_venv/bin/activate

sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

# sudo apt-get install python3-pip
python3 -m pip install -U pip testresources setuptools==49.6.0 

python3 -m pip install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
# sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0 
# env H5PY_SETUP_REQUIRES=0 python3 -m pip install -U h5py==3.1.0
env H5PY_SETUP_REQUIRES=0 python3 -m pip install -U h5py==2.8.0 # h5py=3.1.0 nesla nainstalovat, 2.10.0 ide tiez

python3 -m pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
#python3 -m pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 'tensorflow<2'

python3 -m pip install pillow
python3 -m pip install scikit-learn
python -m pip install pyusb

# swapfile for more memory
git clone https://github.com/JetsonHacksNano/installSwapfile
cd installSwapfile
./installSwapfile.sh
sudo reboot now

# for use scikit-learn - jaccard on deeplab:
# issue https://forums.developer.nvidia.com/t/error-importerror-usr-lib-aarch64-linux-gnu-libgomp-so-1-cannot-allocate-memory-in-static-tls-block-i-looked-through-available-threads-already/166494/3
python -m pip install scikit-learn==0.22 # because python 3.6.9
source ~/.bashrc
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
source ./jetson_benchmark_venv_tf2/bin/activate

# cv2
# get https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-5-5.sh
# sudo chmod 755 ./OpenCV-4-5-5.sh
# ./OpenCV-4-5-5.sh
# rm OpenCV-4-5-5.sh









