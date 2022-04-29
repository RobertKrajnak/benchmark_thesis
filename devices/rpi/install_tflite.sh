# Raspberry PI 3 Model B 1.2

mkdir benchmark_thesis
cd benchmark_thesis
sudo pip3 install python3-numpy python3-dev python3-pip python3-mock

sudo pip3 install virtualenv
virtualenv rpi_benchmark_venv -p python3
source ./rpi_benchmark_venv/bin/activate

# tflite runtime - dont work:
#echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
#curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
#sudo apt-get update
#sudo apt-get install python3-tflite-runtime

# tnsorflow:
#https://qengineering.eu/install-tensorflow-2.1.0-on-raspberry-pi-4.html

# swapfile:
sudo dpyhs -swapfile swapoff
sudo nano  /etc/dpyhs-swapfile
#<CONF_SWAPSIZE=2048>
#(Ctrl + x)
#(Y )= save
#(ENTER) to exit
sudo dpyhs -swapfile swapon
sudo reboot

sudo pip3 install pillow
python -m pip install pyusb
python -m pip install scikit-learn
pip install numpy --upgrade