# tflite wheel:
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime

pip3 install pyusb
pip3 install pillow
pip3 install scikit-learn
pip3 install numpy --upgrade

# grow SD partition space:
sudo /opt/scripts/tools/grow_partition.sh

# swapfile:
sudo mkdir -p /var/cache/swap/   
sudo dd if=/dev/zero of=/var/cache/swap/swapfile bs=2048k count=2000
sudo chmod 0600 /var/cache/swap/swapfile 
sudo mkswap /var/cache/swap/swapfile 
# for each run OS when you can use swap:
sudo swapon /var/cache/swap/swapfile

# ----------------------------------------------------------------------------------------------
# Give all DRAM to Linux: https://www.glennklockwood.com/embedded/beaglebone-ai.html
# ONLY - if you dont use DPS and AVE cores:

#First check your kernel version:
#
#$ uname -r
#4.14.108-ti-r143
#Then edit the device tree that's loaded on boot, making sure to edit the correct version that matches our kernel (4.14 in our case):
#
#$ cd /opt/source/dtb-4.14-ti
#$ git pull
#$ vi src/arm/am5729-beagleboneai.dts
#Scroll down a bit and at around line 23 you'll see a block that begins with reserved-memory {:
#
#/*
#    reserved-memory {
#        #address-cells = <2>;
#        #size-cells = <2>;
#...
#        cmem_block_mem_0: cmem_block_mem@a0000000 {
#            reg = <0x0 0xa0000000 0x0 0x18000000>;
#            no-map;
#            status = "okay";
#        };
#    };
#*/
#Comment this out using C-style comments (/* ... */). Also comment out the following cmem { block because it references cmem memory blocks defined in the reserved-memory block we just disabled above:
#
#/*
#    cmem {
#        compatible = "ti,cmem";
#        #address-cells = <1>;
#
#...
#
#        cmem_block_1: cmem_block@1 {
#            reg = <1>;
#            memory-region = <&cmem_block_mem_1_ocmc3>;
#        };
#    };
#*/
#Finally, comment out the bits that reference these disabled memory regions. There are two for the IPUs (the Cortex-M4s which control the EVEs) and two for the DSPs (C66x):
#
#/*
#&ipu1 {
#        status = "okay";
#        memory-region = <&ipu1_memory_region>;
#};
#
#&ipu2 {
#        status = "okay";
#        memory-region = <&ipu2_memory_region>;
#};
#
#&dsp1 {
#        status = "okay";
#        memory-region = <&dsp1_memory_region>;
#};
#
#&dsp2 {
#        status = "okay";
#        memory-region = <&dsp2_memory_region>;
#};
#*/
#Then back in /opt/source/dtb-4.14-ti (or whatever directory matches your kernel), run make as the debian user:
#
#$ make
#...
#  DTC     src/arm/am5729-beagleboneai.dtb
#Then install the rebuilt device tree:
#
#$ sudo make install
#...
#'src/arm/am5729-beagleboneai.dtb' -> '/boot/dtbs/4.14.108-ti-r143/am5729-beagleboneai.dtb'
#
#After this, cross your fingers and sudo reboot. Once the system comes back up, you should see almost the full 1 GiB now:
#
#$ free -m
#              total        used        free      shared  buff/cache   available
#Mem:            993          51         865           5          76         913
#Swap:             0           0           0