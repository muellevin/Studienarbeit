# normal setup
sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt dist-upgrade -y

# for not vim user:
sudo apt-get install nano -y

# clone the repo
git clone https://github.com/muellevin/Studienarbeit.git
cd Studienarbeit
git checkout develop

# for tensorflow
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran -y
sudo apt-get install python3-pip -y
sudo python3 -m pip install --upgrade pip
sudo -H pip3 install testresources setuptools
# numpy .5 -> core dumb
sudo -H pip3 install numpy==1.19.4 future mock keras_preprocessing keras_applications gast protobuf pybind11 cython pkgconfig packaging h5py -y
sudo -H pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow==2.7.0+nv22.1

# install opencv
sudo apt-get install python3-opencv
# and now remove to get newest version?!
sudo apt-get remove python3-opencv
# warning/error
sudo apt-get install libcanberra-gtk-module -y

# install coral usb accelerator
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
# make sure curl is installed
sudo apt install curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
# sudo apt-get install libedgetpu1-std
sudo apt-get install libedgetpu1-max # using max frequency
# install pycoral
sudo apt-get install python3-pycoral

# install pycuda (normal installation has errors)
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_ROOT=/usr/local/cuda
sudo -H pip3 install pycuda

# jeston stats (nice to have)
sudo -H pip3 install -U jetson-stats
# reboot for jtop
sudo reboot now
