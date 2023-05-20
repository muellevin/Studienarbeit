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
# installing example images for testing porpuse
cd Detection_training/Tensorflow/workspace
git clone https://github.com/muellevin/devset.git images
cd ~/

# sometimes error
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
# for tensorflow
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran -y
sudo apt-get install python3-pip -y
sudo python3 -m pip install --upgrade pip
sudo -H pip3 install testresources setuptools
# numpy .5 -> core dumb
sudo -H pip3 install numpy==1.19.4 future mock gast protobuf pybind11 cython pkgconfig packaging
# h5py needs something special
sudo -H H5PY_SETUP_REQUIRES=0 pip3 install -U --no-build-isolation h5py==3.1.0
sudo -H pip3 install keras_preprocessing keras_applications
sudo -H pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow==2.7.0+nv22.1

# install pytorch
sudo apt-get -y install libomp5 libomp-dev
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5+nv22.01-cp36-cp36m-linux_aarch64.whl
pip3 install --upgrade pip
python3 -m pip install aiohttp scipy=='1.5.3'
export LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH
pip3 install --no-cache $TORCH_INSTALL
# pip3 install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu102 or -f https://download.pytorch.org/whl/cu102/torch_stable.html

# install opencv
sudo apt-get install python3-opencv -y
# and now remove to get newest version?!
sudo apt-get remove python3-opencv -y
# warning/error
sudo apt-get install libcanberra-gtk-module -y
# to install newest version with cuda support follow:
# https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html
# after this script is finished

# install coral usb accelerator
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
# make sure curl is installed
sudo apt install curl -y
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update -y
# sudo apt-get install libedgetpu1-std
sudo apt-get install libedgetpu1-max -y # using max frequency
# install pycoral
sudo apt-get install python3-pycoral -y

# install pycuda (normal installation has errors) this for some reasons now aswell
# export PATH=/usr/local/cuda/bin:$PATH
# export CUDA_ROOT=/usr/local/cuda
# sudo -H pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
# now try this:
# set -e

# if ! which nvcc > /dev/null; then
#   echo "ERROR: nvcc not found"
#   exit
# fi

# arch=$(uname -m)
# folder=${HOME}/src
# mkdir -p $folder

# echo "** Install requirements"
# sudo apt-get install -y build-essential python3-dev python3-pip
# sudo apt-get install -y libboost-python-dev libboost-thread-dev
# sudo pip3 install setuptools

# boost_pylib=$(basename /usr/lib/$(arch)-linux-gnu/libboost_python3.so)
# boost_pylibname=${boost_pylib%.so}
# boost_pyname=${boost_pylibname/lib/}

# echo "** Download pycuda-2019.1.2 sources"
# pushd $folder
# if [ ! -f pycuda-2019.1.2.tar.gz ]; then
#   wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
# fi

# echo "** Build and install pycuda-2019.1.2"
# CPU_CORES=$(nproc)
# echo "** cpu cores available: " $CPU_CORES
# tar xzvf pycuda-2019.1.2.tar.gz
# cd pycuda-2019.1.2
# python3 ./configure.py --python-exe=/usr/bin/python3 --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib/${arch}-linux-gnu --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib/${arch}-linux-gnu --boost-python-libname=${boost_pyname} --boost-thread-libname=boost_thread
# make -j$CPU_CORES
# python3 setup.py build
# sudo python3 setup.py install

# popd

# python3 -c "import pycuda; print('pycuda version:', pycuda.VERSION)"
# somehow dont ask my why oh oh oh
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip3 install pycuda --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64"

# jeston stats (nice to have)
sudo -H pip3 install -U jetson-stats
echo "Yay i should be finished now"
echo "If you want newest opencv version with cuda support follow that guide: https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html"
echo "I am going to reboot in 1 min"
# reboot for jtop
sudo reboot
