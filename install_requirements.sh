set -e

git config --global --add safe.directory "*"

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDA_HOST_COMPILER=/usr/bin/gcc-11


pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install numpy==1.26.4
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile


cd third_party/depthlerf
pip install -e . 
cd ..

cd lerf
pip install -e . 
cd ..

# Install nerfstudio dependencies
cd nerfstudio_export
pip install -e . 
cd ../..

pip install nerfacc

pip install h5py==3.8.0
pip install scikit-learn==1.2.2
pip3 install setuptools==68.0.0
pip install tyro==0.5.12
pip3 install numpy==1.26.4

ns-install-cli --mode install


pip3 install Pillow==9.3.0

pip install lpips==0.1.4
pip install timm

pip install -U gradio
