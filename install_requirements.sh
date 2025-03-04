set -e

git config --global --add safe.directory "*"


pip install --force-reinstall torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install --force-reinstall ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile
pip install transformers

python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git

cd third_party/SAI3D/SemanticSAM/
python -m pip install -r requirements.txt
cd semantic_sam/body/encoder/ops

sh ./make.sh
cd ../../../../
mkdir checkpoints && 
cd checkpoints
wget https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth


cd ../..

pip install segment-anything-hq
pip install git+https://github.com/facebookresearch/segment-anything.git

cd Segmentator
make
cd ../

mkdir -p SAM/checkpoints && 
cd SAM/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..

mkdir -p SAM-HQ/checkpoints && 
cd SAM-HQ/checkpoints
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth

cd ../../..

cd depth-lerf
pip install -e . 
cd ..

# Install nerfstudio dependencies
cd nerfstudio_export
pip install -e . 
cd ../..

pip install h5py==3.8.0
pip install scikit-learn==1.2.2
pip3 install setuptools==68.0.0
pip install tyro==0.5.12
pip3 install numpy==1.26.4

ns-install-cli --mode install


pip3 install Pillow==9.3.0

pip install lpips==0.1.4
pip install timm
