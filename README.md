# OpenLeRF3D
OpenLerf3D: Open-Vocabulary 3D Instance Segmentation with SAI3D and LeRF Guidance

![OpenLerf3D_Model](https://i.ibb.co/MDrzKZz1/openlerf3d-pipeline.png)

## Installation

Create a conda enviroment 

```bash
conda create --name openlerf3d -y python=3.10
conda activate openlerf3d
pip install --upgrade pip
```

Install requirements

```bash
bash install_requirements.sh
```


### Replica Dataset
Download the Replica dataset pre-processed by [NICE-SLAM](https://pengsongyou.github.io/nice-slam) and transform it into [nerfstudio](https://docs.nerf.studio) format using these steps:
```
cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
cd ..
python datasets/replica_preprocess.py
