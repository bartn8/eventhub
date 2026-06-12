# Activate the venv environment
USERNAME="$(whoami)"
CONDA_PATH="/home/${USERNAME}/miniconda3/bin/conda"
CONDA_ENV=eventhub
eval "$( $CONDA_PATH shell.bash hook)"

conda create -n $CONDA_ENV python=3.10 -y
conda activate $CONDA_ENV

pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

pip install einops h5py hdf5plugin matplotlib numba numpy opencv-python opt-einsum pandas pillow scipy seaborn setuptools tqdm typing-extensions yacs natsort argparse imageio imageio-ffmpeg scikit-image plyfile shapely trimesh open3d gpytoolbox lpips pytorch-msssim

pip install kornia timm easydict thop tensorboard

# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
#https://github.com/rahul-goel/fused-ssim/

# module load cuda/12.6;module load ninja;module load gcc
# CUDA_ARCHITECTURES="80;100" for A100-5090
conda install nvidia::cuda-toolkit==12.8.1 nvidia::cuda-nvcc==12.8.93
CUDA_ARCHITECTURES="80;90" pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install torch-scatter --no-build-isolation
#TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install ./event-stereo/src/components/models/baseline/deform_conv --no-build-isolation
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install ./rpg_vid2e/esim_torch/ --no-build-isolation
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install -e cuda/ --no-build-isolation