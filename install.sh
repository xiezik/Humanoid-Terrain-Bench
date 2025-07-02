conda create -n terrain python=3.8
conda activate terrain
cd
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

git clone https://github.com/shiki-ta/Humanoid-Terrain-Bench.git
cd extreme-parkour
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd ~/Humanoid-Terrain-Bench/rsl_rl && pip install -e .
cd ~/Humanoid-Terrain-Bench/legged_gym && pip install -e .
cd ~/Humanoid-Terrain-Bench/challenging_terrain && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask