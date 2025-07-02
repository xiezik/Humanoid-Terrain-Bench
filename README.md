# Humanoid Terrain Bench

### Installation ###
```bash
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
```

### Usage ###
`cd legged_gym/scripts`
1. Train base policy:  
```
python train.py --exptid h1-2 --device cuda:0
```

2. Training Recovery:
```
python train.py --exptid h1-2 --device cuda:0 --resume --resumeid=test --checkpoint=50000
```

3. Play base policy:
```
python play.py --exptid test
```

### Arguments
- --exptid: string,  to describe the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.

