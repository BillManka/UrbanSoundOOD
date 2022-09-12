#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --partition gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module purge
source /home/wim17006/miniconda3/etc/profile.d/conda.sh
conda activate ptaudio
module load cudnn/7.6.5 cuda/10.1

python3 audio_baseline.py custom --model valerdo  --learning_rate 0.001 --epochs 50


