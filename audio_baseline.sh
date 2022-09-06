#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH -p gpu_rtx
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module purge
source /home/wim17006/miniconda3/etc/profile.d/conda.sh
conda activate ptaudio

python3 audio_baseline.py custom --model wrn 

