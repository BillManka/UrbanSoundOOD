#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --partition gpu_rtx
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module purge
# module load cudnn/7.6.5 cuda/10.1
module load cudnn/7.6.5 cuda/11.4-rhel7

source /home/wim17006/miniconda3/etc/profile.d/conda.sh
conda activate ptaudio

python3 audio_test.py -m custom_valerdo_baseline
