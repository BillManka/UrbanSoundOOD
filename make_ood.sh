#!/bin/bash

#SBATCH -n1 -N20
#SBATCH --partition generalsky
#SBATCH --time=02:00:00

module purge
source /home/wim17006/miniconda3/etc/profile.d/conda.sh
conda activate ptaudio

python3 make_ood.py
