#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=paper14
#SBATCH --error=dl_err.%J.err
#SBATCH --output=dl_out.%J.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00

source /../miniconda3/bin/activate
conda activate pytorch_gpu
python3 train.py --batch-size 256 --epochs 30 --lr 0.001 --lmda 0.5 --workers 40 --device cuda

