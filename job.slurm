#!/bin/bash
#SBATCH --job-name=job
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=job.out



# 运行你的脚本
python cs336_alignment/help.py