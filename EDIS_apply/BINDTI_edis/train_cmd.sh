#!/bin/bash
#SBATCH -p gpu31,gpu1,gpu2
#SBATCH -J train
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 5000:00:00
#SBATCH -o ./log/%j.out

export LD_LIBRARY_PATH=/lustre/grp/gyqlab/linxh/cuda-11.8/lib64:/opt/slurm-23.02.5/lib:/opt/slurm-23.02.5/lib/slurm:/opt/openmpi-4.0.4/lib
export PATH=/lustre/grp/gyqlab/linxh/bin:/lustre/grp/gyqlab/linxh/.conda/envs/torch_lius/bin:/lustre/apps/bioapps/.linuxbrew/bin:/opt/slurm-23.02.5/bin:/opt/slurm-23.02.5/sbin:/opt/openmpi-4.0.4/bin:/usr/lib/jvm/jdk-18/bin:/usr/local/go/bin:/usr/local/cuda-11.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin


python3.8 main.py
