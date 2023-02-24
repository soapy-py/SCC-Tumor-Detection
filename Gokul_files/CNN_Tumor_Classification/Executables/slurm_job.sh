#!/bin/bash
#SBATCH --chdir=/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/CNN_Tumor_Classification/Executables
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --job-name=
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH --account=qdp-alpha
#SBATCH --partition=v100_12

#SBATCH --gpu_cmode=shared
source ~/.bashrc
cd /dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/CNN_Tumor_Classification/Executables

conda activate jupyter-ultimate
python /dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/users/Gokul_Srinivasan/SCC-Tumor-Detection/Gokul_files/CNN_Tumor_Classification/Executables/ViT.py
    