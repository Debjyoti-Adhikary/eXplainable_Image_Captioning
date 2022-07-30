#!/bin/bash


#SBATCH -J 2_ComparativeStudy            #Job name(--job-name)
#SBATCH --output /scratch/20cs91f01/Debjyoti/DeepLearning/GradCAM_based_image_captioning/logs/slurm_start_paper_%j.log          #Name of stdout output file(--output)
#SBATCH --error /scratch/20cs91f01/Debjyoti/DeepLearning/GradCAM_based_image_captioning/logs/slurm_start_paper_%j.log   #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name
#SBATCH -q aritracs
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH --gres=gpu:1 # num of GPUs (max 2)
#SBATCH -c 8                    #(--cpus-per-task) Number of Threads
#SBATCH --mem=32000 # RAM ín MBs
#SBATCH --mail-user=debjyoti.das.adhikary@iitkgp.ac.in        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH --time 3-0            # 3 days max

module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cudnn/7.6.2
module load compiler/cuda/10.1

#source /home/$USER/.bashrc
# source /home/$USER/.bash_aliases

export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python --version
nvcc --version
# python pretrain_manuals.py --from_pretrained --layerwise_lr_decay --per_dev_batch_size 8 --checkpoint_path embert_model_from_pretrained_layerwise_lr_decay/checkpoint-460000
#python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 pretrain_manuals.py --from_pretrained --layerwise_lr_decay --per_dev_batch_size 8
python 0_Dataset.py