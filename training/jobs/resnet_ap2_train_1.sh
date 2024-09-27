#!/bin/bash
#SBATCH --job-name=resnet50_ap2_train_1
#SBATCH --output=logs/resnet50_ap2_train_1.log
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1-21:00:00
#SBATCH --cpus-per-task=9

#module purge
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
#module load Python/3.11.3-GCCcore-12.3.0
#which python
#PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
#source $HOME/venvs/tensors/bin/activate
module load CUDA
source $HOME/venvs/tensor_CORF/bin/activate
#module list
#which python

#pip install typing_extensions requests sympy


mkdir -p $TMPDIR/tensor/train
mkdir -p $TMPDIR/tensor/val
#mkdir $TMPDIR/contour
#mkdir $TMPDIR/imagenet_c
#mkdir $TMPDIR/results

parallel -j 9 '
    echo "Unzipping {} to '$TMPDIR'/tensor/train"
    unzip -q {} -d '$TMPDIR'/tensor/train || echo "Error unzipping {}"
' ::: /scratch/s5288843/approach2_tensors/train{1..8}.zip

ls -lrt $TMPDIR/tensor/train | grep ^d | wc -l
echo "Parallel unzip process completed"
time unzip -q /scratch/s5288843/approach2_tensors/val.zip -d $TMPDIR/tensor/val

time python approach2_resnet50_train_1.py
