#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=get_lab_norm
#SBATCH --output=logs/get_lab_norm.log
#SBATCH --partition=regular

source $HOME/venvs/tensor_CORF/bin/activate
module list
which python

time unzip -q /scratch/$USER/ILSVRC/Data/CLS-LOC/train.zip -d $TMPDIR/dataset

cd ..

time python ./get_ab_norm.py
