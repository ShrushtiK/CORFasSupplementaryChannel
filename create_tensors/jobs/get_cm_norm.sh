#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=get_contour_norm
#SBATCH --output=logs/get_contour_norm.log
#SBATCH --partition=regular

source $HOME/venvs/tensor_CORF/bin/activate
module list
which python

mkdir $TMPDIR/contour

time unzip -q /scratch/$USER/contours/contours_train.zip -d $TMPDIR/contour
cd ..
time python get_contour_norm.py