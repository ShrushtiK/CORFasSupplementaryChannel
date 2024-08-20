#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=contours-val
#SBATCH --output=contour-val.log
#SBATCH --partition=regular

module purge
module load MATLAB/2022b-r5
mkdir $TMPDIR/dataset
mkdir $TMPDIR/contour

unzip -q /scratch/$USER/ILSVRC/Data/CLS-LOC/val_labelled.zip -d $TMPDIR/dataset

matlab -nodisplay -singleCompThread < ../script_contour_imagenet.m

cd $TMPDIR/contour/

zip -r /scratch/$USER/contour_val.zip .
