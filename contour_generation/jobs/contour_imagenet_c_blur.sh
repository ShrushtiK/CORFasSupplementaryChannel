#!/bin/bash
#SBATCH --time=5-00:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=contours-blur
#SBATCH --output=logs/contour-blur.log
#SBATCH --partition=regular
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

module purge
module load MATLAB/2022b-r5
mkdir $TMPDIR/dataset
mkdir $TMPDIR/contour

tar -xf /scratch/$USER/blur.tar -C $TMPDIR/dataset

ls -lrt $TMPDIR/dataset

matlab -nodisplay < ../script_contour_imagenet_c.m

cd $TMPDIR/contour/

zip -r /scratch/$USER/contours/contour_blur.zip .
