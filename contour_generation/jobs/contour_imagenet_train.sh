#!/bin/bash
#SBATCH --time=108:00:00
#SBATCH --mem=300GB
#SBATCH --job-name=contours-train-3
#SBATCH --output=contour-train-3.log
#SBATCH --partition=regular

module purge
module load MATLAB/2022b-r5
mkdir $TMPDIR/dataset
mkdir $TMPDIR/contour

unzip -q /scratch/$USER/ILSVRC/Data/CLS-LOC/train.zip -d $TMPDIR/dataset

matlab -nodisplay -singleCompThread < contour_train.m

cd $TMPDIR/contour/

zip -r /scratch/$USER/contour_train.zip .


# Note: The imagenet training set has around 1.28 million images, so this dataset needs to be split for better  
# In this implementation of the project, the dataset was split in four for contour generation
#
# In the script_contour_imagenet.m, tweak the looping directories to 1 to 250, 251 to 500, 501 to 750 and 750 to 1000
