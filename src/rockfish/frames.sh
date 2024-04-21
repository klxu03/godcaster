#!/bin/bash
#
#SBATCH --job-name=godcaster_splits
#SBATCH --output=out.godcaster_splits.log
#SBATCH --error=err.godcaster_splits.log
#
# Number of tasks needed for this job. Generally, used with MPI jobs
#SBATCH --ntasks=1
#SBATCH --partition=parallel
#
# Time format = HH:MM:SS, DD-HH:MM:SS
#SBATCH --time=144:00:00
#
# Minimum memory required per allocated  CPU  in  MegaBytes.
#SBATCH --mem-per-cpu=48000
#SBATCH --cpus-per-task=1
#SBATCH -A ia1
#SBATCH --partition debug
#SBATCH --qos=normal
#
# Send mail to the email address when the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kxu39@jhu.edu
#
# Create a job array
#SBATCH --array=0-71

cd godcaster
cd src/frames

poetry run python main.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_MAX