#!/bin/bash
#
#SBATCH --job-name=godcaster_whisper
#SBATCH --output=out.godcaster_whisper.log
#SBATCH --error=err.godcaster_whisper.log
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
#SBATCH --gres=gpu:1
#SBATCH -A ia1
#SBATCH --partition debug
#SBATCH --qos=normal
#
# Send mail to the email address when the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kxu39@jhu.edu
#

cd godcaster
cd src/captions

# Go ahead and run the video splitting script to split a video by their round and properly store the clips
poetry run python load_distributor.py 