#!/bin/bash
#
#SBATCH --job-name=godcaster_merge
#SBATCH --output=out.godcaster_merge.log
#SBATCH --error=err.godcaster_merge.log
#
# Number of tasks needed for this job. Generally, used with MPI jobs
#SBATCH --ntasks=1
#SBATCH --partition=parallel
#
# Time format = HH:MM:SS, DD-HH:MM:SS
#SBATCH --time=72:00:00
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

# Deletes all .m4a files in the current directory
find . -maxdepth 0 -type f -name "*.m4a" -delete

# Deletes all .mp4 files in the current directory
find . -maxdepth 0 -type f -name "*.mp4" -delete