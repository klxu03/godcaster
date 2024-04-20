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
#SBATCH --time=72:00:00
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

# Directory where merged files will be stored
mkdir -p merged

# Loop through all mp4 files in the current directory
for video in *.mp4; do
    # Extract the base name by splitting at the first '.' and taking the first part
    base_name="${video%%.*}"
    
    # Find the matching m4a file by pattern matching
    audio=$(echo "$base_name".*.m4a)
    
    # Define the output filename in the merged directory
    output="merged/${base_name}.mp4"

    if [ -f "$audio" ]; then
        # Run ffmpeg to merge video and audio
        ffmpeg -i "$video" -i "$audio" -c:v copy -c:a aac "$output"
        echo "Merged $video and $audio into $output"
    else
        # Move the video file to the merged directory if no matching audio file is found
        mv "$video" "$output"
        echo "No matching audio file found for $video. Moved to $output"
    fi
done