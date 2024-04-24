#!/bin/bash
#
#SBATCH --job-name=valid_godcaster
#SBATCH --output=out.valid_godcaster.log
#SBATCH --error=err.valid_godcaster.log
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

# Initialize count of valid directories
valid_dirs=0

# Iterate over each subdirectory in the current directory
for dir in */; do
  if [ -d "$dir" ]; then  # Ensure it's a directory
    # Assume the directory is valid unless proven otherwise
    all_matched=true

    # Loop through each .mp4 file in the subdirectory
    for mp4_file in "$dir"*.mp4; do
      # Construct the name of the corresponding .json file
      base_name=$(basename "$mp4_file" .mp4)
      json_file="${dir}${base_name}.json"

      if [ ! -f "$json_file" ]; then
        all_matched=false
        break
      fi
    done

    # If all .mp4 files had corresponding .json files, increment the count
    if $all_matched; then
      ((valid_dirs++))
    fi
  fi
done

# Print the count of valid directories
echo "Number of valid directories: $valid_dirs"
