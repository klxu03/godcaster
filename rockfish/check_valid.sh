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

# File to write invalid directories
invalid_file="invalid_directories.txt"
no_mp4_file="no_mp4_directories.txt"

# Clear the files at the start of the script
> "$invalid_file"
> "$no_mp4_file"

# Iterate over each subdirectory in the current directory
for dir in */; do
  if [ -d "$dir" ]; then  # Ensure it's a directory
    # Check if there are any .mp4 files
    mp4_files=($(find "$dir" -maxdepth 1 -type f -name "*.mp4"))
    if [ ${#mp4_files[@]} -eq 0 ]; then
      # Log the directory path to the no_mp4 file
      echo "$dir" >> "$no_mp4_file"
      continue  # Skip to the next directory
    fi

    # Assume the directory is valid unless proven otherwise
    all_matched=true

    # Loop through each .mp4 file in the subdirectory
    for mp4_file in "${mp4_files[@]}"; do
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
    else
      # Log the directory path to the invalid file
      echo "$dir" >> "$invalid_file"
    fi
  fi
done

# Print the count of valid directories
echo "Number of valid directories: $valid_dirs"