#!/bin/bash
#===============================================================================
# FILE: download_and_extract.sh
# DESCRIPTION: Script to download and extract files for PlantCLEF Vision project.
# USAGE: ./download_and_extract.sh
# AUTHOR: Jacob A Rose
# CREATED: Monday April 14th, 2025
#===============================================================================
# Note - (Added Sunday Apr 27th, 2025)
# * [TODO] - Consider replacing this script's logic with the following python line:
#   from torchvision.datasets.utils import download_and_extract_archive
# * Source Code of the above line: github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py





# set -euo pipefail
# IFS=$'\n\t'

# Default values for input URL and output directory
input_url=""
output_dir=""


##############################################################################
##############################################################################
# Step 1: Parse command line options
# This script requires two arguments: input URL and output directory.
# The input URL is the location of the file to be downloaded,
# and the output directory is where the file will be extracted.

# Function to display usage
usage() {
    echo "Usage: $0 -i <input_url> -o <output_dir>"
    exit 1
}

# Parse command line options
while getopts "i:o:" opt; do
    case $opt in
        i)
            input_url="$OPTARG"
            ;;
        o)
            output_dir="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done

# Ensure both input URL and output directory are provided
if [[ -z "$input_url" || -z "$output_dir" ]]; then
    usage
fi

##############################################################################
##############################################################################
# Step 2: Download and extract the file

echo "Downloading and extracting files\nFrom: $input_url \nTo: $output_dir"

start_time="$(date)"
echo "[START] time: $start_time"

# Create the output directory if it doesn't exist
# and download the file from the input URL
# using curl, then extract it using tar with a progress bar
mkdir -p "$output_dir" && curl -# -L $input_url | tar -xf - -C $output_dir

# mkdir -p "$output_dir" && curl -# -L $input_url | tar -xf - -C $output_dir 2> error_log.txt

# Check if the extraction was successful
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to extract files from $input_url"

    echo "Contents of $output_dir:"
    ls -l "$output_dir"
    # exit 1
fi


echo "[FINISHED] -- Downloaded and extracting files\nFrom: $input_url \nTo: $output_dir"
echo "[START] time: $start_time"
echo "[FINISH] time: $(date)"
