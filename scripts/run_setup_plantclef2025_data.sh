#!/bin/bash
#===============================================================================
# Script Name: run_setup_plantclef2025_data.sh
# Description: This script is used to run a specific command with hardcoded
#              configuration parameters for setting up PlantCLEF 2025 data.
# Author: Jacob A Rose
# Date: Monday April 14th, 2025

# [TODO] -- Add a custom registry.json file to the plantclef-vision/data directory to allow consistent mapping between source URL/filename and target local directory name
# e.g. {"PlantCLEF2024singleplanttrainingdata_800_max_side_size":
#           "small_singleplanttrainingdata"}


#===============================================================================

# Exit immediately if a command exits with a non-zero status
# set -e

# Enable debugging (uncomment for debugging purposes)
# set -x


# source ~/plantclef-vision/scripts/activate.sh

# Define constants and configuration parameters below

input_url="https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata_800_max_side_size.tar"
output_dir="$HOME/plantclef-vision/data/plantclef2025/PlantCLEF2024singleplanttrainingdata_800_max_side_size"


bash $HOME/plantclef-vision/scripts/download_and_extract.sh -i "$input_url" -o "$output_dir"
