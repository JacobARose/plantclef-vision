#!/bin/bash
"""
Created on Thursday June 5th, 2025
Created by Jacob A Rose


The directory of zip files associated with fossil-leaves should be hosted in an AWS S3 bucket, and mounted to the lightning studio file system at the following path:
`/teamspace/s3_connections/fossil-leaves/`

The contents should be:
```
Extant_Leaves_A-E_v2.0.zip
Extant_Leaves_F-O_v2.0.zip
Extant_Leaves_P-Z_v2.0.zip
Florissant_Fossil_v2.0.zip
General_Fossil_uncropped_v2.0.zip
General_Fossil_v2.0.zip
supplemental_data_v2.0.zip
```


"""


# Define the source directory and target directory
SOURCE_DIR="/teamspace/s3_connections/fossil-leaves"
TARGET_DIR="/teamspace/studios/this_studio/data/fossil-leaves"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# List of zip files to process

# Define groups of zip files as space-separated strings
declare -A ZIP_GROUPS
ZIP_GROUPS["Extant"]="Extant_Leaves_A-E_v2.0.zip Extant_Leaves_F-O_v2.0.zip Extant_Leaves_P-Z_v2.0.zip"
ZIP_GROUPS["Florissant_Fossil"]="Florissant_Fossil_v2.0.zip"
ZIP_GROUPS["General_Fossil"]="General_Fossil_uncropped_v2.0.zip General_Fossil_v2.0.zip"
ZIP_GROUPS["supplemental"]="supplemental_data_v2.0.zip"

# Function to display available groups
function display_groups() {
    echo "SOURCE_DIR: $SOURCE_DIR"
    echo "Available groups:"
    for group in "${!ZIP_GROUPS[@]}"; do
        echo "  - $group"
    done
}

# Prompt user for group selection
echo "Please select a group of zip files to process:"
display_groups
read -p "Enter group name: " GROUP_NAME

# Validate group selection
if [[ -z "${ZIP_GROUPS[$GROUP_NAME]}" ]]; then
    echo "Error: Invalid group name. Please select a valid group."
    exit 1
fi

# Process selected group
IFS=' ' read -r -a ZIP_FILES <<< "${ZIP_GROUPS[$GROUP_NAME]}"
echo "Processing the following files from group '$GROUP_NAME':"
for ZIP_FILE in "${ZIP_FILES[@]}"; do
    echo "  - $ZIP_FILE"
    # unzip -q "$SOURCE_DIR/$ZIP_FILE" -d "$TARGET_DIR" | pv -lep -s $(unzip -l "$SOURCE_DIR/$ZIP_FILE" | awk '/-----/ {getline; print $1}') > /dev/null
    unzip "$SOURCE_DIR/$ZIP_FILE" -d "$TARGET_DIR" | pv -l > /dev/null
done


# # Loop through each zip file and unzip it into the target directory
# for ZIP_FILE in "${ZIP_FILES[@]}"; do
#     if [ -f "$SOURCE_DIR/$ZIP_FILE" ]; then
#         echo "Unzipping $ZIP_FILE..."
#         # unzip -q "$SOURCE_DIR/$ZIP_FILE" -d "$TARGET_DIR"
        # unzip -q "$SOURCE_DIR/$ZIP_FILE" -d "$TARGET_DIR" | pv -lep -s $(unzip -l "$SOURCE_DIR/$ZIP_FILE" | awk '/-----/ {getline; print $1}') > /dev/null
#     else
#         echo "File $ZIP_FILE not found in $SOURCE_DIR"
#     fi
# done

# echo "Unzipping completed."