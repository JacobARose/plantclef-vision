"""
file: clean_old_data.sh
created on: Sunday May 18th, 2025
created by: Jacob A Rose


"""


# Test run - shows what would be deleted
# find /teamspace/studios/this_studio/plantclef-vision/data/PlantCLEF2024singleplanttrainingdata_800_max_side_size -type f | parallel --dry-run rm


# find /teamspace/studios/this_studio/plantclef-vision/data/PlantCLEF2024singleplanttrainingdata_800_max_side_size -type f -print0 | parallel --jobs $(nproc) --progress -0 rm
