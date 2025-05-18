#!/bin/bash

# Define source and destination directories
# SOURCE="/teamspace/studios/plantclef2025-aha8/plantclef-vision/data/plantclef2025/processed/plantclef2024/test/"
# DESTINATION="/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/processed/plantclef2024/test/"


SOURCE="/teamspace/studios/plantclef2025-aha8/plantclef-vision/data/plantclef2025/processed/plantclef2024/val/"
DESTINATION="/teamspace/studios/this_studio/plantclef-vision/data/plantclef2025/processed/plantclef2024/val/"

# Use rsync to copy the directory
rsync -av --progress "$SOURCE" "$DESTINATION"
