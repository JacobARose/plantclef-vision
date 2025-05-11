""" """

import os


def clear_directory(path: str, verbose: bool = False) -> None:
    """
    Clear all files and directories in the specified path.
    """
    status = False
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
            status = True
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
            status = True
    if not verbose:
        return
    if status:
        print(f"Cleared existing data in {path}")
    else:
        print(f"No files to clear in {path}")
