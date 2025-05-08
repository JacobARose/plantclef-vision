"""
file: utils.py
Created on: Tuesday May 6th, 2025
Created by: Jacob A Rose
"""

from datetime import datetime
import os


def print_current_time():
    """
    Print the current date and time in the format YYYY-MM-DD HH:MM:SS.
    """
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


def print_dir_size(dir_path):
    """
    Print the size of a directory in a human-readable format.
    Args:
        dir_path (str): Path to the directory.

    """
    print(f"Analyzing disk usage of directory: {dir_path}")

    total_size = os.popen(f"du -sh {dir_path}").read().strip()
    print(f"Directory Disk Usage: {total_size}")
    print_current_time()

    # total_size = 0
    # for dirpath, dirnames, filenames in os.walk(dir_path):
    #     for f in filenames:
    #         fp = os.path.join(dirpath, f)
    #         total_size += os.path.getsize(fp)
    # print(f"Directory Disk Usage: {total_size / (1024 * 1024):.2f} MB")
