"""
file: utils.py
Created on: Tuesday May 6th, 2025
Created by: Jacob A Rose
"""

from datetime import datetime


def print_current_time():
    """
    Print the current date and time in the format YYYY-MM-DD HH:MM:SS.
    """
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
