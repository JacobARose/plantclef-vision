from setuptools import setup, find_packages
import sys
import os
from pprint import pprint

print(f"Currently in file: {__file__}")
print(f"sys.path = {sys.path}")
print(f"os.getcwd() = {os.getcwd()}")
print("#" * 30)
print("find_packages(include=['plantclef', 'plantclef.*']) = ")

pprint(find_packages(include=["plantclef", "plantclef.*"]))
print("#" * 30)

setup(
    name="plantclef",
    version="0.1.1",
    # package_dir={'': 'plantclef'},
    packages=find_packages(include=["plantclef", "plantclef.*"]),
    # packages=find_packages(where="plantclef/__init__.py")
)
