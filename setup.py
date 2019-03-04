from setuptools import setup
import os
from pathlib import Path

root_dir = str(Path(os.getcwd()).parents[0])
print("root dir: ",root_dir)

setup(
    name='adversarial_sampling_experiments',
    packages=['adversarial_sampling_experiments'],
    package_dir={'':root_dir}
)