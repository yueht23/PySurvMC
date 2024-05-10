from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent

long_description = (this_directory / "Readme.md").read_text()

setup(
    name='PySurvMC',
    version='0.2',
    author='yueht17',
    author_email="yueht17@foxmail.com",
    # description='A Python package for Survival Analysis with Monte Carlo Simulation',

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(),
    install_requires=[
        "pymc>=5.14.0",
        "arviz>=0.13.0",
        "cachetools>=4.2.1",
        "cloudpickle",
        "numpy>=1.15.0",
        "pandas>=0.24.0",
        "pytensor>=2.20,<2.21",
        "rich>=13.7.1",
        "scipy>=1.4.1",
        "typing-extensions>=3.7.4",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
