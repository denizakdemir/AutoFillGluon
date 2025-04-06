"""
Setup script for AutoFillGluon package.
"""

from setuptools import setup, find_packages

# Read the contents of README.md file
with open("readme.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line for line in f.read().splitlines() if not line.startswith("#")]

setup(
    name="autofillgluon",
    version="0.1.0",
    author="Deniz Akdemir",
    author_email="denizakdemir@gmail.com",
    description="Machine learning-based missing data imputation using AutoGluon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/denizakdemir/AutoFillGluon",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "isort>=5.0.0",
            "flake8>=3.8.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    include_package_data=True,
)