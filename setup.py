<<<<<<< HEAD
from setuptools import setup, find_packages

setup(
    name='autogluonImputer',
    version='0.1.0',
    author='Deniz Akdemir',
    author_email='dakdemir@nmdp.org',  # Replace with your email
    description='A package for imputing missing data using AutoGluon.',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dakdemir-nmdp/AutoGluonImputer',  # Replace with your GitHub repository URL
    packages=find_packages(where='scripts'),  # Finds all packages under 'scripts'
    package_dir={'': 'scripts'},  # Maps the root package to the 'scripts' directory
    install_requires=[
        'lightgbm',
        'autogluon',
        'autogluon.eda',
        'autogluon.multimodal',
        'autogluon.tabular',
        'autogluon.timeseries',
        'lifelines',
        'pyreadstat'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify the Python versions supported
)
=======
"""
Setup script for AutoFillGluon package.
"""

from setuptools import setup, find_packages

# Read the contents of README.md file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line for line in f.read().splitlines() if not line.startswith("#")]

setup(
    name="autofillgluon",
    version="0.1.0",
    author="Deniz Akdemir",
    author_email="your.email@example.com",  # Replace with your email
    description="Machine learning-based missing data imputation using AutoGluon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AutoFillGluon",  # Replace with your GitHub repository URL
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
>>>>>>> 74e04e5 (Remove unnecessary files and clean up repository)
