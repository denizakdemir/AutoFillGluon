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
