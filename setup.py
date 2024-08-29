from setuptools import setup, find_packages

setup(
    name='autogluonImputer',
    version='0.1.0',
    author='Deniz Akdemir',
    author_email='dakdemir@nmdp.org',  # Replace with your email
    description='A package for imputing missing data using AutoGluon.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dakdemir-nmdp/AutoGluonImputer',  # Replace with your GitHub repository URL
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        'autogluon',
        'autogluon.eda',
        'autogluon.multimodal',
        'autogluon.tabular',
        'autogluon.timeseries',
        'lifelines',
        'shap',
        'pyreadstat'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify the Python versions supported
)
