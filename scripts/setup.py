from setuptools import setup, find_packages

setup(
    name='autogluonImputer',
    version='0.1.0',
    author='Deniz Akdemir',
    author_email='your.email@example.com',  # Replace with your actual email
    description='A package for imputing missing data using AutoGluon, with scoring utilities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dakdemir-nmdp/AutoGluonImputer',
    packages=find_packages(where='scripts'),  # Specify the directory containing your modules
    package_dir={'': 'scripts'},  # Map the root package to the 'scripts' directory
    install_requires=[
        'autogluon',
        'autogluon.eda',
        'autogluon.multimodal',
        'autogluon.tabular',
        'autogluon.timeseries',
        'lifelines',
        'shap',
        'pyreadstat',
        'python-pptx',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
