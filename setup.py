from setuptools import setup, find_packages

setup(
    name='cnsd',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.15.0',
        'numpy>=1.24.3',
        'scipy>=1.11.3',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'wfdb>=4.1.0',
    ],
    python_requires='>=3.8',
)