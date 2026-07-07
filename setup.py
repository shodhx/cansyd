from pathlib import Path

from setuptools import find_packages, setup

long_description = (
    Path('README.md').read_text(encoding='utf-8') if Path('README.md').exists() else ''
)

extras = {
    'perception': ['tensorflow>=2.16', 'keras>=3.0'],
    'counterfactual': ['dowhy>=0.12', 'pandas>=2.0', 'networkx>=3.0'],
}
extras['all'] = extras['perception'] + extras['counterfactual']

setup(
    name='cnsd',
    version='1.0.0',
    description='Causal Neuro-Symbolic Diagnosis - a five-layer fault-diagnosis framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Abhimanyu Prasad, Kazi Tasfin Mahmud',
    url='https://github.com/abhiprd2000/CNSD',
    license='MIT',
    packages=find_packages(include=['cnsd', 'cnsd.*']),
    python_requires='>=3.11',
    install_requires=['numpy>=2.0', 'scipy>=1.11', 'scikit-learn>=1.4', 'pyyaml>=6.0'],
    extras_require=extras,
)
