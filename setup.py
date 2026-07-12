from pathlib import Path

from setuptools import find_packages, setup

long_description = (
    Path('README.md').read_text(encoding='utf-8') if Path('README.md').exists() else ''
)

long_description = long_description.replace(
    'docs/assets/', 'https://raw.githubusercontent.com/shodhx/cansyd/main/docs/assets/'
)

extras = {
    'perception': ['tensorflow>=2.16', 'keras>=3.0'],
    'counterfactual': ['dowhy>=0.12', 'pandas>=2.0', 'networkx>=3.0'],
}
extras['all'] = extras['perception'] + extras['counterfactual']

setup(
    name='cansyd',
    version='1.0.0',
    description='Causal Neuro-Symbolic Diagnosis - a five-layer fault-diagnosis framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Abhimanyu Prasad, Kazi Tasfin Mahmud',
    url='https://github.com/shodhx/cansyd',
    license='MIT',
    packages=find_packages(include=['cansyd', 'cansyd.*']),
    python_requires='>=3.11',
    install_requires=['numpy>=2.0', 'scipy>=1.11', 'scikit-learn>=1.4', 'pyyaml>=6.0'],
    extras_require=extras,
    keywords='bearing fault diagnosis, causal inference, neuro-symbolic, physics-informed, condition monitoring, PHM',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
    ],
    project_urls={
        'Source': 'https://github.com/shodhx/cansyd',
        'Issues': 'https://github.com/shodhx/cansyd/issues',
        'Changelog': 'https://github.com/shodhx/cansyd/blob/main/CHANGELOG.md',
    },
)
