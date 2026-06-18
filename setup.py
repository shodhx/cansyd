from setuptools import setup, find_packages

setup(
    name='cnsd',
    version='1.0.0',
    description='Causal Neuro-Symbolic Diagnosis - a five-layer '
                'fault-diagnosis system for rotating machinery',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    extras_require={
        'perception': ['tensorflow>=2.10'],   # Layer 1 (CNN) backend
        'counterfactual': ['dowhy>=0.11', 'pandas', 'networkx'],  # Rung 3
        'all': ['tensorflow>=2.10', 'dowhy>=0.11', 'pandas', 'networkx'],
    },
)
