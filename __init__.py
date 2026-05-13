from .core import pipeline
from .data import loaders
from .continual import ccr_lora
from .eval import metrics

__version__ = '1.0.0'

def run():
    from .main import main
    main()

__all__ = ['run', 'pipeline', 'loaders', 'ccr_lora', 'metrics']