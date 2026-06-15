from .metrics import compute_ece, proposition_a_test
from .baseline import (build_wdcnn, build_ticnn, run_published_baselines,
                       train_irm, run_irm)
from .classification import train_cnn, evaluate_protocol_b
