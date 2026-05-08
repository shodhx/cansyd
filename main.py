from data.loaders import *
from core.architecture import *
from core.rules import *
from core.causal import *
from core.pipeline import *
from continual.ccr_lora import *
from continual.cml import *
from eval.baselines import *
from eval.metrics import *

print('CNSD Pipeline Complete')
print(f'CWRU: {len(X_train_all)} train, {len(X_test_all)} test')
print(f'MIT-BIH: {len(X_train_ecg)} train, {len(X_test_ecg)} test')
print(f'MFPT: {len(X_train_mfpt)} train, {len(X_test_mfpt)} test')