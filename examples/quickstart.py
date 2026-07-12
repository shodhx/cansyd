"""Quickstart: diagnose a dataset in five lines."""

# bring your own arrays (signals, labels, operating condition, sampling rate)
import numpy as np

from cansyd import CANSYD, Dataset

X = np.random.randn(200, 1024)
y = np.random.randint(0, 10, 200)
cond = np.random.choice([0, 1, 2, 3], 200)
data = Dataset.from_arrays(X, y, cond, fs=12000)
report = CANSYD().fit(data).diagnose(data)

print(report.summary())
for statement in report.root_causes()[:5]:
    print(statement)
print(f'\n{len(report.conflicts())} units flagged for review (physics vs network).')
