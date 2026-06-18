"""Quickstart: diagnose a dataset in five lines."""
from cnsd import CNSD, load_dataset

data = load_dataset('cwru')
report = CNSD().fit(data).diagnose(data)

print(report.summary())
for statement in report.root_causes()[:5]:
    print(statement)
print(f"\n{len(report.conflicts())} units flagged for review (physics vs network).")
