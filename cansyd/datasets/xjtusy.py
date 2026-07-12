import glob
import os

import numpy as np
import pandas as pd

from cnsd.datasets.contract import Dataset
from cnsd.physics.configs import XJTUSY_PHYSICS

# Known fault mappings per XJTU-SY paper
# We map Bearing ID to CNSD fault class: 1=Outer, 2=Inner, 3=Cage
XJTUSY_FAULTS = {
    '35Hz12kN': {
        'Bearing1_1': 1,  # Outer
        'Bearing1_2': 1,  # Outer
        'Bearing1_3': 1,  # Outer
    },
    '37.5Hz11kN': {
        'Bearing2_1': 2,  # Inner
        'Bearing2_2': 1,  # Outer
        'Bearing2_4': 1,  # Outer
        'Bearing2_5': 1,  # Outer
    },
    '40Hz10kN': {
        'Bearing3_1': 1,  # Outer
        'Bearing3_2': 2,  # Inner
        'Bearing3_3': 2,  # Inner
        'Bearing3_4': 2,  # Inner
        'Bearing3_5': 1,  # Outer
    },
}


def load_xjtusy_domain_split(
    data_dir=r'E:\301\CNSD\data\XJTU-SY\XJTU-SY_Bearing_Datasets',
    window_size=32768,
    train_cond='35Hz12kN',
    test_cond='37.5Hz11kN',
):
    """
    Loads authentic XJTU-SY dataset and strictly splits by condition (Domain Shift).
    Uses the run-to-failure nature to grab the first 15% as Healthy (0) and the
    last 15% as the Fault label.
    """

    def _load_condition(cond_folder, rpm_val):
        X, y, cond = [], [], []
        cond_path = os.path.join(data_dir, cond_folder)

        if not os.path.exists(cond_path):
            return [], [], []

        for bearing_folder in os.listdir(cond_path):
            bearing_path = os.path.join(cond_path, bearing_folder)
            if not os.path.isdir(bearing_path):
                continue

            fault_label = XJTUSY_FAULTS.get(cond_folder, {}).get(bearing_folder, None)
            if (
                fault_label is None or bearing_folder == 'Bearing1_5'
            ):  # skip 1_5 to avoid mixed labels
                continue

            csv_files = sorted(
                glob.glob(os.path.join(bearing_path, '*.csv')),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
            )

            total_files = len(csv_files)
            if total_files < 10:
                continue  # ignore wildly corrupted directories

            healthy_count = max(1, int(total_files * 0.20))
            fault_count = max(1, int(total_files * 0.20))

            # Sliding window parameters
            step_size = 1024

            # Extract Healthy
            for fpath in csv_files[:healthy_count]:
                df = pd.read_csv(fpath)
                sig = df.iloc[:, 0].values  # Horizontal acceleration
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

                # Slicing the 32768 array into overlapping 4096 windows
                for start_idx in range(0, len(sig) - window_size + 1, step_size):
                    X.append(sig[start_idx : start_idx + window_size])
                    y.append(0)
                    cond.append(rpm_val)

            # Extract Fault
            for fpath in csv_files[-fault_count:]:
                df = pd.read_csv(fpath)
                sig = df.iloc[:, 0].values
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)

                for start_idx in range(0, len(sig) - window_size + 1, step_size):
                    X.append(sig[start_idx : start_idx + window_size])
                    y.append(fault_label)
                    cond.append(rpm_val)

        return X, y, cond

    X_train, y_train, c_train = _load_condition(train_cond, 2100.0)
    X_test, y_test, c_test = _load_condition(test_cond, 2250.0)

    if not X_train or not X_test:
        raise FileNotFoundError(
            f'Could not load data. Ensure {data_dir} contains extracted {train_cond} and {test_cond} folders.'
        )

    ds_train = Dataset.from_arrays(
        X=np.array(X_train, dtype=np.float32),
        y=np.array(y_train, dtype=np.int32),
        cond=np.array(c_train, dtype=np.float32),
        fs=25600,
        physics=XJTUSY_PHYSICS,
        taxonomy={0: ('Normal', 'None'), 1: ('Outer Race', 'Medium'), 2: ('Inner Race', 'High')},
        name='XJTUSY_Train',
    )

    ds_test = Dataset.from_arrays(
        X=np.array(X_test, dtype=np.float32),
        y=np.array(y_test, dtype=np.int32),
        cond=np.array(c_test, dtype=np.float32),
        fs=25600,
        physics=XJTUSY_PHYSICS,
        taxonomy={0: ('Normal', 'None'), 1: ('Outer Race', 'Medium'), 2: ('Inner Race', 'High')},
        name='XJTUSY_Test',
    )

    return ds_train, ds_test
