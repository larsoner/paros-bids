from pathlib import Path
import os.path as op
import numpy as np
import mne
from mnefun import extract_expyfun_events


def score(file_path):
    """Scoring function"""
    events, presses = extract_expyfun_events(file_path)[:2]
    i = np.arange(len(events))
    events[i, 2] -= 1
    mask = events[i, 2] > 0
    events = events[mask]
    presses = np.array(presses)[mask].tolist()
    output_file = op.join(file_path.parent, op.basename(file_path)[:-4] + '-eve.fif')
    mne.write_events(output_file, events)

    # get subject performance
    # boolean mask using modulus of target event and 2
    targets = (events[:, 2] % 2 == 0)
    has_presses = np.array([len(pr) > 0 for pr in presses], bool)
    n_targets = np.sum(targets)
    hits = np.sum(has_presses[targets])
    false_alarms = np.sum(has_presses[~targets])
    misses = n_targets - hits
    print('HMF: %s, %s, %s' % (hits, misses, false_alarms))


data_path = "/Users/ktavabi/MEG/paros"
raw_files = Path(data_path).rglob("*-raw.fif")
for rr in raw_files:
    score(rr)