"""Compute TFRs."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import mne

this_dir = Path(__file__).parent
sys.path.append(str(this_dir / '..'))

import paros_bids_utils
import paros_bids_config

results_dir = this_dir / 'results'
results_dir.mkdir(exist_ok=True)

subjects = paros_bids_utils.get_subjects()
groups = paros_bids_utils.get_groups()
task = 'task-lexicaldecision'
mbp_path = paros_bids_config.bids_root / 'derivatives' / 'mne-bids-pipeline'
subjects_dir = (
    paros_bids_config.bids_root / 'derivatives' / 'freesurfer' / 'subjects')

tfr_kinds = ('power', 'itc')
group_names = ('asd', 'control')
tfrs = {kind: {group: list() for group in group_names} for kind in tfr_kinds}
for subject in subjects:
    if subject not in groups['asd'] and subject not in groups['control']:
        continue
    for tfr_kind in tfr_kinds:
        with mne.use_log_level('error'):
            tfr = mne.time_frequency.read_tfrs(
                mbp_path / subject / 'meg' /
                f'{subject}_{task}_{tfr_kind}+lexical+tfr.h5')
        assert len(tfr) == 1, len(tfr)
        tfr = tfr[0]
        if subject in groups['asd']:
            tfrs[tfr_kind]['asd'].append(tfr)
        else:
            assert subject in groups['control'], subject
            tfrs[tfr_kind]['control'].append(tfr)

# cols: power, itc
# rows: asd, control
fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
for ti, tfr_kind in enumerate(tfr_kinds):
    for gi, group in enumerate(group_names):
        these_tfrs = [tfr.copy().apply_baseline((-0.15, 0), 'zscore') for tfr in tfrs[tfr_kind][group]]
        tfr = mne.grand_average(these_tfrs)
        tfr.plot_topo(vmin=-5, vmax=5)
        raise RuntimeError
