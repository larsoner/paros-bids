"""Compute signed source space activations in frontal and auditory labels."""

from pathlib import Path
import sys

import numpy as np
import h5io
import mne
import matplotlib.pyplot as plt

this_dir = Path(__file__).parent
sys.path.append(str(this_dir / '..'))

import paros_bids_config

subjects = [f'sub-{s}' for s in paros_bids_config.subjects]
mbp_path = paros_bids_config.bids_root / 'derivatives' / 'mne-bids-pipeline'
subjects_dir = (
    paros_bids_config.bids_root / 'derivatives' / 'freesurfer' / 'subjects')

# HCPMMP1
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1_combined', 'both', subjects_dir=subjects_dir)
want_labels = [
    'Early Auditory Cortex-lh',
    'Early Auditory Cortex-rh',
    'Inferior Frontal Cortex-lh',
    'Inferior Frontal Cortex-rh',
]
labels = [label for label in labels if label.name in want_labels]
labels = sorted(labels, key=lambda label: want_labels.index(label.name))
assert want_labels == [label.name for label in labels]
# brain = mne.viz.Brain('fsaverage', 'both', 'inflated', subjects_dir=subjects_dir)
# for label in labels:
#     brain.add_label(label)
# raise RuntimeError

# Extract label time courses for all subjects in all four labels
conditions = ('lexical/high', 'lexical/low', 'nonlex/high', 'nonlex/low')
source_path = this_dir / 'time_courses.h5'
src = mne.read_source_spaces(
    subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-5-src.fif')
labels
if not source_path.is_file():
    source_data = dict()
    for si, subject in enumerate(subjects):
        print(f'Processing {si + 1}/{len(subjects)} ({subject})')
        subj_dir = mbp_path / subject / 'meg'
        epochs = mne.read_epochs(
            subj_dir / f'{subject}_task-lexicaldecision_epo.fif')
        epochs.equalize_event_counts()
        assert len(epochs) >= 160, len(epochs)
        inv = mne.minimum_norm.read_inverse_operator(
            subj_dir / f'{subject}_task-lexicaldecision_inv.fif')
        morph = mne.compute_source_morph(
            inv['src'], src_to=src, subjects_dir=subjects_dir, smooth=15)
        source_data[subject] = dict()
        for condition in conditions:
            evoked = epochs[condition].average()
            stc = mne.minimum_norm.apply_inverse(
                evoked, inv, method='dSPM', pick_ori='normal')
            stc_fs = morph.apply(stc)
            ltc = mne.extract_label_time_course(
                stc_fs, labels, src, mode='mean_flip')
            if 'times' not in source_data:
                source_data['times'] = stc.times
            assert np.allclose(source_data['times'], stc.times)
            assert ltc.shape == (len(labels), len(stc.times))
            source_data[subject][condition.replace('/', '_')] = ltc
    h5io.write_hdf5(source_path, source_data)
source_data = h5io.read(source_path)

# Export "evoked" for each subject in each condition

# Create "grand_average" subject that is an `ndarray` of all subjects so we
# can plot mean+/-SEM as well


# Show mean for each condition for each subject, then as grand average:
# - As "evoked" data (source space data back on mags)
# - As PDF 2 columns (left/right) and 2 rows (frontal/auditory), each with
#   four traces (one per condition); grand average should have the SEM
