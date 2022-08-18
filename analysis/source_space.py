"""Compute signed source space activations in frontal and auditory labels."""

# TODO:
# - source space plots
# - split by group

from pathlib import Path
import sys

import numpy as np
import h5io
import mne
import matplotlib.pyplot as plt

this_dir = Path(__file__).parent
sys.path.append(str(this_dir / '..'))

import paros_bids_config

write_evokeds = True

subjects = [f'sub-{s}' for s in paros_bids_config.subjects]
task = 'task-lexicaldecision'
mbp_path = paros_bids_config.bids_root / 'derivatives' / 'mne-bids-pipeline'
subjects_dir = (
    paros_bids_config.bids_root / 'derivatives' / 'freesurfer' / 'subjects')

# HCPMMP1
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1_combined', 'both', subjects_dir=subjects_dir)
label_names = [
    'Early Auditory Cortex-lh',
    'Early Auditory Cortex-rh',
    'Inferior Frontal Cortex-lh',
    'Inferior Frontal Cortex-rh',
]
labels = [label for label in labels if label.name in label_names]
labels = sorted(labels, key=lambda label: label_names.index(label.name))
assert label_names == [label.name for label in labels]
# brain = mne.viz.Brain('fsaverage', 'both', 'inflated', subjects_dir=subjects_dir)
# for label in labels:
#     brain.add_label(label)
# raise RuntimeError

# Extract label time courses for all subjects in all four labels
conditions = ('lexical/high', 'lexical/low', 'nonlex/high', 'nonlex/low')


def sanitize(condition):
    return condition.replace('/', '_')


source_path = this_dir / 'time_courses.h5'
src = mne.read_source_spaces(
    subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif')
if not source_path.is_file():
    source_data = dict()
    for si, subject in enumerate(subjects):
        print(f'Processing {si + 1}/{len(subjects)} ({subject})')
        subj_dir = mbp_path / subject / 'meg'
        epochs = mne.read_epochs(subj_dir / f'{subject}_{task}_epo.fif')
        epochs.equalize_event_counts()
        assert len(epochs) >= 160, len(epochs)
        inv = mne.minimum_norm.read_inverse_operator(
            subj_dir / f'{subject}_{task}_inv.fif')
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
            source_data[subject][sanitize(condition)] = ltc
    h5io.write_hdf5(source_path, source_data)
source_data = h5io.read_hdf5(source_path)
del labels

# Create "grand-average" subject that is an `ndarray` of all subjects so we
# can plot mean+/-SEM as well
source_data['grand-average'] = {
    sanitize(condition): np.array([source_data[subject][sanitize(condition)]
                         for subject in subjects], float)
    for condition in conditions
}

# Export "evoked" for each subject in each condition by using selections
evokeds_dir = this_dir / 'evokeds'
evokeds_dir.mkdir(exist_ok=True)
subject = subjects[0]
evoked = mne.read_evokeds(
    mbp_path / subject / 'meg' / f'{subjects[0]}_{task}_ave.fif')[0]
evoked.data[:] = 0
selections = list()
for label_name in label_names:
    if '-lh' in label_name:
        assert '-rh' not in label_name
        first = 'Left'
    else:
        assert '-rh' in label_name
        first = 'Right'
    if 'Auditory' in label_name:
        assert 'Frontal' not in label_name
        second = 'temporal'
    else:
        assert 'Frontal' in label_name
        second = 'frontal'
    selections.append(mne.read_vectorview_selection(
        f'{first}-{second}', info=evoked.info))
    assert len(selections[-1]), len(selections[-1])
for subject in subjects + ['grand-average']:
    if not write_evokeds:
        continue
    evokeds = list()
    for condition in conditions:
        this_evoked = evoked.copy()
        this_evoked.comment = f'{condition} source space'
        this_evoked.nave = 1
        used = np.zeros(len(evoked.ch_names), bool)
        this_data = source_data[subject][sanitize(condition)]
        for ci, sel in enumerate(selections):
            picks = mne.pick_channels(evoked.ch_names, sel, ordered=True)
            assert 10 < len(picks) < 50, len(picks)
            assert not used[picks].any()
            ltc = this_data
            if subject == 'grand-average':
                assert ltc.ndim == 3 and ltc.shape[0] == len(subjects)
                ltc = ltc.mean(axis=0)  # across all subjects
            assert ltc.shape == (len(conditions), len(source_data['times']),)
            ltc = ltc[ci]
            this_evoked.data[picks] = ltc
            used[picks] = True
        used = used.sum()
        assert 50 < used < 200
        evokeds.append(this_evoked)
    assert len(evokeds) == len(conditions)
    mne.write_evokeds(
        evokeds_dir / f'{subject}-ltc-ave.fif', evokeds, overwrite=True)

# Show mean for each condition for each subject, then as grand average.
# Use 2 columns (left/right) and 2 rows (frontal/auditory), each with
# four traces (one per condition); grand average should have the SEM.
