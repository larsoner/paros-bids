"""Compute signed source space activations in frontal and auditory labels."""

from pathlib import Path
import sys

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import h5io
import mne

this_dir = Path(__file__).parent
sys.path.append(str(this_dir / '..'))

import paros_bids_config

write_evokeds = False

results_dir = this_dir / 'results'
results_dir.mkdir(exist_ok=True)

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

# Create "grand-average" subject (and other relevant groups) that is an
# `ndarray` of all subjects so we can plot mean+/-SEM as well
static_dir = this_dir / '..' / 'static'
wb = openpyxl.load_workbook(
    static_dir / 'GABA_subject_information.xlsx')
ws = [ws for ws in wb.worksheets if ws.title == 'Matches'][0]
asd_col, con_col = 1, 4
assert ws.cell(1, asd_col).value == 'ASD', ws.cell(1, asd_col).value
assert ws.cell(1, con_col).value == 'Control', ws.cell(1, con_col).value
asd = list()
con = list()
for ri in range(2, 100):
    val = ws.cell(ri, asd_col).value
    if not val:
        break
    asd.append('sub-' + val.split('_')[-1])
    val = ws.cell(ri, con_col).value
    con.append('sub-' + val.split('_')[-1])
assert set(asd).intersection(set(con)) == set()
missing_match = set(subjects).difference(set(asd).union(set(con)))
if missing_match:
    print(f'Missing from matching map: {sorted(missing_match)}')
missing_con = set(con).difference(set(subjects))
if missing_con:
    print(f'Missing from control data: {sorted(missing_con)}')
# 421 is missing its match
print('  Removing 451 from con')
con.pop(con.index('sub-451'))
missing_asd = set(asd).difference(set(subjects))
if missing_asd:
    print(f'Missing from asd data:     {sorted(missing_asd)}')
    for key in missing_asd:
        print(f'  Removing {key} from asd')
        asd.pop(asd.index(key))

assert len(subjects) == 36, len(subjects)
groups = {
    'grand-average': asd + con,
    'asd': asd,
    'control': con,
}
assert len(asd) == 16, len(asd)
assert len(con) == 16, len(con)
del asd, con
want_sizes = {
    'grand-average': 32,  # only uses the relevant ones
    'asd': 16,
    'control': 16,
}
for key, group in groups.items():
    assert key not in subjects
    source_data[key] = {
        sanitize(condition): np.array(
            [source_data[subject][sanitize(condition)]
             for subject in subjects
             if subject in group], float)
        for condition in conditions
    }
    assert len(group) == want_sizes[key], (len(group), want_sizes[key])
    assert len(source_data[key]['nonlex_high']) == want_sizes[key]

# Export "evoked" for each subject in each condition by using selections
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
for subject in subjects + list(groups):
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
            if subject in groups:
                assert ltc.ndim == 3 and ltc.shape[0] == len(groups[subject])
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
        results_dir / f'{subject}-ltc-ave.fif', evokeds, overwrite=True)

# Show mean for each condition for each subject, then as grand average.
# Use 2 columns (left/right) and 2 rows (frontal/auditory), each with
# four traces (one per condition); grand average should have the SEM.
colors = {
    'lexical/high': '#66CCEE',
    'lexical/low': '#4477AA',
    'nonlex/high': '#EE6677',
    'nonlex/low': '#AA3377',
}
for subject in subjects + list(groups):
    this_data = np.array([
        source_data[subject][sanitize(condition)]
        for condition in conditions])
    if subject in groups:
        m = np.mean(this_data, axis=1)
        s = np.std(this_data, axis=1) / np.sqrt(len(this_data))
    else:
        m = this_data
        s = None
    assert len(label_names) == 4
    want_shape = (len(conditions), len(label_names), len(source_data['times']))
    assert m.shape == want_shape, (m.shape, want_shape)
    if s is not None:
        assert s.shape == want_shape, (s.shape, want_shape)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True,
                             sharex=True)
    axes = axes.ravel()
    for li, (ax, label_name) in enumerate(zip(axes, label_names)):
        hs = list()
        h_labels = list()
        for ci, condition in enumerate(conditions):
            h_labels.append(condition)
            hs.append(ax.plot(
                source_data['times'], m[ci, li], zorder=4,
                color=colors[condition])[0])
            if s is not None:
                ax.fill_between(
                    source_data['times'],
                    m[ci, li] - s[ci, li],
                    m[ci, li] + s[ci, li],
                    alpha=0.25, zorder=3, color=colors[condition],
                    edgecolor='none')
        ax.axhline(0, color='k', zorder=2, lw=1)
        sps = ax.get_subplotspec()
        if sps.is_first_col() and sps.is_first_row():
            ax.legend(hs, h_labels, loc='upper right', fontsize='x-small',
                      handlelength=1)
        ax.set_ylabel(f'{label_name}\ndSPM (F)', fontsize='small')
        for key in ('top', 'right'):
            ax.spines[key].set_visible(False)
        if sps.is_last_row():
            ax.set_xlabel('Time (s)')
        ax.set(xlim=source_data['times'][[0, -1]])
        if 'Auditory' in label_name:
            assert 'Frontal' not in label_name
            ax.set(ylim=(-3, 3))
        else:
            assert 'Frontal' in label_name
            ax.set(ylim=(-0.5, 0.5))
    fig.suptitle(subject)
    fig.savefig(results_dir / f'{subject}-ltc-ave.png')
    if subject not in groups:
        plt.close(fig)
