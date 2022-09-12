"""Compute source space activations in frontal and auditory labels."""

from pathlib import Path
import sys

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import h5io
from scipy import stats
import mne
from mnefun import clean_brain

this_dir = Path(__file__).parent
sys.path.append(str(this_dir / '..'))

import paros_bids_config

write_evokeds = True
plot_resp = False
plot_nave_figure = False
plot_brain_figure = True
fixed_ori = False

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
tois = ('80-120ms', '150-250ms', '450-850ms')
label_names = [
    'DorsoLateral Prefrontal Cortex-lh',
    'DorsoLateral Prefrontal Cortex-rh',
    'Inferior Frontal Cortex-lh',
    'Inferior Frontal Cortex-rh',
    'Early Auditory Cortex-lh',
    'Early Auditory Cortex-rh',
    # 'Auditory Association Cortex-lh',
    # 'Auditory Association Cortex-rh',
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


extra = '_abs' if not fixed_ori else ''
source_path = this_dir / f'time_courses{extra}.h5'
src = mne.read_source_spaces(
    subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-ico-5-src.fif')
if not source_path.is_file():
    source_data = dict()
    for si, subject in enumerate(subjects):
        print(f'Processing {si + 1:2d}/{len(subjects)} ({subject})', end='')
        subj_dir = mbp_path / subject / 'meg'
        epochs = mne.read_epochs(
            subj_dir / f'{subject}_{task}_proc-clean_epo.fif')
        epochs.equalize_event_counts()
        # Originally 44/condition, ensure at least 25/condition
        assert len(epochs) >= 100, len(epochs)
        inv = mne.minimum_norm.read_inverse_operator(
            subj_dir / f'{subject}_{task}_inv.fif')
        morph = mne.compute_source_morph(
            inv['src'], src_to=src, subjects_dir=subjects_dir, smooth=15)
        source_data[subject] = dict(nave=dict())
        for condition in conditions:
            evoked = epochs[condition].average()
            if fixed_ori:
                pick_ori = 'normal'
                mode = 'mean_flip'
            else:
                pick_ori = None
                mode = 'mean'
            stc = mne.minimum_norm.apply_inverse(
                evoked, inv, method='dSPM', pick_ori=pick_ori)
            stc_fs = morph.apply(stc)
            ltc = mne.extract_label_time_course(
                stc_fs, labels, src, mode=mode)
            if 'times' not in source_data:
                source_data['times'] = stc.times
            assert np.allclose(source_data['times'], stc.times)
            assert ltc.shape == (len(labels), len(stc.times))
            source_data[subject][sanitize(condition)] = ltc
            source_data[subject]['nave'][sanitize(condition)] = evoked.nave
            if condition == conditions[0]:
                print(f' ({evoked.nave} trials)')
    h5io.write_hdf5(source_path, source_data)
source_data = h5io.read_hdf5(source_path)
assert source_data[subjects[0]][conditions[0].replace('/', '_')].shape == (len(label_names), len(source_data['times']))  # noqa

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
    extra_split = False
    if '-lh' in label_name:
        assert '-rh' not in label_name
        first = 'Left'
    else:
        assert '-rh' in label_name
        first = 'Right'
    if 'Association' in label_name:
        assert 'Early' not in label_name
        assert 'Frontal' not in label_name
        assert 'Dorso' not in label_name
        second = 'parietal'
    elif 'Early' in label_name:
        assert 'Frontal' not in label_name
        assert 'Dorso' not in label_name
        second = 'temporal'
    elif 'Frontal' in label_name:
        second = 'frontal'
    else:
        assert 'Dorso' in label_name
        second = 'Vertex'
        extra_split = True

    name = second if extra_split else f'{first}-{second}'
    selections.append(mne.read_vectorview_selection(
        name, info=evoked.info))
    if extra_split:
        comp = np.less if first == 'Left' else np.greater_equal
        orig = len(selections[-1])
        assert orig, orig
        selections[-1] = [
            s for s in selections[-1]
            if comp(evoked.info['chs'][evoked.ch_names.index(s)]['loc'][0], 0)]
        assert np.isclose(orig / 2., len(selections[-1]), atol=3)
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
        for si, sel in enumerate(selections):
            picks = mne.pick_channels(evoked.ch_names, sel, ordered=True)
            picks = picks[~used[picks]]  # vertex and left/right frontal
            assert 10 < len(picks) < 50, len(picks)
            assert not used[picks].any()
            ltc = this_data
            if subject in groups:
                assert ltc.ndim == 3 and ltc.shape[0] == len(groups[subject])
                ltc = ltc.mean(axis=0)  # across all subjects
            assert ltc.shape == (len(label_names), len(source_data['times']),)
            ltc = ltc[si]
            this_evoked.data[picks] = ltc
            used[picks] = True
        used = used.sum()
        assert 50 < used < 235
        evokeds.append(this_evoked)
    assert len(evokeds) == len(conditions)
    mne.write_evokeds(
        results_dir / f'{subject}{extra}-ltc-ave.fif', evokeds, overwrite=True)

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
    if not plot_resp:
        break
    this_data = np.array([
        source_data[subject][sanitize(condition)]
        for condition in conditions])
    if subject in groups:
        m = np.mean(this_data, axis=1)
        s = np.std(this_data, axis=1) / np.sqrt(this_data.shape[1] - 1)
    else:
        m = this_data
        s = None
    assert len(label_names) == 6
    want_shape = (len(conditions), len(label_names), len(source_data['times']))
    assert m.shape == want_shape, (m.shape, want_shape)
    if s is not None:
        assert s.shape == want_shape, (s.shape, want_shape)
    n_row = len(label_names) // 2
    fig, axes = plt.subplots(
        n_row, 2, figsize=(10, 3 * n_row), constrained_layout=True,
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
                    alpha=0.5, zorder=3, color=colors[condition],
                    edgecolor='none')
        ax.axhline(0 if fixed_ori else 1, color='k', zorder=2, lw=1, ls='--')
        sps = ax.get_subplotspec()
        if sps.is_first_col() and sps.is_first_row():
            ax.legend(hs, h_labels, loc='upper right', fontsize='x-small',
                      handlelength=1)
        if sps.is_first_col():
            this_ylabel = label_name[:-3]
            ax.set_ylabel(f'{this_ylabel}\ndSPM (F)', fontsize='small')
        if sps.is_first_row():
            ax.set_title(
                dict(lh='Left', rh='Right')[label_name[-2:]], fontsize='small')
        for key in ('top', 'right'):
            ax.spines[key].set_visible(False)
        if sps.is_last_row():
            ax.set_xlabel('Time (s)')
        ax.set(xlim=source_data['times'][[0, -1]])
        if 'Auditory' in label_name:
            assert 'Frontal' not in label_name
            ax.set(ylim=(-3, 3) if fixed_ori else (0, 4))
        else:
            assert 'Frontal' in label_name or 'Dorso' in label_name
            ax.set(ylim=(-0.5, 0.5) if fixed_ori else (0, 2))
    fig.suptitle(subject)
    fig.savefig(results_dir / f'{subject}{extra}-ltc-ave.png')
    if subject not in groups:
        plt.close(fig)

# Write CSVs
header = ['subject']
label_names_short = [
    ''.join(c for c in ll if c.isupper()) + ll[-2:] for ll in label_names
]
with open(results_dir / f'all{extra}-tois.csv', 'w') as fid:
    for si, subject in enumerate(subjects):
        row = [subject]
        for li, label in enumerate(label_names):
            for toi in tois:
                for condition in conditions:
                    start, stop = map(int, toi.rstrip('ms').split('-'))
                    start = np.argmin(np.abs(source_data['times'] - start / 1e3))
                    stop = np.argmin(np.abs(source_data['times'] - stop / 1e3))
                    if si == 0:
                        header.append('_'.join([
                            label_names_short[li],
                            toi,
                            condition.replace("/", "-"),
                        ]))
                    row.append(
                        f'{source_data[subject][sanitize(condition)][li][start:stop].mean():0.6f}',  # noqa: E501
                    )
        assert len(row) == len(header)
        if si == 0:
            fid.write(','.join(header) + '\n')
        fid.write(','.join(row) + '\n')

if plot_nave_figure:
    for subject in subjects:
        nave = source_data[subject]['nave']
        assert list(nave) == [sanitize(condition) for condition in conditions]
        assert all(val == list(nave.values())[0] for val in nave.values())
    naves = dict()
    for group in ('control', 'asd'):
        naves[group] = np.array(
            [source_data[subject]['nave'][sanitize(conditions[0])]
             for subject in groups[group]], float)
    assert len(naves['control']) == len(naves['asd']) == 16
    fig, ax = plt.subplots(constrained_layout=True)
    ax.violinplot(list(naves.values()))
    t, p = stats.ttest_ind(naves['control'], naves['asd'])
    print(f'Independent t-test: {p=}')
    t, p = stats.ttest_1samp(naves['control'] - naves['asd'], 0)
    print(f'Repeated t-test:    {p=}')

# %%
# Show mean for each condition for each subject, then as grand average.
# Use 2 columns (left/right) and 2 rows (frontal/auditory), each with
# four traces (one per condition); grand average should have the SEM.
colors = {
    'asd_lexical': '#66CCEE',
    'asd_nonlex': '#4477AA',
    'control_lexical': '#EE6677',
    'control_nonlex': '#AA3377',
}
group_nice = dict(asd='ASD', control='Control')
condition_nice = dict(lexical='Lexical', nonlex='Non-lex.')
label_nice = {
    'Early Auditory Cortex': 'Auditory',
    'DorsoLateral Prefrontal Cortex': 'DLPFC',
    'Inferior Frontal Cortex': 'IFC',
}
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
if plot_brain_figure:
    xticks = np.arange(-0.2, 1.21, 0.2)
    brain = mne.viz.Brain(
        'fsaverage', 'both', 'inflated', subjects_dir=subjects_dir,
        background='white', foreground='black', size=(1000, 1000))
    for label in labels:
        brain.add_label(label, borders=False, hemi=label.hemi)
        brain.add_label(label, borders=2, color='k', hemi=label.hemi)
    shape = (len(labels) // 2 + 1, 2)
    rowspan = 1
    fig = plt.figure(figsize=(4.5, 6), constrained_layout=True)
    gs = plt.GridSpec(*shape, figure=fig, bottom=0.2)
    for hi, hemi in enumerate(('lh', 'rh')):
        brain.show_view('lat', hemi=hemi)
        ax = fig.add_subplot(gs[0, hi])
        ax.imshow(clean_brain(brain.screenshot()))
        ax.axis('off')
    brain.close()
    times = source_data['times']
    for li, label_name in enumerate(label_names):
        h_labels = list()
        hs = list()
        ax = fig.add_subplot(gs[li // 2 + rowspan, li % 2])
        for group in ('asd', 'control'):
            for condition in ('lexical', 'nonlex'):
                these_conditions = [
                    f'{condition}/{key}' for key in ('high', 'low')]
                this_data = np.array([[
                    source_data[subject][sanitize(condition)]
                    for condition in these_conditions]
                    for subject in groups[group]], float)
                this_data = this_data.mean(1)  # conditions
                assert this_data.shape == (16, 6, len(times))
                m = np.mean(this_data, axis=0)[li]
                s = np.std(this_data, axis=0)[li] / np.sqrt(len(this_data) - 1)
                h_labels.append(
                    f'{group_nice[group]}\n{condition_nice[condition]}')
                color = colors[f'{group}_{condition}']
                hs.append(ax.plot(times, m, zorder=4, color=color, lw=1)[0])
                ax.fill_between(times, m - s, m + s, alpha=0.5, zorder=3,
                                color=color, edgecolor='none')
        ax.axhline(0 if fixed_ori else 1, color='k', zorder=2, lw=1, ls='--')
        sps = ax.get_subplotspec()
        if sps.is_last_row() and li % 2 == 1:
            fig.legend(
                hs, h_labels, loc='lower center', fontsize='x-small',
                handlelength=1, ncol=4)
        if sps.is_first_col():
            ax.set_ylabel(
                f'{label_nice[label_name[:-3]]}\ndSPM (F)')
        if li // 2 == 0:
            ax.set_title(
                dict(lh='Left', rh='Right')[label_name[-2:]])
        for key in ('top', 'right'):
            ax.spines[key].set_visible(False)
        ax.set(xticks=xticks)
        ax.set(xlim=times[[0, -1]])
        if sps.is_last_row():
            # This is a total hack to get the legend spacing right
            ax.set_xlabel('Time (s)\n\n\n ')
        else:
            ax.set(xticklabels=[''] * len(xticks))
        if 'Auditory' in label_name:
            assert 'Frontal' not in label_name
            ylim = (-3, 3) if fixed_ori else (0.5, 7)
            yticks = np.arange(0.5, 7.1, 0.5)
        elif 'Dorso' in label_name:
            ylim = (-0.5, 0.5) if fixed_ori else (0.5, 2.5)
            yticks = np.arange(0.5, 2.6, 0.5)
        else:
            assert 'Inferior' in label_name
            ylim = (-0.5, 0.5) if fixed_ori else (0.5, 3.5)
            yticks = np.arange(0.5, 3.6, 0.5)
        ax.set(yticks=yticks)
        ax.set(ylim=ylim)
        if not sps.is_first_col():
            ax.set(yticklabels=[''] * len(yticks))
    for ext in ('png', 'pdf'):
        fig.savefig(results_dir / f'brain_traces{extra}.{ext}')
