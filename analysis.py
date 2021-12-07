#!/usr/bin/env python
# %%
import os.path as op
import pandas as pd
import pandas_flavor as pf
import janitor  # noqa
import numpy as np
import mne
from mne import io, combine_evoked
from mne.minimum_norm import make_inverse_operator, apply_inverse
%matplotlib inline

# PANDAS parameters
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)
pd.set_option("precision", 2)

# %%


@pf.register_dataframe_method
def str_remove(df, column_name: str, pattern: str = ""):
    """Wrapper to remove string patten from a column"""
    df[column_name] = df[column_name].str.replace(pattern, "")
    return df


@pf.register_dataframe_method
def explode(df: pd.DataFrame, column_name: str, sep: str):
    """Wrapper to expand column after text processing"""
    df["id"] = df.index
    wdf = (
        pd.DataFrame(df[column_name].str.split(sep).fillna("").tolist())
        .stack()
        .reset_index()
    )
    # exploded_column = column_name
    wdf.columns = ["id", "depth", column_name]  # plural form to singular form
    # wdf[column_name] = wdf[column_name].apply(lambda x: x.strip())  # trim
    wdf.drop("depth", axis=1, inplace=True)

    return pd.merge(df, wdf, on="id", suffixes=("_drop", "")).drop(
        columns=["id", column_name + "_drop"]
    )


# %%
# Workspace parameters
seed = np.random.seed(42)
mrsi_data = "/Users/ktavabi/Documents/Projects/nbwr/nbwr/static/all_nbwr_mrs_results_csfcorr_fits_20180417.xlsx"
root = '/Users/ktavabi/MEG/paros-bids/derivatives/mne-bids-pipeline'
subjects_dir = '/Users/ktavabi/freesurfer'

# %%
f = pd.ExcelFile(mrsi_data)
data = f.parse(sheet_name="FSLcorr_metab", header=1)
df = data.clean_names().str_remove("subject", pattern="sub-nbwr")
df.head()
pivot_long_on = df.columns.values[1:]
df = df.pivot_longer(
    column_names=pivot_long_on,
    names_to="name",
    values_to="value",
    sort_by_appearance=True,
)
df[["hemisphere", "mrsi"]] = df.name.apply(
    lambda x: pd.Series(str(x).split("_", 1))
)
df.drop(labels=["name"], axis=1, inplace=True)
df = df.reorder_columns(['subject', 'hemisphere', 'mrsi', 'value']).encode_categorical(column_names=['hemisphere', 'mrsi'])
df['tx'] = df['subject'].apply(lambda x: 'asd' if np.int16(x) < 400 else 'td')
df.head()
df = df[df.subject != '307']
# df.to_csv('out.csv')


# %%
erfs = dict()
for group in ('asd', 'td'):
    erfs[group] = list()
    subjects = df[df['tx'] == group]['subject'].unique()
    assets = [op.join(root, 'sub-%s' % subject, 'meg','sub-%s_task-lexicaldecision_ave.fif') % subject for subject in sorted(subjects)]
    for ii in (0, 1):  # lexical/nonlexical
        erfs[group].append([mne.Evoked(asset, condition=ii, proj=True, kind="average") for asset in assets])


# %%
for group in ('asd', 'td'):
    mne.viz.plot_compare_evokeds(erfs[group],
    legend='upper left', show_sensors='upper right')


# %%
stcs = dict()
residuals = dict()
for group in ('asd', 'td'):
    stcs[group] = list()
    residuals[group] = list()
    subjects = df[df['tx'] == group]['subject'].unique()
    for subject in sorted(subjects):
        epochsfile = op.join(root, 'sub-%s' % subject, 'meg','sub-%s_task-lexicaldecision_epo.fif') % subject
        epochs = mne.read_epochs(epochsfile)
        noise_cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)
        evokedfile = op.join(root, 'sub-%s' % subject, 'meg','sub-%s_task-lexicaldecision_ave.fif') % subject
        erfs = [mne.Evoked(evokedfile, condition=ii, proj=True, kind="average") for ii in [0, 1]]  # lexical/nonlexical
        info = mne.io.read_info(evokedfile)
        src = subjects_dir + '/sub-nbwr%s/bem/sub-nbwr%s_ses-1_fsmempr_ti1100_rms_1_freesurf_hires-oct-6-src.fif' % (subject, subject)
        bem = subjects_dir + '/sub-nbwr%s/bem/sub-nbwr%s-5120-5120-5120-bem-sol.fif' % (subject, subject)
        fwd = mne.make_forward_solution(info, 'fsaverage', src, bem)
        kernel = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
        del fwd
        output = [apply_inverse(erf, kernel, lambda2=0.111111, method='dSPM', pick_ori=None, return_residual=True, verbose=True) for erf in erfs]
    stcs[group].append(output[0])
    residuals[group].append(output[1])
# %%
