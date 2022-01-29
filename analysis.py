#!/usr/bin/env python
# %%
import os.path as op
from scipy.misc import derivative
from scipy import stats as stats
import pandas as pd
import pandas_flavor as pf
import janitor  # noqa
import numpy as np
import mne
from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc
from mne import io, combine_evoked, read_source_estimate

%config InlineBackend.figure_format = "retina"
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
mrsi_data = "/Users/ktavabi/Documents/Projects/paros-bids/static/all_nbwr_mrs_results_csfcorr_fits_20180417.xlsx"
bidsroot = "/Volumes/LaCie/MEG/paros-bids"
derivatives = bidsroot + "/derivatives/mne-bids-pipeline/"
subjects_dir = "/Volumes/LaCie/freesurfer"

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
df = df.reorder_columns(
    ["subject", "hemisphere", "mrsi", "value"]
).encode_categorical(column_names=["hemisphere", "mrsi"])
df["grp"] = df["subject"].apply(lambda x: "asd" if np.int16(x) < 400 else "td")
df = df[df.subject != "307"]
df.head()

# %%
subjects = df["subject"].unique()
p_threshold = 1 / (2 ** len(subjects))
fsaverage = '/Volumes/LaCie/freesurfer/fsaverage/bem/fsaverage-oct6-src.fif'
# compute adjacency
adjacency = mne.spatial_src_adjacency(mne.read_source_spaces(fsaverage))
files = [op.join(derivatives, f"sub-{grp}{subj}", "meg", f"sub-{grp}{subj}_
TD = [read_source_spaces(fs) for fs in files]
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjects) - 1)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_jobs=1,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
# %%
assets = dict()
erfs = dict()
conds = ['word', 'nonword']
groups = ['asd', 'td']

for ix, (cond, grp) in enumerate(zip(conds, groups)):
    erfs[cond] = list()
    subjects = df[df["grp"] == grp]["subject"].unique()
    for subj in subjects:
        read_in = op.join(derivatives, f"sub-{grp}{subj}", "meg", f"sub-{grp}{subj}_task-lexicaldecision_ave.fif")
        erfs[cond].append(mne.read_evokeds(read_in, kind='average')[ix])
    assert len(erfs[cond]) == len(subjects)
    assets[grp] = erfs[cond]  # XXX Pydantic here?
mne.viz.plot_compare_evokeds(assets['asd'])