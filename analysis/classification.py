#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Classify data between groups and rank feature importance"""
# %%
import os.path as op

import janitor as jn  # noqa
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

import numpy as np
import pandas as pd
import pandas_flavor as pf
import seaborn as sns
from mne import read_source_estimate
from pandas_profiling import ProfileReport
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

set_matplotlib_formats("retina")
# %%
# PANDAS parameters
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 500)


@pf.register_dataframe_method
def str_remove(_df, column_name: str, pattern: str = ""):
    """Wrapper to remove string patten from a column"""
    _df[column_name] = _df[column_name].str.replace(pattern, "")
    return _df


@pf.register_dataframe_method
def explode(_df: pd.DataFrame, column_name: str, sep: str):
    """Wrapper to expand column after text processing"""
    _df["id"] = _df.index
    w_df = (
        pd.DataFrame(_df[column_name].str.split(sep).fillna("").tolist())
        .stack()
        .reset_index()
    )
    # exploded_column = column_name
    w_df.columns = ["id", "depth", column_name]  # plural form to singular form
    # w_df[column_name] = w_df[column_name].apply(lambda x: x.strip())  # trim
    w_df.drop("depth", axis=1, inplace=True)

    return pd.merge(_df, w_df, on="id", suffixes=("_drop", "")).drop(
        columns=["id", column_name + "_drop"]
    )


# %%
# Workspace parameters
seed = np.random.seed(42)
mrsi_data = (
    "/Users/kam/Documents/Projects/Job/paros-bids/static/mrsiCsfCorrFits-20180417.xlsx"
)
# https://github.com/ydataai/pandas-profiling/issues/954
study_name = "paros-bids"
bids_root = "/Volumes/LaCie/paros-bids"
deriv_root = f"{bids_root}/derivatives/bids-pipeline"
subjects_dir = "/Volumes/LaCie/freesurfer"
payload_dir = "/Users/kam/Documents/Projects/Job/paros-bids/payload"

# %%
f = pd.ExcelFile(mrsi_data)
data = f.parse(sheet_name="FSLcorr_metab", header=1)
_df = data.clean_names().str_remove("subject", pattern="sub-nbwr")
pivot_long_on = _df.columns.values[1:]
_df = _df.pivot_longer(
    column_names=pivot_long_on,
    names_to="name",
    values_to="value",
    sort_by_appearance=True,
)

_df[["hemisphere", "mrsi"]] = _df.name.apply(lambda x: pd.Series(str(x).split("_", 1)))

_df.drop(labels=["name"], axis=1, inplace=True)
_df = _df.reorder_columns(["subject", "hemisphere", "mrsi", "value"]).encode_categorical(
    column_names=["hemisphere", "mrsi"]
)
_df["grp"] = _df["subject"].apply(lambda x: "asd" if np.int16(x) < 400 else "td")
_df["hemisphere"] = _df["hemisphere"].map({"left": "lh", "right": "rh"})
_df = _df[_df.subject != "307"]
_df = pd.pivot_table(
    _df, values="value", index=["subject", "hemisphere", "grp"], columns=["mrsi"]
).reset_index()

# %%
print(_df.head())

subjects = _df["subject"].unique()
print(subjects)
meg_latency = np.zeros((len(subjects), 2, 2))  # subjects*conditions*hemisphere
meg_pos = np.zeros_like(meg_latency)

# Blow-up (subjects * conditions * hemisphere) into labeled TIDY data frame
__df = jn.expand_grid(
    others={"subject": subjects, "condition": [1, 2], "hemisphere": ["lh", "rh"]}
)

for si, subject in enumerate(subjects):
    for ci, condition in enumerate(["lexical", "nonlex"]):
        stc = read_source_estimate(
            op.join(
                deriv_root,
                f"sub-{subject}",
                "meg",
                f"sub-{subject}_task-lexicaldecision_{condition}+dSPM+morph2fsaverage+hemi-lh.stc",
            )
        )
        for hii, hem in enumerate(["lh", "rh"]):
            meg_pos[si, ci, hii], meg_latency[si, ci, hii] = stc.crop(
                tmin=0.150, tmax=0.500
            ).get_peak(hemi=hem)

# %%
nn, cc, hh = meg_latency.shape
xs = nn * cc * hh
stc_data = np.hstack((meg_latency.reshape(xs, 1), meg_pos.reshape(xs, 1)))
stc_data = pd.DataFrame(stc_data, columns=["latency", "pos"])  # unlabeled meg features
_df_meg = pd.concat([__df, stc_data], axis=1, ignore_index=True).clean_names()
_df_meg = _df_meg.rename_columns(
    new_column_names={
        "0": "subject",
        "1": "condition",
        "2": "hemisphere",
        "3": "latency",
        "4": "position",
    }
)
DATASET = pd.merge(_df, _df_meg, on=["subject", "hemisphere"], how="inner")
print(DATASET.describe())


# %%
if not op.isfile(op.join(payload_dir, "profile.html")):
    profile = ProfileReport(
        DATASET,
        title="Pandas Profiling Report",
        vars={"num": {"low_categorical_threshold": 0}},
        explorative=True,
    )
    profile.to_file("profile.html")

sns.pairplot(DATASET, hue="grp")

# %%
# create a pipeline object
pipe = make_pipeline(StandardScaler(), LogisticRegression())
X = DATASET[
    [
        "crpluspcr",
        "gaba",
        "gabaovercr",
        "glu_80ms",
        "gluovergaba",
        "gpcpluspch",
        "mins",
        "naaplusnaag",
        "latency",
    ]
].values
Y = DATASET["grp"].map({"asd": 1, "td": 2}).values
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed)
model = pipe.fit(X_train, y_train)
model_accuracy = accuracy_score(pipe.predict(X_test), y_test)

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed)

# %%
fig, ax = plt.subplots()
feature_names = np.array(
    [
        "crpluspcr",
        "gaba",
        "gabaovercr",
        "glu_80ms",
        "gluovergaba",
        "gpcpluspch",
        "mins",
        "naaplusnaag",
        "latency",
    ]
)
sorted_idx = result.importances_mean.argsort()
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx]
)
ax.set_title("Permutation Importance of each feature")
ax.set_ylabel("Features")
fig.tight_layout()
plt.show()

# %%
cols = [
    "Creatine",
    "GABA",
    "GABA|Creatine",
    "Glutamate",
    "E/I",
    "Choline",
    "Myoinositol",
    "NAA",
    "Latency",
]
_df = DATASET.drop(columns=["condition", "hemisphere", "position"])
fg = sns.FacetGrid(_df, row="grp", despine=True, height=10)

fg.map_dataframe(
    lambda data, color: sns.heatmap(
        data.corr(method="spearman"),
        square=True,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        cmap="BrBG",
        fmt=".2f",
        annot=True,
        mask=np.triu(np.ones_like(data.corr())),
        annot_kws={"size": 100 / np.sqrt(len(data))},
        yticklabels=cols,
        xticklabels=cols,
    )
)

titles = ["ASD", "TD"]

for ax, title in zip(fg.axes.flatten(), titles):
    ax.set_title(title)


# %%
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(
    _df.corr(method="spearman")[["gluovergaba"]].sort_values(
        by="gluovergaba", ascending=False
    ),
    vmin=-1,
    vmax=1,
    annot=True,
    cmap="BrBG",
)
heatmap.set_title(
    "Features Correlating with E/I ratio", fontdict={"fontsize": 12}, pad=16
)

#%% [markdown]
# # Results
# According to the correlation (Spearman's $\rho=-0.25$) between non-word 
# evoked response peak latency and glutamate measurements is a robust 
# indication of abbarant excitatory neurotransmission in ASD subject. 
# Indicating that upto 25% of variance in the data is accounted for by an 
# interaction between MEEG and MRSI features.
# The permutation feature importance is defined to be the decrease in a model 
# score when a single feature value is randomly shuffled [1].

# [1] L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.

