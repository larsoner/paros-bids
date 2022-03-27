#%%
import os.path as op
from scipy.misc import derivative
from scipy import stats as stats
import pandas as pd
import pandas_flavor as pf
import janitor  as jn # noqa
import numpy as np
import seaborn as sns
import mne
from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc
from mne import io, combine_evoked, read_source_estimate

%config InlineBackend.figure_format = "retina"
%matplotlib inline

# PANDAS parameters
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 500)

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
study_name = "paros-bids"
bids_root = "/Volumes/LaCie/MEG/paros-bids"
deriv_root = f"{bids_root}/derivatives/bids-pipeline"
subjects_dir = "/Volumes/LaCie/freesurfer"

# %%
f = pd.ExcelFile(mrsi_data)
data = f.parse(sheet_name="FSLcorr_metab", header=1)
df = data.clean_names().str_remove("subject", pattern="sub-nbwr")
pivot_long_on = df.columns.values[1:]
df = df.pivot_longer(
    column_names=pivot_long_on,
    names_to="name",
    values_to="value",
    sort_by_appearance=True)

df[["hemisphere", "mrsi"]] = df.name.apply(
    lambda x: pd.Series(str(x).split("_", 1)))

df.drop(labels=["name"], axis=1, inplace=True)
df = df.reorder_columns(["subject", "hemisphere", "mrsi", "value"]).encode_categorical(column_names=["hemisphere", "mrsi"])
df["grp"] = df["subject"].apply(lambda x: "asd" if np.int16(x) < 400 else "td")
df["hemisphere"] = df["hemisphere"].map({"left": "lh", "right": "rh"})
df = df[df.subject != "307"]
df = pd.pivot_table(df, values = "value", index=["subject", "hemisphere", "grp"], columns=["mrsi"]).reset_index()
df.head()

# %%
subjects = df["subject"].unique()
print(subjects)
meg_data = np.zeros((len(subjects), 2, 2, 1))  # subjects*conditions*hemisphere
_df = jn.expand_grid(others={"subject":subjects, "condition":[1,2], "hemisphere":["lh", "rh"]})

for si, subject in enumerate(subjects):
    for ci, condition in enumerate(["lexical", "nonlex"]):
        stc =  read_source_estimate(op.join(deriv_root, f"sub-{subject}", "meg",
            f"sub-{subject}_task-lexicaldecision_{condition}+dSPM+morph2fsaverage+hemi-lh.stc"))
        for hii, hem in enumerate(["lh", "rh"]):
            _, meg_data[si, ci, hii] = stc.get_peak(hemi=hem)

l,m,n,r = meg_data.shape
stc_data = meg_data.reshape(l*m*n,1)
stc_data = pd.DataFrame(stc_data, columns =["latency"])  # unlabeled
df_meg = pd.concat([_df, stc_data], axis=1, ignore_index=True).clean_names()
df_meg = df_meg.rename_columns(new_column_names={"0":"subject", "1":"condition", "2":"hemisphere", "3":"latency"})
df_meg.info()
dataset = pd.merge(df, df_meg, on=["subject", "hemisphere"], how="inner")
dataset.head()
dataset.describe()

# %%
dataset.info()
profile = ProfileReport(dataset, title="Pandas Profiling Report", explorative=True)
profile.to_file("profile.html")
sns.pairplot(dataset, hue='grp')

# %%
dataset.columns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
 )
X = dataset[['crpluspcr', 'gaba', 'gabaovercr', 'glu_80ms', 'gluovergaba', 'gpcpluspch', 'mins', 'naaplusnaag', 'latency']].values
X.shape
Y = dataset["grp"].map({"asd": 1, "td": 2}).values
Y.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=seed)
model = pipe.fit(X_train, y_train)
model_accuracy = accuracy_score(pipe.predict(X_test), y_test)
model_accuracy
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed)

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
feature_names = np.array(['crpluspcr', 'gaba', 'gabaovercr', 'glu_80ms', 'gluovergaba', 'gpcpluspch', 'mins', 'naaplusnaag', 'latency'])
sorted_idx = result.importances_mean.argsort()
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx]
)
ax.set_title("Permutation Importance of each feature")
ax.set_ylabel("Features")
fig.tight_layout()
plt.show()

# The permutation feature importance is defined to be the decrease in a model score when a single feature value is randomly shuffled [1].

# [1] L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001.
