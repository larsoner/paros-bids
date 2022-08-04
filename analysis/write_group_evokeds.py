#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Average sensor and source ERF data across subjects within groups."""

# %%
import os
import numpy as np
import mne
from paros_helpers import load_paths, load_subjects, load_params


#  configure paths and subjects
data_root, subjects_dir, results_dir, bids_root = load_paths(include_inv_params=False)
deriv_root = f"{bids_root}/derivatives/bids-pipeline"
if not os.path.exists(os.path.join(results_dir, "results")):
    os.makedirs(os.path.join(results_dir, "results"))
    results_dir = os.path.join(results_dir, "results")
param_dir = os.path.join("..", "params")
subjects = load_subjects()
excludes = load_params(os.path.join(param_dir, "exclude.yaml"))

src = mne.read_source_spaces(
    os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-ico-5-src.fif")
)
fsave_vertices = [s["vertno"] for s in src]

# stc categories
handles = {
    "lexical": "lexical",
    "non-lexical": "nonlex",
    "High": "lexicalhigh",
    "Low": "lexicallow",
}
inv_params = load_params(os.path.join(param_dir, "inverse_params.yaml"))
task = "lexicaldecision"


# %%
asd_grp = [
    "sub-007",
    "sub-017",
    "sub-038",
    "sub-081",
    "sub-088",
    "sub-107",
    "sub-110",
    "sub-132",
    "sub-135",
    "sub-136",
    "sub-144",
    "sub-215",
    "sub-226",
    "sub-301",
    "sub-309",
    "sub-317",
]
td_grp = [
    "sub-401",
    "sub-404",
    "sub-405",
    "sub-407",
    "sub-409",
    "sub-421",
    "sub-426",
    "sub-427",
    "sub-428",
    "sub-431",
    "sub-432",
    "sub-437",
    "sub-440",
    "sub-442",
    "sub-443",
    "sub-444",
    "sub-447",
    "sub-448",
    "sub-449",
    "sub-451",
]
for name, group in zip(["asd", "td"], [asd_grp, td_grp]):
    subjects = group
    naves = [[] for _ in range(6)]  # Containers for all the categories
    all_evokeds = [[] for _ in range(6)]
    all_stcs = [[] for _ in range(4)]
    missing = []

    for ix, subject in enumerate(subjects):
        if subject in excludes:
            continue
        print(f"processing subject: {subject}")
        evokeds = mne.read_evokeds(
            os.path.join(deriv_root, subject, "meg", f"{subject}_task-{task}_ave.fif")
        )
        assert len(evokeds) == len(all_evokeds)

        for idx, evoked in enumerate(evokeds):
            all_evokeds[idx].append(evoked)  # Insert to the container
            naves[idx] = evoked.nave  # Save the number of averages

        for idx, handle in enumerate(handles):  # load stcs
            try:
                stc = mne.read_source_estimate(
                    os.path.join(
                        deriv_root,
                        subject,
                        "meg",
                        f"{subject}_task-{task}_{handle}+{inv_params['method']}+morph2fsaverage+hemi-lh.stc",
                    )
                )
            except OSError as noent:
                missing.append((subject, handle))
                print(noent)
            all_stcs[idx].append(stc)
    assert len(all_stcs[0]) == len(subjects)
    # Combine evokeds across individuals
    for idx, evokeds in enumerate(all_evokeds):
        all_evokeds[idx] = mne.combine_evoked(evokeds, "equal")
    mne.evoked.write_evokeds(
        os.path.join(results_dir, f"sub-{name}-average_task-{task}_ave.fif"),
        all_evokeds,
        overwrite=True,
    )
    # container arr for group source data
    data = np.zeros(
        (
            len(all_stcs),  # categories
            len(subjects),  # subjects
            all_stcs[0][0].shape[0],  # vertices
            all_stcs[0][0].shape[1],  # samples
        )
    )
    for idx, (stcs, handle) in enumerate(
        zip(all_stcs, handles)
    ):  # Loop over categories of stcs
        data[idx] = [stc_i.data for stc_i in stcs]  # Average across subjects
        # data[idx] = np.mean(data, axis=0)  # Average across subjects
        grp_stc = mne.SourceEstimate(
            data[idx].mean(0),
            fsave_vertices,
            inv_params["tmin"],
            inv_params["tstep"],
            subject=name,
        )  # Create a SourceEstimate object]
        grp_stc.save(os.path.join(results_dir, f"{name}_{handle}.stc"), overwrite=True)
# %%
