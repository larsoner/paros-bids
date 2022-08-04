#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""data munging wrappers

authors:    Daniel McCloy
            Alexandre Gramfort
            Kambiz Tavabi

license: MIT
"""

import os
from functools import partial

import numpy as np
import yaml
from xarray import DataArray

paramdir = os.path.join("..", "params")
yamload = partial(yaml.load, Loader=yaml.FullLoader)


def load_params(fname):
    """Load parameters from YAML file."""
    with open(fname, "r", encoding="utf-8") as file:
        params = yamload(file)
    return params


def load_paths(include_inv_params=True):
    """Load necessary filesystem paths."""
    paths = load_params(os.path.join(paramdir, "paths.yaml"))
    if include_inv_params:
        params = load_params(os.path.join(paramdir, "inverse_params.yaml"))
        _dir = f"{params['orientation_constraint']}-{params['estimate_type']}"
        paths["results_dir"] = os.path.join(paths["results_dir"], _dir)
    return paths["data_root"], paths["subjects_dir"], paths["results_dir"], paths["bids_root"]


def load_subjects(skip=True):
    """Load subject IDs."""
    subjects = load_params(os.path.join(paramdir, "subjects.yaml"))
    # skip bad subjects
    if skip:
        skips = load_params(os.path.join(paramdir, "skip_subjects.yaml"))
        subjects = sorted(set(subjects) - set(skips))
    return subjects


def get_skip_regexp(regions=(), skip_unknown=True, prefix=""):
    """Convert an iterable of region names to a label regexp excluding them."""
    unknown = ("unknown", r"\?\?\?")
    if skip_unknown:
        regions = regions + unknown
    if prefix:
        regions = tuple(f"{prefix}-{region}" for region in regions)
    if len(regions) > 0:
        return f"(?!{'|'.join(regions)})"


def get_slug(subject, freq_band, condition, parcellation=None):
    """Assemble a filename slug from experiment parameters."""
    parcellation = "" if parcellation is None else f"{parcellation}-"
    return f"{parcellation}{subject}-{condition}-{freq_band}-band"


def get_halfvec(array, k=0):
    """Convert matrix to (row-wise) vectorized upper triangle."""
    indices = np.triu_indices_from(array, k=k)
    halfvec = array.values[indices] if isinstance(array, DataArray) else array[indices]
    return halfvec