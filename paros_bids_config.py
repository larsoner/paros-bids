import importlib
import functools
import os
import pdb
import traceback
import sys
import copy
import coloredlogs
import logging
from typing import Optional, Union, Iterable
try:
    from typing import Literal
except ImportError:  # Python <3.8
    from typing_extensions import Literal

import numpy as np
import mne
from mne_bids.path import get_entity_vals

study_name = "paros-bids"
bids_root = "/Users/ktavabi/MEG/paros/bids"
deriv_root = "/Users/ktavabi/MEG/paros/bids/derivatives/paros"
subjects_dir = "/Users/ktavabi/freesurfer"
interactive = False
crop = [200, 600]
sessions = "all"
task = str = "lexicaldecision"
eeg_bipolar_channels = {
    "HEOG": ("HEOG_left", "HEOG_right"),
    "VEOG": ("VEOG_lower", "VEOG_upper"),
}
ch_types = ['meg']
###############################################################################
# MAXWELL FILTER PARAMETERS
# -------------------------
# done in 01-import_and_maxfilter.py
use_maxwell_filter = True
mf_reference_run = '01'
find_flat_channels_meg =  True
find_noisy_channels_meg =  True
mf_st_duration =  10.0
mf_head_origin = "auto"
mf_reference_run =  None
mf_cal_fname =  None
mf_ctc_fname =  None
###############################################################################
# STIMULATION ARTIFACT
# --------------------
# used in 01-import_and_maxfilter.py
fix_stim_artifact =  False
stim_artifact_tmin =  0.0
stim_artifact_tmax =  0.01
###############################################################################
# FREQUENCY FILTERING
# -------------------
# done in 02-frequency_filter.py
l_freq =  None
h_freq =  55.0
###############################################################################
# RESAMPLING
# ----------
resample_sfreq = 500
decim = 1
###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
reject = {"grad": 4000e-13, "mag": 4e-12, "eeg": 150e-6}
reject_tmin = -0.2
reject_tmax = 1.3
###############################################################################
# RENAME EXPERIMENTAL EVENTS
# --------------------------
rename_events = dict()
###############################################################################
# EPOCHING
# --------
conditions = ["lexical", "nonlexical"]
conditions = ["lexical/high", "lexical/low"]
epochs_tmin = -0.2
epochs_tmax = 1.3
baseline = (None, 0)
contrasts = [("lexical", "nonlex"), ("lexical/high", "lexical/low")]
###############################################################################
# ARTIFACT REMOVAL
# ----------------
use_ssp = True
###############################################################################
# DECODING
# --------
decode = True
decoding_metric = "roc_auc"
decoding_n_splits = 5
n_boot = 5000
###############################################################################
# GROUP AVERAGE SENSORS
# ---------------------
interpolate_bads_grand_average = True
###############################################################################
# TIME-FREQUENCY
# --------------
time_frequency_conditions = ["lexical", "nonlex"]
###############################################################################
# SOURCE ESTIMATION PARAMETERS
# ----------------------------
bem_mri_images = "auto"
recreate_bem = False
spacing = "oct6"
mindist = float = 5
inverse_method = "dSPM"
process_er = False
noise_cov = ["emptyroom"]


###############################################################################
# ADVANCED
# --------
l_trans_bandwidth = "auto"
h_trans_bandwidth = "auto"
N_JOBS = 4
shortest_event = 1
allow_maxshield = True
log_level = "info"
mne_log_level = "error"
on_error = "debug"