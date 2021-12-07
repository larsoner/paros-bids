study_name = "paros-bids"
bids_root = "/Users/ktavabi/MEG/paros-bids"
subjects_dir = "/Users/ktavabi/freesurfer"
interactive = False
crop = [200, 600]
sessions = "all"
task = "lexicaldecision"
eeg_bipolar_channels = {
    "HEOG": ("HEOG_left", "HEOG_right"),
    "VEOG": ("VEOG_lower", "VEOG_upper"),
}
# subjects = []

###############################################################################
# MAXWELL FILTER PARAMETERS
# -------------------------
# done in 01-import_and_maxfilter.py
use_maxwell_filter = True
find_flat_channels_meg = True
find_noisy_channels_meg = True
mf_st_duration = 10.0
mf_head_origin = "auto"
mf_cal_fname = "/Users/ktavabi/Github/mnefun/mnefun/data/sss_cal.dat"
mf_ctc_fname = "/Users/ktavabi/Github/mnefun/mnefun/data/ct_sparse.fif"
ch_types = ["meg"]
data_type = "meg"

###############################################################################
# STIMULATION ARTIFACT
# --------------------
# used in 01-import_and_maxfilter.py
fix_stim_artifact = False
stim_artifact_tmin = 0.0
stim_artifact_tmax = 0.01
###############################################################################
# FREQUENCY FILTERING
# -------------------
# done in 02-frequency_filter.py
l_freq = None
h_freq = 55.0
###############################################################################
# RESAMPLING
# ----------
resample_sfreq = 500
decim = 1
###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
# reject = dict(mag=3000e-15, grad=3000e-13)
reject = dict(mag=6000e-15, grad=6000e-13)

reject_tmin = -0.2
reject_tmax = 1.3
###############################################################################
# RENAME EXPERIMENTAL EVENTS
# --------------------------
rename_events = dict()
###############################################################################
# EPOCHING
# --------
conditions = ["lexical", "nonlex", "lexical/high", "lexical/low"]
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
# interpolate_bads_grand_average = True
###############################################################################
# TIME-FREQUENCY
# --------------
time_frequency_conditions = ["lexical", "nonlex"]
###############################################################################
# SOURCE ESTIMATION PARAMETERS
# ----------------------------
run_source_estimation = False
bem_mri_images = "auto"
recreate_bem = False
spacing = "oct6"
mindist = 5
inverse_method = "dSPM"
process_er = True
noise_cov = "emptyroom"
###############################################################################
# ADVANCED
# --------
l_trans_bandwidth = "auto"
h_trans_bandwidth = "auto"
shortest_event = 1
allow_maxshield = True
log_level = "info"
mne_log_level = "error"
on_error = "debug"
