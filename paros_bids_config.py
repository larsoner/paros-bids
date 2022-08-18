# DONE:
# - Get running
# - Check events are correct
# - Running FreeSurfer recon-all for all subjects
#
# DOING:
# - Creating BEMs and head surfaces for all subjects
# - Run source estimation
# - Check source space images in report
#
# TODO:
# - Use mxne to get 2-dipole solutions
# - Mark bad channels manually
# - Check movement to see if we need movecomp
# - Add run_ssp step
# - Reenable decoding
# - Fix dataset to be anonymized and rerun
# - Update dataset_description.json
# - Add MF stuff to dataset rather than specifying path

from pathlib import Path
import mnefun
this_dir = Path(__file__).parent
mf_path = Path(mnefun.__file__).parent / 'data'

N_JOBS = 4
on_error = "debug"

study_name = "paros-bids"
bids_root = this_dir / 'paros-bids'
interactive = False
sessions = "all"
task = "lexicaldecision"
eeg_bipolar_channels = {
    "HEOG": ("HEOG_left", "HEOG_right"),
    "VEOG": ("VEOG_lower", "VEOG_upper"),
}
subjects = [
    "007",
    "017",
    "038",
    "081",
    "088",
    "107",
    "110",
    "132",
    "135",
    "136",
    "144",
    "215",
    "226",
    "301",
    "309",
    "317",
    "401",
    "404",
    "405",
    "407",
    "409",
    "421",
    "426",
    "427",
    "428",
    "431",
    "432",
    "437",
    "440",
    "442",
    "443",
    "444",
    "447",
    "448",
    "449",
    "451",
]

###############################################################################
# MAXWELL FILTER PARAMETERS
# -------------------------
use_maxwell_filter = True
find_flat_channels_meg = True
find_noisy_channels_meg = True
mf_st_duration = 10.0
mf_head_origin = "auto"
mf_cal_fname = mf_path / 'sss_cal.dat'
mf_ctc_fname = mf_path / 'ct_sparse.fif'
ch_types = ["meg"]
data_type = "meg"

###############################################################################
# FREQUENCY FILTERING
# -------------------
l_freq = None
h_freq = 40.0

#########################################################################
# RESAMPLING
# ----------
decim = 5

#########################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
# reject = dict(mag=3000e-15, grad=3000e-13)
reject = "autoreject_global"

reject_tmin = -0.2
reject_tmax = 1.3

#########################################################################
# EPOCHING
# --------
conditions = ["lexical", "nonlex", "lexical/high", "lexical/low"]
epochs_tmin = -0.2
epochs_tmax = 1.3
baseline = (None, 0)
contrasts = [("lexical", "nonlex"), ("lexical/high", "lexical/low")]
interpolate_bads_grand_average = True

#########################################################################
# ARTIFACT REMOVAL
# ----------------
spatial_filter = None  # TODO: "ssp"

#########################################################################
# DECODING
# --------
decode = False
decoding_metric = "roc_auc"
decoding_n_splits = 5
n_boot = 5000

#########################################################################
# SOURCE ESTIMATION PARAMETERS
# ----------------------------
run_source_estimation = True
bem_mri_images = "T1"
freesurfer_verbose = False
spacing = "oct6"
mindist = 5
source_info_path_update = {"processing": "clean", "suffix": "epo"}
inverse_method = "dSPM"
process_er = True
noise_cov = (None, 0)  # "emptyroom"
