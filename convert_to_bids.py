import os.path as op
import shutil
from pathlib import Path

import mne
import numpy as np
from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
from mnefun import extract_expyfun_events

data_path = "/Users/ktavabi/MEG/paros/"
bids_root = op.join(data_path + "bids")

if op.exists(bids_root):
    shutil.rmtree(bids_root)

raw_files = Path(data_path).rglob("*_meg-raw.fif")

event_id = {
    "low_lexical": 11,
    "high_lexical": 21,
    "low_control": 31,
    "high_control": 41,
    "low_target": 16,
    "high_target": 26,
}

for rr in raw_files:
    raw = mne.io.read_raw_fif(rr, allow_maxshield="yes")
    sid = raw.info["subject_info"]["first_name"][-3:]
    raw.info["line_freq"] = 60
    bids_path = BIDSPath(
        subject=sid, session="01", run="01", task="lexicaldecision", root=bids_root
    )
    write_raw_bids(
        raw,
        bids_path,
        events_data=op.join(rr.parent, op.basename(rr)[:-4] + "-eve.fif"),
        event_id=event_id,
        overwrite=True,
    )
    # ERM
    erm_fname = op.join(data_path, 'sub-nbwr'+sid, 'raw_fif', op.basename(rr)[:-11] + "erm-raw.fif")
    erm = mne.io.read_raw_fif(erm_fname, allow_maxshield="yes")
    erm.info["line_freq"] = 60
    er_date = erm.info["meas_date"].strftime("%Y%m%d")
    er_bids_path = BIDSPath(subject="emptyroom", session=er_date,
                        task="noise", root=bids_root)
    write_raw_bids(erm, er_bids_path, overwrite=True)
