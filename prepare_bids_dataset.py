import os.path as op

import mne
import numpy as np
from mne_bids import (
    BIDSPath,
    get_anat_landmarks,
    print_dir_tree,
    write_anat,
    write_raw_bids,
)
from mnefun import extract_expyfun_events


def score(file_path):
    """Scoring function"""
    events, presses = extract_expyfun_events(file_path)[:2]
    i = np.arange(len(events))
    events[i, 2] -= 1
    mask = events[i, 2] > 0
    events = events[mask]
    presses = np.array(presses)[mask].tolist()

    # get subject performance
    # boolean mask using modulus of target event and 2
    targets = events[:, 2] % 2 == 0
    has_presses = np.array([len(pr) > 0 for pr in presses], bool)
    n_targets = np.sum(targets)
    hits = np.sum(has_presses[targets])
    false_alarms = np.sum(has_presses[~targets])
    misses = n_targets - hits
    print("HMF: %s, %s, %s" % (hits, misses, false_alarms))
    return events


bids_root = "/Volumes/LaCie/MEG/paros-bids"
subjects_dir = "/Volumes/LaCie/freesurfer"
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

event_id = {
    "lexical/low": 11,
    "lexical/high": 21,
    "nonlex/low": 31,
    "nonlex/high": 41,
    "target/low": 16,
    "target/high": 26,
}

print_dir_tree(bids_root, max_depth=3)
datatype = "meg"
bids_path = BIDSPath(root=bids_root, datatype=datatype)

task = "lexicaldecision"
suffix = "meg"

for subject in subjects:
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        suffix=suffix,
        datatype=datatype,
        root=bids_root,
    )

    raw_fname = op.join(
        bids_root, "sub-%s_task-%s_%s.fif" % (subject, task, suffix)
    )
    events_data = score(raw_fname)
    output_path = op.join(bids_root, "..", subject)
    raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=True)
    # specify power line frequency as required by BIDS
    raw.info["line_freq"] = 60

    write_raw_bids(
        raw,
        bids_path,
        events_data=events_data,
        event_id=event_id,
        overwrite=True,
    )
    # MRI scan
    t1_fname = op.join(subjects_dir, f"sub-{subject}", "mri", "T1.mgz")
    # transformation matrix
    trans = mne.read_trans(op.join(bids_root, f"sub-{subject}-trans.fif"))
    t1w_bids_path = BIDSPath(subject=subject, root=bids_root, suffix="T1w")
    landmarks = get_anat_landmarks(
        t1_fname,
        info=raw.info,
        trans=trans,
        fs_subject=f"sub-{subject}",
        fs_subjects_dir=subjects_dir,
    )
    t1w_bids_path = write_anat(
        image=t1_fname,
        bids_path=t1w_bids_path,
        landmarks=landmarks,
        verbose=True,
    )
    anat_dir = t1w_bids_path.directory
    # ERM
    erm_fname = op.join(bids_root, "sub-%s_erm_raw.fif" % subject)
    erm = mne.io.read_raw_fif(erm_fname, allow_maxshield="yes")
    erm.info["line_freq"] = 60
    er_date = erm.info["meas_date"].strftime("%Y%m%d")
    er_bids_path = BIDSPath(
        subject="emptyroom", session=er_date, task="noise", root=bids_root
    )
    write_raw_bids(erm, er_bids_path, overwrite=True)
    print(bids_path)
