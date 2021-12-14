#!/usr/bin/env python

import os.path as op
import shutil

import numpy as np
import matplotlib.pyplot as plt

from nilearn.plotting import plot_anat

import mne
from mne.datasets import sample
from mne import head_to_mri

from mne_bids import (write_raw_bids, BIDSPath, write_anat, get_anat_landmarks,
                      get_head_mri_trans, print_dir_tree)


subject = '007'
# Get the path to  MRI scan
t1_fname = op.join('sub-nbwr%s', 'mri', 'T1.mgz' % subject)

# Load the transformation matrix and show what it looks like
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
trans = mne.read_trans(trans_fname)
print(trans)