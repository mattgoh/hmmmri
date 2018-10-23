#!/usr/bin/env python

from datetime import datetime
import HMM_fit_images as HMM_fit
import numpy as np
import os, sys
import debug, pdb

##########################
## Inputs
##########################

csv = '/mnt/macdata/groups/imaging_core/matt/projects/HMM/voxelwise.wmaps.QB3_smallsample.csv'
mask = '/mnt/macdata/groups/imaging_core/matt/projects/HMM/mask_80_split0000.nii.gz'
outdir = '/mnt/macdata/groups/imaging_core/matt/projects/HMM/models'

wmin = -3
wmax = 3
wstep = 1

wbins = np.arange(wmin, wmax + wstep, wstep)

##########################
## Run
##########################

mask_struct, mask_image, mask_flat, nz_index, voxel_length = HMM_fit.load_mask(mask)
estimators = HMM_fit.build_estimators(csvfile = csv, max_timepoints = 100)
hmm = HMM_fit.HMM(estimators = estimators, wbins = wbins, nonzero_indices = nz_index, time_delta_min = 300, time_delta_max = 440, Procs = 16, outdir = outdir)
hmm.run()