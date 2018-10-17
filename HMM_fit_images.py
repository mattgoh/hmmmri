from hmmlearn import hmm
import nibabel as ni
import numpy as np
import csv
import os, sys
import pdb

def build_estimators(csvfile, timepoints = 100):
    """ Build PIDN estimators dictionary

    csv order must be PIDN, DCDate, Path 

    """
    f = open(csvfile, 'rb')
    reader = csv.reader(f)

    estimators = OrderedDict()
    for row in reader:
        if "PIDN" not in row[0]:
            # Add PIDN
            PIDN = row[0]
            if not any(s in row[0][:1] for s in ["#", "/", "%"]): # check first character
                if PIDN not in estimators.keys():
                    estimators[PIDN]          = OrderedDict()
                    estimators[PIDN]["dates"] = OrderedDict()
                    estimators[PIDN]["ID"]    = PIDN
            # Add dates
            DATE = row[1]
            if (len(estimators[PIDN].keys()) < timepoints) & \
               (DATE not in estimators[PIDN]["dates"].keys()):
                    # Add paths
                    PATH = row[2]
                    estimators[PIDN]["dates"][DATE] = PATH

    return estimators

def GaussianHMM_fit(X, lengths, n_components):
    """
    hmm init_params:
        's' : startprob
        't' : transmat
        'm' : means
        'c' : covars
        'w' : GMM mixing weights
    """

    model = hmm.GaussianHmm(n_components = n_components, n_iters = 1000)
    model.fit(X, lengths = lengths)

    return model