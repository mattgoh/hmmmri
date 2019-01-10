#!/home/sgoh/anaconda2/bin/python
from hmmlearn import hmm
from itertools import permutations
import threading, Queue
import joblib
import numpy as np
import nibabel as ni
import pandas as pd
import matplotlib.pyplot as plt
import csv
import itertools
import time
import pdb
import sys, os
singlelock = threading.Lock()

## 
## we are on local ~/workspace
##

class HMM():
    def __init__(self):
        #self.data   = data
        #self.labels = labels
        #self.n_components = n_components
        self.model = ""
        self.qi    = ""
        self.qj    = ""

    def remap_states_(self, states):
        unique =  np.unique(states)
        perm   = permutations(unique)
        remap  = []
        for i, l in enumerate(list(perm)):
            remap.append(dict(zip(unique, l)))

        return remap

    def accuracy(self, pred, labels):
        pred   = np.asarray(pred)
        labels = np.asarray(labels)
        err_count   = np.where(abs(pred - labels) > 0, 1, 0)
        err_percent = err_count.sum() / float(len(err_count))
        acc         = 1 - err_percent
        print "acc", acc

        return acc

    def infer_state(self, data, labels, lengths):
        cprint(" ## Inferring states")
        # infer S_i
        X                  = np.asarray(data).reshape(-1,1)
        self.qi            = self.model.predict(X, lengths)
        self.qi_remap      = self.remap_states_(self.qi) # remap predicted states
        pred               = [[perm[s] for s in self.qi] for perm in self.qi_remap]
        acc_cumulative     = [self.accuracy(var, labels) for var in pred]
        self.sorted_acc    = np.sort(np.asarray(acc_cumulative)).reshape(-1,1)
        self.sorted_labels = np.argsort(np.asarray(acc_cumulative)).reshape(-1,1)

    def predict_next_state(self, qi, labels_test):
        self.qj = [np.argmax(self.model.transmat_[row]) for row in self.qi]
        remap   = self.remap_states_(self.qj)
        pred    = [[perm[s] for s in self.qj] for perm in remap]
        [self.accuracy(l, labels_test[:, 1]) for l in pred]

    def train(self, data, lengths, n_components):
        #self.model = hmm.GMMHMM(n_components = 3, n_iter = 200, covariance_type = 'full')
        self.model  = hmm.GaussianHMM(n_components = n_components, n_iter = 200, covariance_type = 'full')
        X = np.asarray(data).reshape(-1, 1)
        tic = time.clock()
        self.model.fit(X, lengths)
        toc = time.clock()
        print "Training time:", toc - tic

    def test(self,data, labels, lengths):
        self.infer_state(data, labels, lengths)

def cprint(message):
    print '\033[92m'+message+'\033[0m'

###################################
#    Static/Dynamic Parameters    #
###################################
PROCS                = 8
NUM_SAMPLES          = 100
SEQUENCE_LENGTH      = 4
OUTDIR               = "/mnt/ssd2/HMMsim00"

# States Definition
s0 = {'pdf': 'normal',
      'mu': 0,
      'sigma': 0.08}
s1 = {'pdf': 'normal',
      'mu': 0.4,
      'sigma': 0.08}
s2 = {'pdf': 'normal',
      'mu': 0.8,
      'sigma': 0.08}
s3 = {'pdf': 'normal',
      'mu': 0.8,
      'sigma': 0.08}
# s4 = {}

# s5 = {}

STATES     = [s0, s1, s2, s3]
###################################
#              MAIN               #
###################################
COMPONENTS = len(STATES)
# Train
with open(os.path.join(OUTDIR, "Sim.train.data.seq.len.mu.{:02d}.csv".format(SEQUENCE_LENGTH)), 'rb') as f:
    reader      = csv.reader(f)
    data_train_ = list(itertools.chain.from_iterable(reader))
    data_train  = [float(val) for val in filter(None, data_train_)]
with open(os.path.join(OUTDIR, "Sim.train.labels.seq.len.mu.{:02d}.csv".format(SEQUENCE_LENGTH)), 'rb') as f:
    reader        = csv.reader(f)
    labels_train_ = list(itertools.chain.from_iterable(reader))
    labels_train  = [int(float(val)) for val in filter(None, labels_train_)]
with open(os.path.join(OUTDIR, "Sim.train.lengths.seq.len.mu.{:02d}.csv".format(SEQUENCE_LENGTH)), 'rb') as f:
    reader         = csv.reader(f)
    lengths_train_ = list(itertools.chain.from_iterable(reader))
    lengths_train  = [int(float(val)) for val in filter(None, lengths_train_)]
# Test
with open(os.path.join(OUTDIR, "Sim.test.data.seq.len.mu.{:02d}.csv".format(SEQUENCE_LENGTH)), 'rb') as f:
    reader     = csv.reader(f)
    data_test_ = list(itertools.chain.from_iterable(reader))
    data_test  = [float(val) for val in filter(None, data_test_)]
with open(os.path.join(OUTDIR, "Sim.test.labels.seq.len.mu.{:02d}.csv".format(SEQUENCE_LENGTH)), 'rb') as f:
    reader        = csv.reader(f)
    labels_test_ = list(itertools.chain.from_iterable(reader))
    labels_test  = [int(float(val)) for val in filter(None, labels_test_)]
with open(os.path.join(OUTDIR, "Sim.test.lengths.seq.len.mu.{:02d}.csv".format(SEQUENCE_LENGTH)), 'rb') as f:
    reader        = csv.reader(f)
    lengths_test_ = list(itertools.chain.from_iterable(reader))
    lengths_test  = [int(float(val)) for val in filter(None, lengths_test_)]
# Transition matrix
transmat = np.loadtxt(os.path.join(OUTDIR, 'Transition_matrix.csv'), delimiter = ',')
# Train model
cprint("-------------------------------------------------")
cprint("Training with sequence length mu={:d} sigma=1".format(SEQUENCE_LENGTH))
cprint("-------------------------------------------------")
#
hmm_method = HMM()
hmm_method.train(data = data_train, lengths = lengths_train, n_components = COMPONENTS)
print "#############################"
print "True transmat:\n", transmat
print "Fitted transmat:\n", hmm_method.model.transmat_
df  = pd.DataFrame()
hmm_method.infer_state(data_test, labels_test, lengths_test)
df["Acc.train.len.mu={:02d}".format(SEQUENCE_LENGTH)]  = np.squeeze(hmm_method.sorted_acc)
df["Perm.train.len.mu={:02d}".format(SEQUENCE_LENGTH)] = np.squeeze(hmm_method.sorted_labels)
with open(os.path.join(OUTDIR, "Remappings.current.state.train_seq_len.{:02d}.txt".format(SEQUENCE_LENGTH)), 'w') as fp: # These should all be the same
    for perm in hmm_method.qi_remap:
        fp.write("{:s}\n".format(str(perm)))
df.to_csv(os.path.join(OUTDIR, "Final_results.csv"), index = False)
joblib.dump(hmm_method.model, os.path.join(OUTDIR, "HMM_model.joblib"))