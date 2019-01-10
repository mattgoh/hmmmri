#!/home/sgoh/anaconda2/bin/python
from .. import debug
from hmmlearn import hmm
from itertools import permutations
import threading, Queue
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

class Simulate_data():
    def __init__(self, states_dict, transmat, samples = 900):
        self.states       = states_dict
        self.transmat     = transmat
        self.n_samples    = samples
        self.n_components = len(states_dict)
        self.queue_       = [Queue.Queue()]
        #
        self.check_states_def_()

    def mixture_pdf_(self, mean, sigma, scale, size):
        """
        Generate mixture pdf of normal and exponential distribution using 3000 samples

        mean (normal)
        sigma (normal)
        scale (exponential)
        """

        norm = np.random.normal(loc = mean, scale = sigma, size = 50000)
        exp  = np.random.exponential(scale = scale, size = 50000)
        self.mixture_ = np.concatenate([norm, exp])

    def check_states_def_(self):
        print "## Checking states definition"
        for q in self.states:
            try:
                q["pdf"]
                if q["pdf"] == "normal":
                    try:
                        [q[param] for param in ["mu", "sigma"]]
                    except KeyError, e:
                        print 'KeyError: missing paramater {:s}'.format(str(e))
                elif q["pdf"] == "mixture":
                    try:
                        [q[param] for param in ["mu", "sigma", "scale"]]
                    except KeyError, e:
                        print 'KeyError: missing parameter {:s} in {:s}'.format(str(e),q)
                else:
                    raise ValueError, "Unexpected pdf value {:s} in {:s}".format(q["pdf"], q)
            except KeyError, e:
                print 'KeyError: missing {:s} in {:s}'.format(str(e),q)
                sys.exit(1)

    def generate_sample_(self, state, pdf = 'normal', size = 1):
        
        """ Generate 1 sample from specified distribution """

        if pdf == 'normal':
            x = np.random.normal(loc = state['mu'], scale = state['sigma'], size = size)
        elif pdf == 'mixture':
            x = np.random.choice(self.mixture_, size = size, replace = True)
        else:
            print "unrecognized type {:s}".format(state); sys.exit(1)

        return x

    def next_state_(self, qi):
        return np.random.choice(np.arange(0, self.n_components), p = self.transmat[int(qi)])

    def generate_state_sequence(self, sequence_length):
        # Generate list of states for a given observaton sequence length

        q = []
        # initialize q_i with uniform dist.
        q.append(np.random.randint(0, self.n_components, size = 1)[0])
        #
        for i in range(1, sequence_length):
            q.append(self.next_state_(q[i-1]))

        return q

    def generate_sequence_length_(self, seq_length_mu, seq_length_std, size):
        lengths_arr = np.random.randint(low = seq_length_mu - seq_length_std, high = seq_length_mu + seq_length_std +1, size = size)

        return lengths_arr

    def generate_observations(self, sequence_length, size):
        """
        normal or mixture
        Q: state sequence matrix
        """
        Q   = []
        obs = []
        lengths_arr = self.generate_sequence_length_(seq_length_mu = sequence_length, seq_length_std = 1, size = size)
        for l in lengths_arr:
            Q += self.generate_state_sequence(l)
        for s in Q:
            obs.append(self.generate_sample_(self.states[s], pdf = self.states[s]["pdf"], size = 1)[0])

        return Q, obs, lengths_arr

    def array_to_image_(self, array, nz_indices, target_shape, target_num_voxels):
        image_ = np.zeros(target_num_voxels)
        image_[nz_indices] = array
        final_image = np.reshape(image_, target_shape)

        return final_image

    def save_nifti_(self, data, affine, outdir, filename):
        image   = ni.Nifti1Image(data, affine, header = None)
        outfile = os.path.join(outdir, filename)
        image.to_filename(outfile)

    def float_to_nifti_(self, state_seq, obs_seq, lengths_seq, outdir):
        """ This function is multithreaded """
        try:
            while True:
                sample_idx  = self.queue_[0].get() # index in lengths list
                #
                series      = lengths_seq[sample_idx]                
                start_index = sum(lengths_seq[:sample_idx])
                #
                singlelock.acquire()
                for tp in range(series):
                    obs_image   = self.array_to_image_(obs_seq[start_index+tp], 0 , (121, 145, 121), 121*145*121)
                    state_image = self.array_to_image_(state_seq[start_index+tp]+100, 0 , (121, 145, 121), 121*145*121) # add 100 to remove ambiguity of 0s in image
                    image_ = []; image_.extend((obs_image, state_image))
                    image_ = np.asarray(image_)
                    image_ = np.rollaxis(image_, 0, 4)
                    self.save_nifti_(image_, np.eye(4), outdir, filename = "Sim_S{:04d}_T{:02d}".format(sample_idx, tp))
                singlelock.release()
                self.queue_[0].task_done()
        except:
            print "Unexpected error:", sys.exc_info()

    def float_to_nifti_threading_(self, procs, state_seq, obs_seq, lengths_seq, outdir):
        nifti_dir = os.path.join(outdir, "Nifti")
        if not os.path.exists(nifti_dir):
            os.mkdir(nifti_dir)
        print "saving test data to nifti..."
        for i in range(procs):
            print "thread", i
            t = threading.Thread(target = self.float_to_nifti_, args = [state_seq, obs_seq, lengths_seq, nifti_dir])
            t.daemon = True
            t.start()
        for sample_idx in range(len(lengths_seq)):
            self.queue_[0].put(sample_idx)
        # block until tasks are done
        self.queue_[0].join()

def cprint(message):
    print '\033[92m'+message+'\033[0m'

def save_data(fname, outdir, data, lengths_list):
    cprint("Saving data")
    fp  = open(os.path.join(outdir, "{:s}".format(fname)), 'w')
    for idx, series in enumerate(lengths_list):
        start_index = sum(lengths_list[:idx])
        stop_index  = start_index + series
        [fp.write("{:1.03f},".format(val)) for val in data[start_index:stop_index]]
        fp.write("\n")
    fp.close()

def Generate_train_data(sim_instance, train_seq_len, size, outdir, procs = 6):
    cprint("Generating train data")
    state_seq_train, data_train, lengths_train = sim_instance.generate_observations(sequence_length = train_seq_len, size = size)
    labels_train = list(state_seq_train)
    # Save data
    save_data(fname        = "Sim.train.data.seq.len.mu.{:02d}.csv".format(train_seq_len),
              outdir       = outdir,
              data         = data_train,
              lengths_list = lengths_train)
    # Save labels
    save_data(fname        = "Sim.train.labels.seq.len.mu.{:02d}.csv".format(train_seq_len),
              outdir       = outdir,
              data         = labels_train,
              lengths_list = lengths_train)
    #
    np.savetxt(os.path.join(OUTDIR, "Sim.train.lengths.seq.len.mu.{:02d}.csv".format(train_seq_len)), lengths_train, delimiter=",", fmt = "%1d")

    return data_train, labels_train, lengths_train

def Generate_test_data(sim_instance, test_seq_len, size, outdir, procs = 6):
    cprint("Generating test data")
    # fixed test sequence length
    state_seq_test, data_test, lengths_test = sim_instance.generate_observations(sequence_length = test_seq_len, size = size)
    labels_test = list(state_seq_test)
    # Save data
    save_data(fname        = "Sim.test.data.seq.len.mu.{:02d}.csv".format(test_seq_len),
              outdir       = outdir,
              data         = data_test,
              lengths_list = lengths_test)
    # Save labels
    save_data(fname        = "Sim.test.labels.seq.len.mu.{:02d}.csv".format(test_seq_len),
              outdir       = outdir,
              data         = labels_test,
              lengths_list = lengths_test)
    #
    np.savetxt(os.path.join(OUTDIR, "Sim.test.lengths.seq.len.mu.{:02d}.csv".format(test_seq_len)), lengths_test, delimiter=",", fmt = "%1d")
    sim_instance.float_to_nifti_threading_(procs = procs, state_seq = state_seq_test, obs_seq = data_test, lengths_seq = lengths_test, outdir = outdir)

    return data_test, labels_test, lengths_test

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
# Transition matrix
transmat = np.random.rand(COMPONENTS, COMPONENTS)
transmat /= np.tile(transmat.sum(axis = 1), (COMPONENTS, 1)).T # normalize so rowsum = 1
cprint("Saving transmat")
np.savetxt(os.path.join(OUTDIR, "Transition_matrix.csv"), transmat, delimiter = ",", fmt="%1.03f")
print transmat
np.random.seed(1)

sim = Simulate_data(states_dict = STATES,
                    transmat    = transmat,
                    samples     = NUM_SAMPLES)
data_train, labels_train, lengths_train = Generate_train_data(sim_instance  = sim,
                                                              train_seq_len = SEQUENCE_LENGTH, 
                                                              size          = NUM_SAMPLES,
                                                              outdir        = OUTDIR,
                                                              procs         = PROCS)
data_test, labels_test, lengths_test = Generate_test_data(sim_instance = sim, 
                                                          test_seq_len = SEQUENCE_LENGTH,
                                                          size         = NUM_SAMPLES,
                                                          outdir       = OUTDIR,
                                                          procs        = PROCS)