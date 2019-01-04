#!/home/sgoh/anaconda2/bin/python

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
import debug
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

    # def generate_state_sequence(self, sequence_length):
    #     Q = np.abs(np.empty((self.n_samples, sequence_length)))
    #     # initialize q_i with uniform dist.
    #     Q[:,0] = np.random.randint(0, self.n_components, size = self.n_samples)
    #     #
    #     for i in range(1, sequence_length):
    #         Q[:, i] = np.asarray([self.next_state_(qi) for qi in Q[:, i-1]])
    #     Q = Q.astype(int)
    #     print "## State sequence:", Q.shape

    #     return Q

    def generate_state_sequence(self, sequence_length):
        # Generate list of states for a given observaton sequence length

        q = []
        # initialize q_i with uniform dist.
        q.append(np.random.randint(0, self.n_components, size = 1)[0])
        #
        for i in range(1, sequence_length):
            q.append(self.next_state_(q[i-1]))

        return q

    def generate_observations(self, sequence_length, size):
        """
        normal or mixture
        Q: state sequence matrix
        """
        Q   = []
        obs = []
        lengths = np.random.randint(low = sequence_length - 1, high = sequence_length + 2, size = size) # 
        for l in lengths:
            Q += self.generate_state_sequence(l)
        for s in Q:
            obs.append(self.generate_sample_(self.states[s], pdf = self.states[s]["pdf"], size = 1)[0])

        return Q, obs, lengths

    def plot_data(self, data, labels, components):
        colors = ['red', 'blue', 'orange']
        fig    = plt.figure(figsize = (10,4))
        nbins  = 30

        # State 1
        fig.add_subplot(121)
        for comp in range(components):
            ind = np.flatnonzero(labels[:, 0] == comp)
            plt.hist(data[ind, 0], color = colors[comp], bins = nbins, ec ='k', alpha = 0.7)
        plt.title('$s_i$', fontsize = 18)

        # State 2
        fig.add_subplot(122)
        for comp in range(components):
            ind = np.flatnonzero(labels[:, 1] == comp)
            plt.hist(data[ind, 1], color = colors[comp], bins = nbins, ec ='k', alpha = 0.7)
        plt.title('$s_j$', fontsize = 18)
        plt.tight_layout()
        plt.show()

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

def save_data(fname, outdir, data, lengths_list):
    fp  = open(os.path.join(outdir, "{:s}".format(fname)), 'w')
    for idx, series in enumerate(lengths_list):
        start_index = sum(lengths_list[:idx])
        stop_index  = start_index + series
        [fp.write("{:1.03f},".format(val)) for val in data[start_index:stop_index]]
        fp.write("\n")
    fp.close()

def Generate_train_data(sim_instance, train_seq_len, size, outdir, procs = 6):
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

def Load_train_data(seq_len, outdir):
    # Load test data instead
    with open(os.path.join(OUTDIR, "Sim.train.data.seq.len.mu.{:02d}.csv".format(seq_len)), 'rb') as f:
        reader      = csv.reader(f)
        data_train_ = list(itertools.chain.from_iterable(reader))
        data_train  = [float(val) for val in filter(None, data_train_)]
    with open(os.path.join(OUTDIR, "Sim.train.labels.seq.len.mu.{:02d}.csv".format(seq_len)), 'rb') as f:
        reader        = csv.reader(f)
        labels_train_ = list(itertools.chain.from_iterable(reader))
        labels_train  = [int(float(val)) for val in filter(None, labels_train_)]
    with open(os.path.join(OUTDIR, "Sim.train.lengths.seq.len.mu.{:02d}.csv".format(seq_len)), 'rb') as f:
        reader         = csv.reader(f)
        lengths_train_ = list(itertools.chain.from_iterable(reader))
        lengths_train  = [int(float(val)) for val in filter(None, lengths_train_)]

    return data_train, labels_train, lengths_train

def Load_test_data(seq_len, outdir):
    # Load test data instead
    with open(os.path.join(OUTDIR, "Sim.test.data.seq.len.mu.{:02d}.csv".format(seq_len)), 'rb') as f:
        reader     = csv.reader(f)
        data_test_ = list(itertools.chain.from_iterable(reader))
        data_test  = [float(val) for val in filter(None, data_test_)]
    with open(os.path.join(OUTDIR, "Sim.test.labels.seq.len.mu.{:02d}.csv".format(seq_len)), 'rb') as f:
        reader        = csv.reader(f)
        labels_test_ = list(itertools.chain.from_iterable(reader))
        labels_test  = [int(float(val)) for val in filter(None, labels_test_)]
    with open(os.path.join(OUTDIR, "Sim.test.lengths.seq.len.mu.{:02d}.csv".format(seq_len)), 'rb') as f:
        reader        = csv.reader(f)
        lengths_test_ = list(itertools.chain.from_iterable(reader))
        lengths_test  = [int(float(val)) for val in filter(None, lengths_test_)]

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
# transmat = np.array([[0.5, 0.3, 0.2],
#                      [0.1, 0.6, 0.3],
#                      [0.1, 0.2, 0.7]])
# transmat /= transmat.sum(axis = 1)
print transmat
np.random.seed(1)

sim = Simulate_data(states_dict = STATES,
                    transmat    = transmat,
                    samples     = NUM_SAMPLES)
if False:
    data_test, labels_test, lengths_test = Generate_test_data(sim_instance = sim, 
                                                              test_seq_len = SEQUENCE_LENGTH,
                                                              size         = NUM_SAMPLES,
                                                              outdir       = OUTDIR,
                                                              procs        = PROCS)
else:
    # Load test data instead
    data_test, labels_test, lengths_test = Load_test_data(seq_len = SEQUENCE_LENGTH,
                                                          outdir  = OUTDIR)
if False:
    data_train, labels_train, lengths_train = Generate_train_data(sim_instance  = sim,
                                                                  train_seq_len = SEQUENCE_LENGTH, 
                                                                  size          = NUM_SAMPLES,
                                                                  outdir        = OUTDIR,
                                                                  procs         = PROCS)
else:
    data_train, labels_train, lengths_train = Load_test_data(seq_len = SEQUENCE_LENGTH,
                                                             outdir  = OUTDIR)
# Train model
cprint("-------------------------------------------------")
cprint("Training with sequence length mu={:d} sigma=1".format(SEQUENCE_LENGTH))
cprint("-------------------------------------------------")
#
hmm_method = HMM()
hmm_method.train(data = data_train, lengths = lengths_train, n_components = COMPONENTS)
np.savetxt(os.path.join(OUTDIR, "Transition_matrix.csv"), transmat, delimiter = ",", fmt="%1.03f")
np.savetxt(os.path.join(OUTDIR, "Fitted_transmat.train_seq_len.{:02d}".format(SEQUENCE_LENGTH)), hmm_method.model.transmat_)
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