#!/home/sgoh/anaconda2/bin/python

from hmmlearn import hmm
from itertools import permutations
import threading, Queue
import numpy as np
import nibabel as ni
import pandas as pd
import matplotlib.pyplot as plt
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

    def generate_state_sequence(self, sequence_length):
        Q = np.abs(np.empty((self.n_samples, sequence_length)))
        # initialize q_i with uniform dist.
        Q[:,0] = np.random.randint(0, self.n_components, size = self.n_samples)
        #
        for i in range(1, sequence_length):
            Q[:, i] = np.asarray([self.next_state_(qi) for qi in Q[:, i-1]])
        Q = Q.astype(int)
        print "## State sequence:", Q.shape

        return Q

    def generate_observations(self, Q, sequence_length):
        """
        normal or mixture
        Q: state sequence matrix
        """

        obs = np.empty((self.n_samples, sequence_length))
        for seq in range(sequence_length):
            obs[:, seq] = [self.generate_sample_(self.states[q], pdf = self.states[q]["pdf"], size = 1 ) for q in Q[:, seq]]
        print "#############################"
        print "Observations", obs.shape

        return obs

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

    def float_to_nifti_(self, state_seq, obs_seq, outdir):
        """ This function is multithreaded """
        try:
            while True:
                sample_idx = self.queue_[0].get()
                singlelock.acquire()
                for col_idx in range(state_seq.shape[1]):
                    #print sample_idx, col_idx, state_seq[sample_idx, col_idx]
                    obs_image   = self.array_to_image_(obs_seq[sample_idx, col_idx], 0, (121, 145, 121), 121*145*121)
                    state_image = self.array_to_image_(state_seq[sample_idx, col_idx], 0 , (121, 145, 121), 121*145*121)
                    image_ = []
                    image_.extend((obs_image, state_image))
                    image_ = np.asarray(image_)
                    image_ = np.rollaxis(image_, 0, 4)
                    self.save_nifti_(image_, np.eye(4), outdir, filename = "Sim_S{:04d}_T{:02d}".format(sample_idx, col_idx))
                singlelock.release()
                self.queue_[0].task_done()
        except:
            print "Unexpected error:", sys.exc_info()

    def float_to_nifti_threading_(self, procs, state_seq, obs_seq, outdir):
        nifti_dir = os.path.join(outdir, "Nifti")
        if not os.path.exists(nifti_dir):
            os.mkdir(nifti_dir)
        print "saving test data to nifti..."
        for i in range(procs):
            print "thread", i
            t = threading.Thread(target = self.float_to_nifti_, args = [state_seq, obs_seq, nifti_dir])
            t.daemon = True
            t.start()
        for sample_idx in range(state_seq.shape[0]):
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
        err_count   = np.where(abs(pred - labels) > 0, 1, 0)
        err_percent = err_count.sum() / float(len(err_count))
        acc         = 1 - err_percent
        print "acc", acc

        return acc

    def infer_state(self, data, labels, target = 0):
        """
        target: state i, j, k,...
        """

        #print " ## Inferring state {:d}".format(target)
        color(" ## Inferring state {:d}".format(target))
        # infer S_i
        X                  = data[:, target].reshape(-1,1)
        n                  = data.shape[0]
        lengths_arr        = np.array([X.shape[1]] * int(n))
        self.qi            = self.model.predict(X, lengths_arr)
        self.qi_remap      = self.remap_states_(self.qi) # remap predicted states
        pred               = [[perm[s] for s in self.qi] for perm in self.qi_remap]
        acc_cumulative     = [self.accuracy(l, labels[:, target]) for l in pred]
        self.sorted_acc    = np.sort(np.asarray(acc_cumulative)).reshape(-1,1)
        self.sorted_labels = np.argsort(np.asarray(acc_cumulative)).reshape(-1,1)

    def predict_next_state(self, qi, labels_test):
        self.qj = [np.argmax(self.model.transmat_[row]) for row in self.qi]
        remap   = self.remap_states_(self.qj)
        pred    = [[perm[s] for s in self.qj] for perm in remap]
        [self.accuracy(l, labels_test[:, 1]) for l in pred]

    def train(self, data, n_components):
        #self.model = hmm.GMMHMM(n_components = 3, n_iter = 200, covariance_type = 'full')
        self.model  = hmm.GaussianHMM(n_components = n_components, n_iter = 200, covariance_type = 'full')
        X           = data.flatten().reshape(-1,1)
        lengths_arr = np.array([data.shape[1]] * int(data.shape[0]))
        tic = time.clock()
        self.model.fit(X, lengths_arr)
        toc = time.clock()
        print "Training time:", toc-tic

    def test(self,data, labels, target):
        self.infer_state(data, labels, target)

def color(message):
    print '\033[92m'+message+'\033[0m'

def Generate_test_data(sim_instance, test_seq_len, outdir, procs = 6):
    # fixed test sequence length
    state_seq_test = sim_instance.generate_state_sequence(sequence_length = test_seq_len)
    data_test      = sim_instance.generate_observations(Q = state_seq_test, sequence_length = test_seq_len)
    labels_test    = state_seq_test.copy()
    # Save data
    obs_header   = ','.join(["obs_{:d}".format(n) for n in range(0, test_seq_len)])
    state_header = ','.join(["state_{:d}".format(n) for n in range(0, test_seq_len)])
    np.savetxt(os.path.join(OUTDIR, "Transition_matrix.csv"), transmat, delimiter = ",", fmt="%1.03f")
    np.savetxt(os.path.join(OUTDIR, "Sim.test.data.seq_len.{:02d}.csv".format(test_seq_len)), data_test, delimiter=",", fmt = "%1.03f", header = obs_header)
    np.savetxt(os.path.join(OUTDIR, "Sim.test.labels.seq_len.{:02d}.csv".format(test_seq_len)), labels_test, delimiter=",", fmt = "%1d", header = state_header)
    sim_instance.float_to_nifti_threading_(procs = procs, state_seq = state_seq_test, obs_seq = data_test, outdir = outdir)

    return data_test, labels_test

###################################
#    Static/Dynamic Parameters    #
###################################
PROCS                = 8
NUM_SAMPLES          = 900
TEST_SEQUENCE_LENGTH = 5
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

STATES     = [s0, s1, s2, s3]
COMPONENTS = len(STATES)

# Transition matrix
transmat = np.random.rand(COMPONENTS, COMPONENTS)
transmat /= np.tile(transmat.sum(axis = 1), (COMPONENTS, 1)).T
# transmat = np.array([[0.5, 0.3, 0.2],
#                      [0.1, 0.6, 0.3],
#                      [0.1, 0.2, 0.7]])
# transmat /= transmat.sum(axis = 1)
print transmat

###########################
#          MAIN           #
###########################
np.random.seed(1)

sim = Simulate_data(states_dict = STATES,
                    transmat    = transmat,
                    samples     = NUM_SAMPLES)
if False:
    data_test, labels_test = Generate_test_data(sim_instance = sim, 
                                                test_seq_len = TEST_SEQUENCE_LENGTH, 
                                                outdir       = OUTDIR,
                                                procs        = 6)
else:
    # Load test data instead
    data_test   = np.loadtxt(os.path.join(OUTDIR, "Sim.test.data.seq_len.{:02d}.csv".format(TEST_SEQUENCE_LENGTH)), delimiter = "," )
    labels_test = np.loadtxt(os.path.join(OUTDIR, "Sim.test.labels.seq_len.{:02d}.csv".format(TEST_SEQUENCE_LENGTH)), delimiter = "," )

# Vary train sequence length
df  = pd.DataFrame()
df2 = pd.DataFrame()
for train_seq_len in range(2,10): # lower bound must be min 2
    tmp_df  = pd.DataFrame()
    tmp_df2 = pd.DataFrame()
    # Train data
    color("-------------------------------------------------")
    color("Training with sequence length {:d}".format(train_seq_len))
    color("-------------------------------------------------")
    #
    state_seq_train = sim.generate_state_sequence(sequence_length = train_seq_len)
    data_train      = sim.generate_observations(Q = state_seq_train, sequence_length = train_seq_len)
    labels_train    = state_seq_train.copy()
    obs_header      = ','.join(["obs_{:d}".format(n) for n in range(0, train_seq_len)])
    state_header    = ','.join(["state_{:d}".format(n) for n in range(0, train_seq_len)])
    np.savetxt(os.path.join(OUTDIR, "Sim.train.data.seq_len.{:02d}.csv".format(train_seq_len)), data_train, delimiter=",", fmt = "%1.03f", header = obs_header)
    np.savetxt(os.path.join(OUTDIR, "Sim.train.labels.seq_len.{:02d}.csv".format(train_seq_len)), labels_train, delimiter=",", fmt = "%1d", header = state_header)
    #
    hmm_method = HMM()
    hmm_method.train(data_train, n_components = COMPONENTS)
    #
    print "#############################"
    print "True transmat:\n", transmat
    print "Fitted transmat:\n", hmm_method.model.transmat_
    for t in range(0, TEST_SEQUENCE_LENGTH):
        hmm_method.test(data_test, labels_test, target = t)
        acc_    = pd.DataFrame()
        labels_ = pd.DataFrame()
        acc_["Train_len_{:02d}_target_{:02d}".format(train_seq_len, t)]    = np.squeeze(hmm_method.sorted_acc)
        labels_["Train_len_{:02d}_target_{:02d}".format(train_seq_len, t)] = np.squeeze(hmm_method.sorted_labels)
        tmp_df  = pd.concat([tmp_df, acc_], axis = 1)
        tmp_df2 = pd.concat([tmp_df2, labels_], axis = 1)
        with open(os.path.join(OUTDIR, "Remappings.current.state.train_seq_len.{:02d}.txt".format(train_seq_len)), 'w') as fp: # These should all be the same
            for perm in hmm_method.qi_remap:
                fp.write("{:s}\n".format(str(perm)))
    df  = pd.concat([df, tmp_df], axis = 1)
    df2 = pd.concat([df2, tmp_df2], axis = 1)
df.to_csv(os.path.join(OUTDIR, "State_inference_acc.csv"), index = False)
df2.to_csv(os.path.join(OUTDIR, "State_inference_best_permutation.csv"), index = False)