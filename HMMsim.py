from hmmlearn import hmm
from itertools import permutations
import threading, Queue
import numpy as np
import nibabel as ni
import matplotlib.pyplot as plt
import time
import debug
import pdb
import sys, os
singlelock = threading.Lock()

class Simulate_data():
    def __init__(self, states_dict, transmat, samples = 900):
        self.states       = states_dict
        self.transmat     = transmat
        self.n_samples    = samples
        self.n_components = len(states_dict)
        self.queue_ = [Queue.Queue()]
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
        print "#############################"
        print "Checking states definition"
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
        # transmat_ = self.transmat[(int(qi))][:self.n_components] / self.transmat[(int(qi))][:self.n_components].sum()
        # return np.random.choice(np.arange(0, self.n_components), p = transmat_)
        #pdb.set_trace()
        return np.random.choice(np.arange(0, self.n_components), p = self.transmat[int(qi)])

    def generate_state_sequence(self, sequence_length):
        Q = np.abs(np.empty((self.n_samples, sequence_length)))
        # initialize q_i with uniform dist.
        Q[:,0] = np.random.randint(0, self.n_components, size = self.n_samples)
        #
        for i in range(1, sequence_length):
            Q[:, i] = np.asarray([self.next_state_(qi) for qi in Q[:, i-1]])
        Q = Q.astype(int)
        print "#############################"
        print "State sequence:", Q.shape

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
        print "saving test data to nifti..."
        for i in range(procs):
            print "thread", i
            t = threading.Thread(target = self.float_to_nifti_, args = [state_seq, obs_seq, outdir])
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
        #
        self.qi = ""
        self.qj = ""

    def remap_states(self, states):
        unique =  np.unique(states)
        perm   = permutations(unique)
        remap  = []
        for i, l in enumerate(list(perm)):
            remap.append(dict(zip(unique, l)))

        return remap

    def accuracy(self, pred, labels):
        err_count = np.where(abs(pred - labels) > 0, 1, 0)
        err_percent = err_count.sum() / float(len(err_count))
        print "acc", 1- err_percent

    def infer_state(self, data, labels, target = 0):
        """
        target: state i, j, k,...
        """
        print "#############################"
        print "Inferring state {:d}".format(target)
        # infer S_i
        X = data[:, target].reshape(-1,1)
        n = data.shape[0]
        lengths_arr = np.array([X.shape[1]] * int(n))
        self.qi = self.model.predict(X, lengths_arr)
        remap = self.remap_states(self.qi)
        pred = [[perm[s] for s in self.qi] for perm in remap]
        [self.accuracy(l, labels[:, target]) for l in pred]

    def predict_next_state(self, qi, labels_test):
        self.qj = [np.argmax(self.model.transmat_[row]) for row in self.qi]
        remap = self.remap_states(self.qj)
        #pdb.set_trace()
        pred = [[perm[s] for s in self.qj] for perm in remap]
        [self.accuracy(l, labels_test[:, 1]) for l in pred]

    def train(self, data, n_components):
        #self.model = hmm.GMMHMM(n_components = 3, n_iter = 200, covariance_type = 'full')
        self.model = hmm.GaussianHMM(n_components = n_components, n_iter = 200, covariance_type = 'full')
        X = data.flatten().reshape(-1,1)
        lengths_arr = np.array([data.shape[1]] * int(data.shape[0]))
        tic = time.clock()
        self.model.fit(X, lengths_arr)
        toc = time.clock()
        print "Training time:", toc-tic

    def test(self,data, labels, target):
        self.infer_state(data, labels, target)


def run(procs, states, transmat, num_samples, train_sequence_length, outdir):

    #
    #print "#############################"
    #print "Predict next state"
    #hmm_method.predict_next_state(hmm_method.qi, labels_test)

########################### 
#    Static Parameters    #
###########################

np.random.seed(1)
OUTDIR = "/home/sgoh/workspace//simdata"
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

############################ 
#    Dynamic Parameters    #
############################
NUM_SAMPLES = 500
TEST_SEQUENCE_LENGTH = 4

###########################
#          MAIN           #
###########################
states = [s0, s1, s2, s3]

# N components
components = len(states)

# Transition matrix
transmat = np.random.rand(components, components)
transmat /= np.tile(transmat.sum(axis = 1), (components, 1)).T
print transmat
# transmat = np.array([[0.5, 0.3, 0.2],
#                      [0.1, 0.6, 0.3],
#                      [0.1, 0.2, 0.7]])
# transmat /= transmat.sum(axis = 1)

sim = Simulate_data(states_dict = states,
                    transmat    = transmat,
                    samples     = NUM_SAMPLES)
# Test data
state_seq_test = sim.generate_state_sequence(sequence_length = TEST_SEQUENCE_LENGTH)
data_test      = sim.generate_observations(Q = state_seq_test, sequence_length = TEST_SEQUENCE_LENGTH)
labels_test    = state_seq_test.copy()
obs_header     = ','.join(["obs_{:d}".format(n) for n in range(0, TEST_SEQUENCE_LENGTH)])
state_header   = ','.join(["state_{:d}".format(n) for n in range(0, TEST_SEQUENCE_LENGTH)])
np.savetxt(os.path.join(outdir, "Sim.test.data.csv"), data_test, delimiter=",", fmt = "%1.03f", header = obs_header)
np.savetxt(os.path.join(outdir, "Sim.test.labels.csv"), labels_test, delimiter=",", fmt = "%1d", header = state_header)
sim.float_to_nifti_threading_(procs = procs, state_seq = state_seq_test, obs_seq = data_test, outdir = outdir)



for train_seq_len in range(2,10): # lower bound must be min 2
    # init
    components = len(states)
    sim = Simulate_data(states_dict = states,
                        transmat    = transmat,
                        samples     = num_samples)
    # Train data
    print "-------------------------------------------------"
    print "Training with sequence length {:d}".format(s)
    print "-------------------------------------------------"
    #
    state_seq_train = sim.generate_state_sequence(sequence_length = train_sequence_length)
    data_train      = sim.generate_observations(Q = state_seq_train, sequence_length = train_sequence_length)
    labels_train    = state_seq_train.copy()
    np.savetxt(os.path.join(outdir, "Sim.train.data.seq.{:02d}.csv".format(train_sequence_length)), data_train, delimiter=",", fmt = "%1.03f", header = obs_header)
    np.savetxt(os.path.join(outdir, "Sim.train.labels.seq.{:02d}.csv".format(train_sequence_length)), labels_train, delimiter=",", fmt = "%1d", header = state_header)
    #
    hmm_method = HMM()
    hmm_method.train(data_train, n_components = components)
    #
    print "#############################"
    print "True transmat:\n", transmat
    print "Fitted transmat:\n", hmm_method.model.transmat_
    for t in range(0,test_sequence_length):
        hmm_method.test(data_test, labels_test, target = t)


for train_seq_len in range(2,10): # lower bound must be min 2
    run(procs                 = 6,
        states                = states,
        transmat              = transmat,
        num_samples           = NUM_SAMPLES,
        train_sequence_length = train_seq_len,
        outdir                = OUTDIR)
