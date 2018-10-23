import _strptime
from collections import OrderedDict
from datetime import datetime
from hmmlearn import hmm
from sklearn.externals import joblib
import nibabel as ni
import numpy as np
import threading, Queue
import csv
import os, sys
import debug, pdb

singlelock = threading.Lock()

def load_mask(mask):
    mask_struct  = ni.load(mask)
    mask_image   = mask_struct.get_data()
    mask_flat    = mask_image.flatten()
    nz_index     = mask_flat.nonzero()[0]
    voxel_length = len(nz_index)

    return mask_struct, mask_image, mask_flat, nz_index, voxel_length

def build_estimators(csvfile, max_timepoints = 100):
    """ Build PIDN estimators dictionary

    csv order must be PIDN, DCDate, Path 

    """

    print "Building estimators"
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
            print PIDN, DATE
            if (len(estimators[PIDN].keys()) < max_timepoints) & \
               (DATE not in estimators[PIDN]["dates"].keys()):
                    # Add paths
                    PATH = row[2]
                    estimators[PIDN]["dates"][DATE] = PATH

    return estimators

def multi_dim_list(num_lists):
    
    """ Create a list of n lists where n = num_lists """

    ndlist = []
    for l in range(0, num_lists):
        newlist = []
        ndlist.append(newlist)

    return ndlist

def md_list_to_array(inlist, num_lists):

    """ Convert multi dim list to array of arrays"""

    for v in range(0, num_lists):
        inlist[v] = np.asarray(inlist[v])

    return inlist

def check_delta(datestring1, datestring2):
    date1 = datetime.strptime(datestring1, "%Y-%m-%d")
    date2 = datetime.strptime(datestring2, "%Y-%m-%d")
    delta = (date2 - date1).days

    return delta

class Setup_subject():
    # PIDN should be a dictionary object
    def __init__(self, ID, DATE, PATH):
        self.id           = ID
        self.date         = DATE
        self.path         = PATH
        self.image        = None    # dict of 3D image matrices
        self.image_array  = None    # dict of 1D arrays
        self.image_masked = None    # dict of 1D arrays
        self.load_status  = False

    def load_image(self):
        image_           = ni.load(self.path)
        self.image       = image_.get_data()
        self.image_array = self.image.flatten()
        self.load_status = True

    def mask_image(self, nonzero_indices):
        self.image_masked = self.image_array[nonzero_indices]

class HMM(threading.Thread):
    # receives PIDN group data from queue
    def __init__(self,
                 estimators,
                 wbins,
                 nonzero_indices,
                 time_delta_min, 
                 time_delta_max,
                 outdir,
                 Procs = 16):
        threading.Thread.__init__(self)

        # IO and threading
        self.procs_ = Procs
        self.queue_ = Queue.Queue()
        self.outdir = outdir
        self.train  = ""
        self.test   = ""

        # Estimators
        self.estimators    = estimators

        # Image properties
        self.nz_indices = nonzero_indices        # array of nonzero indices
        self.n_voxels   = len(nonzero_indices)   # scalar number of voxels        

        # HMM
        self.time_series = 2  # remove hardcode in future
        self.wbins           = wbins
        self.n_hidden_states = len(wbins)
        self.global_obs_list = multi_dim_list(self.n_voxels)

        # Define time delta
        self.time_delta_min = time_delta_min
        self.time_delta_max = time_delta_max

    def append_series(self, v, s1, s2):
        state_i = s1.image_masked[v]
        state_j = s2.image_masked[v]
        singlelock.acquire()
        print s1.id, s1.date, s2.date, v
        self.global_obs_list[v].extend((state_i, state_j))
        singlelock.release()

    def record_time_point(self, f, subjid1, date1, date2, delta):
        
        """ This function is threadlocked """

        try:
            f.write("{:s},{:s},{:s},{:d}\n".format(subjid1, date1, date2, delta))
        except NameError:
            print "File object 'f' has not been opened"
            sys.exit(1)

    def build_observations(self):
        
        """ This function is multithreaded """

        try:
            while True:
                PIDN = self.queue_.get() # dict object
                tps  = len(PIDN["dates"].keys())

                # init segment 1 and segment 2
                s1 = None
                s2 = None
                for t in range(tps - 1):
                    if (s1 is None) and (s2 is None):
                        # We are in initial run or previous segment failed check
                        s1 = Setup_subject(ID   = PIDN["ID"],
                                           DATE = PIDN["dates"].keys()[t],
                                           PATH = PIDN["dates"].values()[t])
                        s2 = Setup_subject(ID   = PIDN["ID"],
                                           DATE = PIDN["dates"].keys()[t+1],
                                           PATH = PIDN["dates"].values()[t+1])
                    else:
                        # Previous segment was good
                        s1 = s2
                        s2 = Setup_subject(ID   = PIDN["ID"],
                                           DATE = PIDN["dates"].keys()[t+1],
                                           PATH = PIDN["dates"].values()[t+1])
                    delta = check_delta(s1.date, s2.date)
                    #
                    if self.time_delta_min <= delta <= self.time_delta_max:
                        # Load images
                        if not s1.load_status:
                            s1.load_image()
                            s1.mask_image(nonzero_indices = self.nz_indices)
                        s2.load_image()
                        s2.mask_image(nonzero_indices = self.nz_indices)

                        # Record time point
                        singlelock.acquire()
                        self.record_time_point(self.train, s1.id, s1.date, s2.date, delta)
                        singlelock.release()
                        #
                        for v in range(0, self.n_voxels):
                            self.append_series(v, s1, s2)
                    else:
                        singlelock.acquire()
                        self.record_time_point(self.test, s1.id, s1.date, s2.date, delta)
                        singlelock.release()
                        s1 = None
                        s2 = None
                #
                self.queue_.task_done()
        except:
            print "Unexpected error:", sys.exc_info()
            #pdb.set_trace()

    def build_length_list(self):
        self.global_len_list = []
        for v in range(0, self.n_voxels):
            self.global_len_list.append(np.array([self.time_series] * (len(self.global_obs_list[v]) / self.time_series)))

    def GaussianHMM_fit(self, X, lengths, n_components):
        """
        hmm init_params:
            's' : startprob
            't' : transmat
            'm' : means
            'c' : covars
            'w' : GMM mixing weights
        """

        model = hmm.GaussianHMM(n_components = n_components, n_iter = 2000)
        model.fit(X, lengths = lengths)

        return model

    def vxTrain(self):
        try:
            while True:
                vidx = self.queue_.get() # get voxel index
                #singlelock.acquire()
                print "v{}".format(vidx)
                #singlelock.release()
                X = np.asarray(self.global_obs_list[vidx]).reshape(-1, 1)
                lengths = self.global_len_list[vidx]
                v_model = self.GaussianHMM_fit(X, lengths, n_components = self.n_hidden_states)
                singlelock.acquire()
                self.models[vidx].append(v_model)
                singlelock.release()
                # save model
                # model.transmat_
                # model.startprob_
        except:
            print "Unexpected error:", sys.exc_info()
            pdb.set_trace()

    def threading_(self):
        self.train = open(os.path.join(self.outdir, 'Train.csv'), 'w')
        self.test  = open(os.path.join(self.outdir, 'Validation.csv'), 'w')
        self.train.write("PIDN,Start_Date,End_Date,Delta\n")
        self.test.write("PIDN,Start_Date,End_Date,Delta\n")
        #
        print "Building observations"
        for i in range(self.procs_):
            t = threading.Thread(target = self.build_observations)
            t.daemon = True
            t.start()
        for pidn in self.estimators:
            self.queue_.put(self.estimators[pidn])
        # block until tasks are done
        self.queue_.join()
        self.train.close()
        self.test.close()

    def vxTrain_threading_(self):
        self.build_length_list()
        self.queue_ = Queue.Queue()
        self.models = multi_dim_list(self.n_voxels)
        print "Training models"
        for i in range(self.procs_):
            t = threading.Thread(target = self.vxTrain)
            t.daemon = True
            t.start()
        for v in range(0, self.n_voxels):
            self.queue_.put(v)
        # block until tasks are done
        self.queue_.join()
        print "Saving models"
        joblib.dump(models, os.path.join(self.outdir, 'Vx_models.pkl'))

    def run(self):
        self.threading_()
        self.vxTrain_threading_()