import os
import pickle
from typing import Generic

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from skimage.io import imread, imshow
from skimage.transform import resize
#from sklearn import neighbors
#from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from bokeh.plotting import figure,output_file,show

# custom src modules
try:
    from src import utils
except ModuleNotFoundError:
    import utils

class Flylsh():
    ''' Fly Olfactory Network as proposed by Dasgupta et al 2017.

        :param seed: (int) random seed for numpy random generator
        :param num_orn: (int) number of ORNs for fly input (should be dimension of data)
        :param num_pn: (int) number of PNs - must be same as num_orn
        :param num_kc: (int) number of KCs neurons - default should be 40-fold of num_pc
        :param p_pn_kc: (float) probability of PN->KC connection - default ~6 per

        :return apl: ([csr_matrix, ...]), list of LSH for a single datum input.

    '''

    @utils.benchmark
    def __init__(self, seed: int=42,
                 num_orn: int=50,
                 num_pn: int=50,
                 num_kc: int=2000,
                 p_pn_kc: float=0.003):
        np.random.seed(seed)
        self.data = None
        self.labels = None
        self.hashlen = None
        self.orn = np.random.rand(num_orn,)
        self.pn = np.random.rand(num_pn,)
        self.num_kc = num_kc
        self.kc = csr_matrix(np.random.rand(num_kc,num_pn), shape=(num_kc, num_pn))
        # binary probability mask for mean of 6 neurons connected to each Kenyon cell
        self.mask = np.random.choice([0, 1],
                                     size=(self.pn.shape[0],self.kc.shape[0]),
                                     p=[1-p_pn_kc, p_pn_kc])
        #self.W = np.random.rand(self.pn.shape[0], self.kc.shape[0]) * self.mask
        self.W = self.mask
        self.apl = csr_matrix(self.kc,shape=self.kc.shape)

    def orn_to_pn(self):
        ''' Center the Mean step.

            self.orn should be a (1-D) column vector,

            return: mean_centered column vector
        '''
        return self.orn - np.mean(self.orn, axis=0)

    def pn_to_kc(self): # 40-fold expansion
        ''' 40-fold expansion to high-dimensional array.

            self.pn: column vector, same dimensions as self.orn
            self.W: a binary mask, with second dimension a 40-fold increase
            of self.pn[0]

            return: larger sparse, randomly connected array
        '''
        return csr_matrix((self.pn[:, None] * self.W).T)

    def wta(self, hashlen: float=0.05):
        ''' Winner Take All inhibition neuron.

            kc to wta (via APL inhibitory neuron).

            param: hashlen: how long output hash should be truncated to
            as fraction of the input data length (5% default)
            '''

        # of the 2000 neurons, 295 are currently non-zero
        top_k = int(self.kc.shape[0] * hashlen)

        kc_sum = np.sum(self.kc, axis=1) / self.num_kc    # divide by bin width, for discretization
        #kc_abs = abs(kc_sum)

        #eps = kc_abs[np.argsort(abs(kc_abs))][::-1][top_k-1]
        eps = float(kc_sum[np.argsort(kc_sum)][::-1][top_k-1])
        # selects to kth largest element value. all the rest get turned to zero
        apl = kc_sum
        apl[abs(apl) < eps] = 0
        self.apl = csr_matrix(apl)

        return self.apl

    #@utils.benchmark
    def FeedForward(self, datum, hashlen: float=0.05):
        self.hashlen = hashlen
        datum = datum.ravel()
        assert datum.shape == self.orn.shape
        # feed inputs into orn
        self.orn = datum
        # orn to pn, center the means
        self.pn = self.orn_to_pn()
        # pn to kenyon, project to sparse projected
        self.kc = self.pn_to_kc()
        # WTA algorihm
        self.apl = self.wta(hashlen=hashlen)

        return self.apl
