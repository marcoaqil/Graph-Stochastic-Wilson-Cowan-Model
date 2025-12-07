#import mkl
#mkl.set_num_threads(25)

import numpy as np
import scipy as sp 
from joblib import Parallel, delayed

import os
import glob
import fathon
from natsort import natsorted
from fathon import fathonUtils as fu


paths_e = natsorted(glob.glob('/data1/projects/dumoulinlab/Lab_members/Marco/NFsim/10min_beta_sim_hcpmeg_20khz_fftfilterdown500hz/beta_I_activity*.npz'))
#from 20k to 1k hertz to avoid storing too much data
# dsample_1 = 20
# nsim = 120
# #to 250hz for faster calculations
# dsample_2 = 4
# fs=20000//(dsample_1*dsample_2)
# sims = np.concatenate([np.load(p)[:,::dsample_2] for p in paths_e[:nsim]],axis=-1)

nsim = 120
fs = 500

sims = np.concatenate([np.load(p)['arr_0'] for p in paths_e[:nsim]],axis=-1)

#sims[10000:] = 0

eigenvectors = np.load('/data1/projects/dumoulinlab/Lab_members/Marco/NFsim/eigvecs_DTI_fgCCfix_subcortex_dti50.npy')

sims_g = np.dot(eigenvectors,sims)

del sims

# from scipy.fftpack import rfft, irfft

# def phaseScrambleTS(ts):
#     """Returns a TS: original TS power is preserved; TS phase is shuffled."""
#     fs = rfft(ts)
#     # rfft returns real and imaginary components in adjacent elements of a real array
#     pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
#     phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
#     phase_fsr = phase_fs.copy()
#     np.random.shuffle(phase_fsr)
#     # use broadcasting and ravel to interleave the real and imaginary components. 
#     # The first and last elements in the fourier array don't have any phase information, and thus don't change
#     fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
#     fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
#     tsr = irfft(fsrp)
#     return tsr

# sims_g_scrambled = np.array(Parallel(n_jobs=25,verbose=1)(delayed(phaseScrambleTS)(sims_g[i]) \
#                                                         for i in range(sims_g.shape[0])))

# del sims_g

#sos_alpha = sp.signal.firwin(int(167/2), (6,11), fs=fs, pass_zero='bandpass')#, window='blackman')

#sos_alpha = sp.signal.firwin(int(91/2), (11,25), fs=fs, pass_zero='bandpass')
#sos_alpha = sp.signal.firwin(int(333/2), (3,6), fs=fs, pass_zero='bandpass')

sos_alpha = sp.signal.firwin(int(41/2), (25, 40), fs=fs, pass_zero='bandpass')


#here
filtered = sp.signal.lfilter(sos_alpha, 1, sims_g)
#sos_alpha = sp.signal.butter(N=4, Wn=[6,11], btype='bandpass', fs=fs, output='sos')

#filtered = sp.signal.sosfilt(sos_alpha, sims_g)

#filtered = sims_g

del sims_g


env = np.abs(sp.signal.hilbert(filtered))

del filtered

#np.save('/data1/projects/dumoulinlab/Lab_members/Marco/images/old/NFsim/I_alpha812_10min_20khzsim_deci250_envelope.npy',env)

min_window = 1
max_window = 25
polOrd = 3

winSizes = np.logspace(np.log10(min_window*fs),np.log10(max_window*fs),dtype='int',num=20)#fu.linRangeByStep(2*fs, 20*fs, step=100)
revSeg = True

def DFA_H(tseries):

    a = fu.toAggregated(tseries)

    pydfa = fathon.DFA(a)


    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    H, H_intercept  = pydfa.fitFlucVec(logBase=10)

    return H


all_H = Parallel(n_jobs=25)(delayed(DFA_H)(env[vx]) for vx in range(env.shape[0]))
    
np.save('/data1/projects/dumoulinlab/Lab_members/Marco/NFsim/DFAexp_I_2540gamma_hcpmeg_10min_20khzsim_fftfilterdown500hz_125sec_polyorder3_adaptaps.npy',all_H)    