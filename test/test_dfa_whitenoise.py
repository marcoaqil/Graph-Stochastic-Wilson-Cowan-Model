# import mkl
# mkl.set_num_threads(17)

import numpy as np
import scipy as sp 
from joblib import Parallel, delayed

import os
import glob
import fathon
from natsort import natsorted
from fathon import fathonUtils as fu


nsim = 120
fs = 500


def generate_pink_noise(n_instances, n_samples):
    # Frequencies for FFT
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)

    # 1/sqrt(f) amplitude spectrum for pink noise
    amplitude = np.where(freqs == 0, 0, 1 / np.sqrt(freqs))

    # Initialize array for pink noise
    pink_noise = np.zeros((n_instances, n_samples), dtype=np.float32)

    for i in range(n_instances):
        # Generate random phases
        random_phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))

        # Create frequency spectrum
        spectrum = amplitude * random_phases

        # Inverse FFT to obtain pink noise
        noise = np.fft.irfft(spectrum, n=n_samples)

        # Normalize the noise
        pink_noise[i] = (noise - np.mean(noise)) / np.std(noise)

    return pink_noise

# Parameters
n_instances = 20484
n_samples = 300000

# Generate pink noise
#sims_g_noise = generate_pink_noise(n_instances, n_samples)

sims_g_noise = np.random.default_rng().normal(0, 1, size=(20484,300000))

#sos_alpha = sp.signal.firwin(int(167/2), (6,11), fs=fs, pass_zero='bandpass')
#sos_alpha = sp.signal.firwin(int(91/2), (11,25), fs=fs, pass_zero='bandpass')
#sos_alpha = sp.signal.firwin(int(333/2), (3,6), fs=fs, pass_zero='bandpass')

sos_alpha = sp.signal.firwin(int(41/2), (25, 40), fs=fs, pass_zero='bandpass')

#sos_alpha = sp.signal.firwin(101, (13,30), fs=fs, pass_zero='bandpass')

#theta
#sos_alpha = sp.signal.firwin(101, (4,8), fs=fs, pass_zero='bandpass')

filtered = sp.signal.lfilter(sos_alpha, 1, sims_g_noise)

#filtered = sims_g_noise
#sos_alpha = sp.signal.butter(N=4, Wn=[6,11], btype='bandpass', fs=fs, output='sos')

#filtered = sp.signal.sosfilt(sos_alpha, sims_g_noise)

del sims_g_noise

env = np.abs(sp.signal.hilbert(filtered))

del filtered



# env = np.abs(np.random.default_rng().normal(0, 1, size=(20484,300000)))



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


    
np.save('/data1/projects/dumoulinlab/Lab_members/Marco/NFsim/DFAexp_whitenoise_2540gamma_125sec_polyorder3_adaptaps.npy',all_H)    