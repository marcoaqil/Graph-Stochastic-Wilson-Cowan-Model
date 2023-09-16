import mkl
mkl.set_num_threads(191)

import numpy as np
import scipy as sp 

from scipy import stats, io, sparse
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.analysis import *
from core.simulation import *

eigenvalues = np.load('/projects/0/vuse0612/SM-pRF/NFm/eigvals_DTI_fgCCfix_subcortex_dti50.npy')
eigenvectors = np.load('/projects/0/vuse0612/SM-pRF/NFm/eigvecs_DTI_fgCCfix_subcortex_dti50.npy')

better_result=dict(x=np.load('/projects/0/vuse0612/SM-pRF/NFm/de_fitting_23_subc_dti50.npy'))

Graph_Kernel='Damped Wave'

aEE=better_result['x'][0]
aIE=better_result['x'][1]
aEI=better_result['x'][2]
aII=better_result['x'][3]
dE=better_result['x'][4]
dI=better_result['x'][5]
P=better_result['x'][6]
Q=better_result['x'][7]
sEE=better_result['x'][8]
sIE=better_result['x'][9]
sEI=better_result['x'][10]
sII=better_result['x'][11]
D=1
tE=better_result['x'][12]
tI=better_result['x'][13]
aDWEE=better_result['x'][14]
aDWIE=better_result['x'][15]
aDWEI=better_result['x'][16]
aDWII=better_result['x'][17]
bDWEE=better_result['x'][18]
bDWIE=better_result['x'][19]
bDWEI=better_result['x'][20]
bDWII=better_result['x'][21]

snE=0.0000001
#snE=0.0001
snI=snE

Ess = 0.00493218
Iss = 0.07516145

Time=60*10
Delta_t=0.0001

E_total = Graph_Wilson_Cowan_Model(Ess, Iss, Time, Delta_t,                          
                         aEE, aIE, aEI, aII,
                         sEE, sIE, sEI, sII, D,
                         dE, dI, P, Q, tE, tI, 
                                                                  
                                    aDWEE,aDWIE, aDWEI, aDWII,
                                    bDWEE, bDWIE, bDWEI, bDWII,  
                                                     
                        snE, snI, Graph_Kernel,                                  
                         one_dim=False, eigvals=eigenvalues, eigvecs=eigenvectors,
                         Visual=False, SaveActivity=True, Filepath='', checkpoint_timesteps=100000)  
