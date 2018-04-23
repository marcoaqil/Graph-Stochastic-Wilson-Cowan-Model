import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import special
from random import gauss
#import hdf5storage
import h5py
import timeit
from numba import jit
#from sympy.solvers.solveset import nonlinsolve
#from sympy.core.symbol import symbols
#from sympy import exp
from scipy import stats
import os
from analysis import GraphKernel

####################################################################################################
####################################################################################################    
####GRAPH SIMULATIONS
####################################################################################################
####################################################################################################    
####################################################################################################   
#testing any time-evolution propagator defined by a kernel on the graph
####################################################################################################   

def graph_propagator_test(u_0, Graph_Kernel='Gaussian', t=1, 
                          one_dim=False, syn=0, gridsize=1000,  h=0.01,
                          eigvals=None, eigvecs=None):
       
    
    if one_dim==True:
        diagonals = [-np.ones(gridsize-1),2*np.ones(gridsize+1),-np.ones(gridsize-1)]
        Laplacian = sp.sparse.diags(diagonals,[-1,0,1]).toarray()
        
        #closed boundaries
        Laplacian[0,0]=1
        Laplacian[gridsize-1,gridsize-1]=1
        
        for p in range(syn):
            k1=int(np.floor(gridsize*np.random.rand()))
            k2=int(np.floor(gridsize*np.random.rand()))
            Laplacian[k1,k2]=-1
            Laplacian[k2,k1]=-1
            Laplacian[k1,k1]+=1
            Laplacian[k2,k2]+=1
            
        #periodic boundary
        #Laplacian[0,gridsize-1]=-1
        #Laplacian[gridsize-1,0]=-1
        Laplacian/=(h**2)
        s, U = np.linalg.eigh(Laplacian)
        U = sp.sparse.csc_matrix(U)
    else:
        s=eigvals
        U=eigvecs
    #note that s is a vector of eigenvalues, not the diagonal matrix of eigenvalues
    #s_matrix=sp.sparse.diags(s).toarray()
    kernel_matrix=sp.sparse.diags(GraphKernel(s,t, type=Graph_Kernel))
    Laplacian_based_propagator = np.dot(U, np.dot(kernel_matrix, U.T))
    
    u_t=np.dot(Laplacian_based_propagator,u_0)
    
    return u_t

####################################################################################################
####################################################################################################
##################################################################
##GRAPH STOCHASTIC WILSON COWAN
##################################################################
#compute the four diffusion operators beforehand
def graph_WCM_propagators(Graph_Kernel='Gaussian',
                       alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1,
                       sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1,
                       one_dim=False, syn=0, gridsize=1000, h=0.01,
                       eigvals=None, eigvecs=None):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D
    
    ForceParallel=True
    
    if one_dim==True:
        diagonals = [-np.ones(gridsize-1),2*np.ones(gridsize+1),-np.ones(gridsize-1)]
        Laplacian = sp.sparse.diags(diagonals,[-1,0,1]).toarray()
        
        #closed boundaries
        #Laplacian[0,0]=1
        #Laplacian[gridsize-1,gridsize-1]=1
        
        for p in range(syn):
            k1=int(np.floor(gridsize*np.random.rand()))
            k2=int(np.floor(gridsize*np.random.rand()))
            Laplacian[k1,k2]=-1
            Laplacian[k2,k1]=-1
            Laplacian[k1,k1]+=1
            Laplacian[k2,k2]+=1
            
        #periodic boundary
        Laplacian[0,gridsize-1]=-1
        Laplacian[gridsize-1,0]=-1
        Laplacian/=(h**2)
        s, U = np.linalg.eigh(Laplacian)
        V=U.T           
  
        
    else:
        s = eigvals
        U = eigvecs
        V = eigvecs.T
           
     
    if ForceParallel==True:   
        diag_prop_EE = alpha_EE * GraphKernel(s, t_EE, Graph_Kernel)
        diag_prop_IE = alpha_IE * GraphKernel(s, t_IE, Graph_Kernel)
        diag_prop_EI = alpha_EI * GraphKernel(s, t_EI, Graph_Kernel)
        diag_prop_II = alpha_II * GraphKernel(s, t_II, Graph_Kernel)  
         
        mask_EE = np.flatnonzero(diag_prop_EE)
        mask_IE = np.flatnonzero(diag_prop_IE)
        mask_EI = np.flatnonzero(diag_prop_EI)
        mask_II = np.flatnonzero(diag_prop_II)
        
        EE_skip = diag_prop_EE[mask_EE]
        IE_skip = diag_prop_IE[mask_IE]
        EI_skip = diag_prop_EI[mask_EI]
        II_skip = diag_prop_II[mask_II]
            
        prop_EEV = EE_skip[:,None] * V[mask_EE,:]
        prop_IEV = IE_skip[:,None] * V[mask_IE,:]
        prop_EIV = EI_skip[:,None] * V[mask_EI,:]
        prop_IIV = II_skip[:,None] * V[mask_II,:]
           
        propagator_EE = transpose_parallel_dot(V[mask_EE,:], prop_EEV)     #np.dot(U, np.dot(s_exp_matrix_EE,V))  
        propagator_IE = transpose_parallel_dot(V[mask_IE,:], prop_IEV)
        propagator_EI = transpose_parallel_dot(V[mask_EI,:], prop_EIV)
        propagator_II = transpose_parallel_dot(V[mask_II,:], prop_IIV)
    else:
        diag_prop_EE = sp.sparse.diags(alpha_EE * GraphKernel(s, t_EE, Graph_Kernel)).toarray()
        diag_prop_IE = sp.sparse.diags(alpha_IE * GraphKernel(s, t_IE, Graph_Kernel)).toarray()
        diag_prop_EI = sp.sparse.diags(alpha_EI * GraphKernel(s, t_EI, Graph_Kernel)).toarray()
        diag_prop_II = sp.sparse.diags(alpha_II * GraphKernel(s, t_II, Graph_Kernel)).toarray()
        
        propagator_EE = np.dot(U, np.dot(diag_prop_EE,V))  
        propagator_IE = np.dot(U, np.dot(diag_prop_IE,V))
        propagator_EI = np.dot(U, np.dot(diag_prop_EI,V))
        propagator_II = np.dot(U, np.dot(diag_prop_II,V))
        
    
    return propagator_EE, propagator_IE, propagator_EI, propagator_II
 
@jit(nopython=True, parallel=True)
def transpose_parallel_dot(A, B):  
    return np.dot(A.T, B)

@jit(nopython=True, parallel=True)    
def GWCM_Loop(E_0, I_0,  Delta_t,
           propagator_EE, propagator_IE, propagator_EI, propagator_II, 
           d_e, d_i, P, Q, tau_e, tau_i, Noise_E, Noise_I):
    
    time_E = Delta_t/tau_e 
    time_I = Delta_t/tau_i 
    print(propagator_EE.shape)
    E_Delta_t = E_0 + time_E*(-d_e*E_0 + 1/(1+np.exp(-np.dot(propagator_EE,E_0) + np.dot(propagator_IE,I_0) - P))+ Noise_E/np.sqrt(Delta_t)) 
    I_Delta_t = I_0 + time_I*(-d_i*I_0 + 1/(1+np.exp(-np.dot(propagator_EI,E_0) + np.dot(propagator_II,I_0) - Q))+ Noise_I/np.sqrt(Delta_t)) 
    print(E_Delta_t.shape)
    return E_Delta_t, I_Delta_t

#Wilson Cowan model 
def Graph_Wilson_Cowan_Model(E_0, I_0, Time, Delta_t,
                          propagator_EE, propagator_IE, propagator_EI, propagator_II, 
                          d_e=1, d_i=1, P=0, Q=0, tau_e=1, tau_i=1, sigma_noise_e=1, sigma_noise_i=1, Visual=False):
    
    if Visual==True:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, len(E_0))
        ax.set_ylim(0, 1)
        #line2, = ax.plot(np.arange(len(I_0)), I_0, 'b-')
        #line1, = ax.plot(np.arange(len(E_0)), E_0, 'r-')
        
        ax.plot(I_0, 'b-')
        ax.plot(E_0, 'r-')
        fig.canvas.draw()
    
    E_Delta_t = np.zeros_like(E_0)
    I_Delta_t = np.zeros_like(I_0)
    
    
    for i in range(int(round(Time/Delta_t))):
        print(i)
        if sigma_noise_e!=0 or sigma_noise_i!=0:
            Noise_E = sigma_noise_e * np.array([gauss(0.0, 1.0) for k in range(len(E_0))])
            Noise_I = sigma_noise_i * np.array([gauss(0.0, 1.0) for k in range(len(I_0))])
        else:
            Noise_E = 0
            Noise_I = 0
        #it turns out that the alphas ARE important, ie. at least manually, i cant get oscillations if i set them to one
        #simply manipulating the sigmas doesnt appear to be enough for oscillations. analysis or systematic numerics would solve this
        #impulse response    
        # if i==500:
        #     E_Delta_t[600:650]=0.9*np.ones(50)
        #     I_Delta_t[580:620]=0.9*np.ones(40)
        
        E_Delta_t, I_Delta_t = GWCM_Loop(E_0, I_0, Delta_t,
                           propagator_EE, propagator_IE, propagator_EI, propagator_II, 
                           d_e, d_i, P, Q, tau_e, tau_i, Noise_E, Noise_I)
            

        if Visual==True and i%10 == 0:
            
            ax.clear()
            #line2.set_ydata(I_Delta_t)
            #line1.set_ydata(E_Delta_t)
            ax.plot(I_Delta_t, 'b-')
            ax.plot(E_Delta_t, 'r-')
            fig.canvas.draw()
            fig.canvas.flush_events()
           
        E_0 = np.copy(E_Delta_t)   
        I_0 = np.copy(I_Delta_t)
        #print(E_0.shape)
        #print(str(E_0[10])+" "+str(I_0[20]))

            
            
    return E_Delta_t, I_Delta_t