import numpy as np
import scipy as sp
from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import signal

#import hdf5storage
import h5py
import timeit
import time
#from sympy.solvers.solveset import nonlinsolve
#from sympy.core.symbol import symbols
#from sympy import exp
from scipy import stats
import os
from .analysis import *
opj = os.path.join
#written for python 3.6 
####################################################################################################
####################################################################################################    
####GRAPH SIMULATIONS
####################################################################################################
####################################################################################################    
####################################################################################################   
#testing any time-evolution propagator defined by a kernel on the graph
####################################################################################################   

def graph_propagator_test(u_0, Time, Delta_t, kernel_param, Graph_Kernel, a=1, b=1, c=1, sigma_noise=0,
                          one_dim=False, syn=0, gridsize=1000,  h=0.01, GF_domain=False, eigvals=None, eigvecs=None,                         
                          Visual=False, SaveActivity=False, Filepath=' ', NSim=0):
       
    if one_dim==True:
        s,U = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)

    else:
        s=eigvals
        U=eigvecs
    #note that s is a vector of eigenvalues, not the diagonal matrix of eigenvalues
    #s_matrix=sp.sparse.diags(s).toarray()
    if Graph_Kernel!='Damped Wave':
        kernel_gf = GraphKernel(s,kernel_param, type=Graph_Kernel)
        if GF_domain == False:
            kernel_matrix=sp.sparse.diags(kernel_gf).toarray()
            Laplacian_based_propagator = np.dot(U, np.dot(kernel_matrix, U.T))
      
    else:
        kernel_gf, kernel_gf_prime=GraphKernel(s,kernel_param, type=Graph_Kernel, a=a, b=b, c=c, prime=True)
        
        if GF_domain == False:
            Laplacian_based_propagator = np.dot(U, np.dot(sp.sparse.diags(kernel_gf).toarray(), U.T))
            Laplacian_based_propagator_prime = np.dot(U, np.dot(sp.sparse.diags(kernel_gf_prime).toarray(), U.T))
            

        
        
    Timesteps = int(round(Time/Delta_t))
    u_Delta_t = np.zeros_like(u_0)
    u_prime = np.zeros_like(u_0)
    
    if SaveActivity==True or GF_domain == True:
        u_total = np.zeros((len(u_0),Timesteps))   
    
    if Visual==True and GF_domain == False:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, len(u_0))
        ax.set_ylim(0, 0.1)
        
        #line2, = ax.plot(np.arange(len(I_0)), I_0, 'b-')
        #line1, = ax.plot(np.arange(len(E_0)), E_0, 'r-')        
        ax.plot(u_0, 'b-')     
        fig.canvas.draw()
    
    for i in range(Timesteps):
        
        if sigma_noise!=0:
            Noise = sigma_noise * np.array([gauss(0.0, 1.0) for k in range(len(u_0))])
        else:
            Noise = 0
        
        
        if SaveActivity==True or GF_domain == True:
            u_total[:,i]=np.copy(u_0)
            
        if i%10 == 0:
            print(i)
        #impulse response    
        # if i==500:
        #     E_Delta_t[600:650]=0.9*np.ones(50)
        #     I_Delta_t[580:620]=0.9*np.ones(40)        
        if Graph_Kernel!='Damped Wave':
            if GF_domain == False:
                u_Delta_t = np.dot(Laplacian_based_propagator,u_0)+np.sqrt(Delta_t)*Noise
            else:
                u_Delta_t = kernel_gf * u_0 + np.sqrt(Delta_t)*Noise
        else:
            if GF_domain == False:
                u_Delta_t = np.dot(Laplacian_based_propagator,u_0)+np.dot(Laplacian_based_propagator_prime,u_prime)+np.sqrt(Delta_t)*Noise
                u_prime=(u_Delta_t-u_0)/kernel_param
            else:
                u_Delta_t = kernel_gf * u_0 + kernel_gf_prime * u_prime + np.sqrt(Delta_t)*Noise
                u_prime=(u_Delta_t-u_0)/kernel_param
            
 
            
        if Visual==True and i%5 == 0 and GF_domain == False:
            time.sleep(0.03)
            ax.clear()
            ax.set_xlim(0, len(u_0))
            ax.set_ylim(0,0.1)
            #line2.set_ydata(I_Delta_t)
            #line1.set_ydata(E_Delta_t)
            ax.plot(u_Delta_t, 'b-')           
            fig.canvas.draw()
            fig.canvas.flush_events()
           
        u_0 = np.copy(u_Delta_t)   
    
    if SaveActivity==True:
                    
        if Filepath==' ':
            if one_dim==True:
                filepath = 'G:/Macbook Stuff/Simulation Results/1D '+Graph_Kernel+' Kernel Test t=%.f/'%(kernel_param)
            else:
                filepath = 'G:/Macbook Stuff/Simulation Results/'+Graph_Kernel+' Kernel Test t=%.f/'%(kernel_param)
        
        else:
            filepath=Filepath
                        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
               
        with h5py.File(filepath+"%d# Sim Activity.h5"%(NSim)) as hf:
            if "Activity" not in list(hf.keys()):
                hf.create_dataset("Activity",  data=u_total)
            else:
                print("Warning: overwriting results of a previous simulation.")
                del hf["Activity"]
                hf.create_dataset("Activity",  data=u_total)    
    
    if GF_domain == False:
        return u_Delta_t
    else:
        return u_total

####################################################################################################
####################################################################################################
##################################################################
##GRAPH STOCHASTIC WILSON COWAN
##################################################################
#compute the four diffusion operators beforehand
def graph_WCM_propagators(t_EE=10, t_IE=10, t_EI=10, t_II=10, 
                        aDW_EE=1, aDW_IE=1, aDW_EI=1, aDW_II=1,
                       bDW_EE=1, bDW_IE=1, bDW_EI=1, bDW_II=1,
                          Graph_Kernel='Gaussian', one_dim=False, syn=0, gridsize=1000, h=0.01,
                          eigvals=None, eigvecs=None):
    

                
    if one_dim==True:
        s,U = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)
        V=U.T
    else:
        s=eigvals
        U=eigvecs
        V=eigvecs.T
           
    diag_prop_EE = sp.sparse.diags(GraphKernel(s,t_EE,type=Graph_Kernel,a=aDW_EE,b=bDW_EE)).toarray()
    diag_prop_IE = sp.sparse.diags(GraphKernel(s,t_IE,type=Graph_Kernel,a=aDW_IE,b=bDW_IE)).toarray()
    diag_prop_EI = sp.sparse.diags(GraphKernel(s,t_EI,type=Graph_Kernel,a=aDW_EI,b=bDW_EI)).toarray()
    diag_prop_II = sp.sparse.diags(GraphKernel(s,t_II,type=Graph_Kernel,a=aDW_II,b=bDW_II)).toarray()

    propagator_EE = np.dot(U, np.dot(diag_prop_EE,V))
    propagator_IE = np.dot(U, np.dot(diag_prop_IE,V))
    propagator_EI = np.dot(U, np.dot(diag_prop_EI,V))
    propagator_II = np.dot(U, np.dot(diag_prop_II,V))
    

    return propagator_EE, propagator_IE, propagator_EI, propagator_II
 
#@jit(nopython=True, parallel=True)
def transpose_parallel_dot(A, B):  
    return np.dot(A.T, B)

#@jit(nopython=True, parallel=True)    
def GWCM_Loop(E_0, I_0, Delta_t,
              alpha_EE, alpha_IE, alpha_EI, alpha_II,
           propagator_EE, propagator_IE, propagator_EI, propagator_II,
           d_e, d_i, P, Q, tau_e, tau_i, Noise_E, Noise_I):
    
    time_E = Delta_t/tau_e 
    time_I = Delta_t/tau_i 
    #print(I_0.dtype)

    E_Delta_t = E_0 + time_E*(-d_e*E_0 + 1/(1+np.exp(- alpha_EE * np.dot(propagator_EE,E_0) + alpha_IE * np.dot(propagator_IE,I_0)  - P)))+ Noise_E*np.sqrt(Delta_t)/tau_e 
    I_Delta_t = I_0 + time_I*(-d_i*I_0 + 1/(1+np.exp(- alpha_EI * np.dot(propagator_EI,E_0) + alpha_II * np.dot(propagator_II,I_0) - Q)))+ Noise_I*np.sqrt(Delta_t)/tau_i 
    #print(E_Delta_t.shape)
    return E_Delta_t, I_Delta_t


#Wilson Cowan model 
def Graph_Wilson_Cowan_Model(Ess, Iss, Time, Delta_t,
                          alpha_EE=100, alpha_IE=100, alpha_EI=100, alpha_II=100,
                          sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                          d_e=1, d_i=1, P=0, Q=0, tau_e=1, tau_i=1, 
                        aDW_EE=1, aDW_IE=1, aDW_EI=1, aDW_II=1,
                       bDW_EE=1, bDW_IE=1, bDW_EI=1, bDW_II=1,
                          sigma_noise_e=1, sigma_noise_i=1,
                          Graph_Kernel='Gaussian', one_dim=False, syn=0, gridsize=1000, h=0.01, eigvals=None, eigvecs=None,
                          Visual=False, SaveActivity=False, Filepath=' ', NSim=0):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D   

    print(f"{t_EE} {t_IE} {t_EI} {t_II}")
    
    propagator_EE, propagator_IE, propagator_EI, propagator_II = graph_WCM_propagators(
                       t_EE, t_IE, t_EI, t_II, 
                        aDW_EE, aDW_IE, aDW_EI, aDW_II,
                       bDW_EE, bDW_IE, bDW_EI, bDW_II,
                       Graph_Kernel, one_dim, syn, gridsize, h, eigvals,eigvecs)
    
    

        
    if one_dim==True:    
        E_0=Ess*np.ones(gridsize, dtype='float64')
        I_0=Iss*np.ones(gridsize, dtype='float64')
    else:
        E_0=Ess*np.ones(len(eigvals), dtype='float64')
        I_0=Iss*np.ones(len(eigvals), dtype='float64')
        
    
    E_Delta_t = np.zeros_like(E_0)
    I_Delta_t = np.zeros_like(I_0)
    
    Timesteps = int(round(Time/Delta_t))
    
    
    E_total = np.zeros((len(E_0),Timesteps-1000), dtype='float32')
    
    if Visual==True:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, len(E_0))
        #ax.set_ylim(0, 1)
        #line2, = ax.plot(np.arange(len(I_0)), I_0, 'b-')
        #line1, = ax.plot(np.arange(len(E_0)), E_0, 'r-')
        
        ax.plot(I_0, 'b-')
        ax.plot(E_0, 'r-')
        fig.canvas.draw()
    
    numerical_SS = True
    
    if numerical_SS == True:
        Ess_numerical=[]
        Iss_numerical=[]
    
    for i in range(Timesteps):
        if sigma_noise_e!=0 or sigma_noise_i!=0:
            Noise_E = sigma_noise_e * np.random.default_rng().normal(0, 1, size=len(E_0))
            Noise_I = sigma_noise_i * np.random.default_rng().normal(0, 1, size=len(I_0))
        else:
            Noise_E = 0
            Noise_I = 0

               
        E_Delta_t, I_Delta_t = GWCM_Loop(E_0, I_0,  Delta_t,
                            alpha_EE, alpha_IE, alpha_EI, alpha_II,
                           propagator_EE, propagator_IE, propagator_EI, propagator_II,
                           d_e, d_i, P, Q, tau_e, tau_i, Noise_E, Noise_I)
   
         
        if i>=1000:
            E_total[:,i-1000]=np.copy(E_Delta_t).astype('float32')
            if numerical_SS == True:
                Ess_numerical.append(np.mean(E_Delta_t))
                Iss_numerical.append(np.mean(I_Delta_t))
            

        if i%30 == 0:
            print(i)
            print(np.abs(E_Delta_t-Ess).max())
            print(np.abs(I_Delta_t-Iss).max())
            if Visual==True:
                ax.clear()
                ax.set_ylim(Ess-1e1*sigma_noise_e, Ess+1e1*sigma_noise_e)
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

    if SaveActivity==True:
                    
        if Filepath==' ':
            filepath = 'G:/Macbook Stuff/Results/'+Graph_Kernel+' Kernel/aEE=%.3f aIE=%.3f aEI=%.3f aII=%.3f dE=%.3f dI=%.3f ' %(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i)
            filepath += 'P=%.3f Q=%.3f sEE=%.3f sIE=%.3f sEI=%.3f sII=%.3f D=%.3f tE=%.3f tI=%.3f/'%(P,Q,sigma_EE,sigma_IE,sigma_EI,sigma_II,D,tau_e,tau_i) 
        else:
            filepath=Filepath
                        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        #make DAT files with sim-only parameters (delta t, time, etc)
        with h5py.File(filepath+"Activity E0=%.5f Sim #%d.h5"%(Ess, NSim)) as hf:
            if "Activity" not in list(hf.keys()):
                hf.create_dataset("Activity",  data=E_total)
            else:
                print("Warning: overwriting results of a previous simulation.")
                del hf["Activity"]
                hf.create_dataset("Activity",  data=E_total) 
                
    if numerical_SS==True:
        print(np.mean(np.array(Ess_numerical)))
        print(np.mean(np.array(Iss_numerical)))     
                
    return E_total

#################################################################################
#
# LINEARIZED MODEL
#
##################################################################################
def Linearized_GLDomain_Wilson_Cowan_Model(Ess, Iss, Time, Delta_t,
                          alpha_EE=100, alpha_IE=100, alpha_EI=100, alpha_II=100,
                          sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                          d_e=1, d_i=1, P=0, Q=0, tau_e=1, tau_i=1, 
                       aDW_EE=1, aDW_IE=1, aDW_EI=1, aDW_II=1,
                       bDW_EE=1, bDW_IE=1, bDW_EI=1, bDW_II=1,
                          sigma_noise_e=1, sigma_noise_i=1,
                          Graph_Kernel='Gaussian', one_dim=False, syn=0, gridsize=1000, h=0.01, eigvals=None, eigvecs=None,
                          Visual=False, SaveActivity=False, Filepath=' ', NSim=0):



    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D   
    
    ass = d_e*Ess*(1-d_e*Ess)
    bss = d_i*Iss*(1-d_i*Iss)

    print(f"{alpha_EE} {alpha_IE} {alpha_EI} {alpha_II}")
    
    #eigenvectors are used for plotting purposes only. 
    if one_dim==True:
        s, U = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)
    else:
        s=eigvals
        U=eigvecs
    
    #fluctuations about the steady state
    beta_E_0 = np.zeros(len(s), dtype='float64')
    beta_I_0 = np.zeros(len(s), dtype='float64')
     
    if Graph_Kernel == 'Damped Wave':
        prop_EE = alpha_EE * GraphKernel(s, t_EE, Graph_Kernel,a=aDW_EE,b=bDW_EE)
        prop_IE = alpha_IE * GraphKernel(s, t_IE, Graph_Kernel,a=aDW_IE,b=bDW_IE)
        prop_EI = alpha_EI * GraphKernel(s, t_EI, Graph_Kernel,a=aDW_EI,b=bDW_EI)
        prop_II = alpha_II * GraphKernel(s, t_II, Graph_Kernel,a=aDW_II,b=bDW_II)



    
    beta_E_Delta_t = np.zeros_like(beta_E_0)
    beta_I_Delta_t = np.zeros_like(beta_I_0)
    
    Timesteps = int(round(Time/Delta_t))    
    time_E = Delta_t/tau_e 
    time_I = Delta_t/tau_i 
    
    beta_E_total = np.zeros((len(beta_E_0),Timesteps-1000), dtype='float32')  
     
    
    
    if Visual==True:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, len(beta_E_0))
        #ax.set_ylim(0, 1)
        #line2, = ax.plot(np.arange(len(I_0)), I_0, 'b-')
        #line1, = ax.plot(np.arange(len(E_0)), E_0, 'r-')
        
        ax.plot(np.dot(U,beta_I_0), 'b-')
        ax.plot(np.dot(U,beta_E_0), 'r-')
        fig.canvas.draw()
    
    
       
    
    
    for i in range(Timesteps):


        if sigma_noise_e!=0 or sigma_noise_i!=0:
            Noise_E = sigma_noise_e * np.random.default_rng().normal(0, 1, size=len(beta_E_0))
            Noise_I = sigma_noise_i * np.random.default_rng().normal(0, 1, size=len(beta_I_0))
        else:
            Noise_E = 0
            Noise_I = 0
               
        beta_E_Delta_t = beta_E_0 + time_E*((-d_e+ass*prop_EE)*beta_E_0 - ass*prop_IE*beta_I_0) + Noise_E*np.sqrt(Delta_t)/tau_e
        beta_I_Delta_t = beta_I_0 + time_I*(bss*prop_EI*beta_E_0 - (d_i+bss*prop_II)*beta_I_0) + Noise_I*np.sqrt(Delta_t)/tau_i


         
        if i>=1000:
            beta_E_total[:,i-1000]=np.copy(beta_E_Delta_t).astype('float32')
            
            
        if i%100 == 0:
            print(i)
            print(np.abs(beta_E_0).max())   
            print(np.abs(beta_I_0).max())
            if Visual==True:
            
                ax.clear()
                ax.set_ylim(-1e1*sigma_noise_e, 1e1*sigma_noise_e)
    
                ax.plot(np.dot(U,beta_I_Delta_t), 'b-')
                ax.plot(np.dot(U,beta_E_Delta_t), 'r-')
                fig.canvas.draw()
                fig.canvas.flush_events()
            
        beta_E_0 = np.copy(beta_E_Delta_t)   
        beta_I_0 = np.copy(beta_I_Delta_t)    
        
    if SaveActivity==True:
                    
        if Filepath==' ':
            filepath = 'G:/Macbook Stuff/Results/'+Graph_Kernel+' Kernel/aEE=%.3f aIE=%.3f aEI=%.3f aII=%.3f dE=%.3f dI=%.3f ' %(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i)
            filepath += 'P=%.3f Q=%.3f sEE=%.3f sIE=%.3f sEI=%.3f sII=%.3f D=%.3f tE=%.3f tI=%.3f/'%(P,Q,sigma_EE,sigma_IE,sigma_EI,sigma_II,D,tau_e,tau_i) 
        else:
            filepath=Filepath
            
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
            #make DAT files with sim-only parameters (delta t, time, etc)
        with h5py.File(filepath+"Beta_Activity E0=%.5f Sim #%d.h5"%(Ess, NSim)) as hf:
            if "Beta_Activity" not in list(hf.keys()):
                hf.create_dataset("Beta_Activity",  data=beta_E_total)
            else:
                print("Warning: overwriting results of a previous simulation.")
                del hf["Beta_Activity"]
                hf.create_dataset("Beta_Activity",  data=beta_E_total)    

    return beta_E_total


#################################################################################
#
# activity analysis
#
##################################################################################   
def Activity_Analysis(Ess, Iss, Delta_t,
                      alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1,
                      sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                      d_e=1, d_i=1, P=0, Q=0, tau_e=1, tau_i=1, 
                       aDW_EE=1, aDW_IE=1, aDW_EI=1, aDW_II=1,
                       bDW_EE=1, bDW_IE=1, bDW_EI=1, bDW_II=1,
                      sigma_noise_e=1, sigma_noise_i=1,
                      Graph_Kernel='Gaussian', 
                      E_total=None, beta_E_total=None, compute_FC=False,
                      prediction=False, min_omega=0, max_omega=100, delta_omega=0.1, temporal_downsampling=1,
                      Spatial_scaling=[1,0], Temporal_scaling=[1,0],
                      one_dim=True, syn=0, gridsize=1000, h=0.01, eigvals=None, eigvecs=None, Visual=True, Save_Results=False, Filepath=' ', NSim=0):

    plt.rcParams.update({'font.size': 26})
    #plt.rcParams.update({'pdf.fonttype':42})
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    
    if Filepath != ' ' and not os.path.exists(Filepath):
        os.makedirs(Filepath)
        
    if Save_Results==True:                    
        if Filepath==' ':
            filepath = 'G:/Macbook Stuff/Results/'+Graph_Kernel+' Kernel/aEE=%.3f aIE=%.3f aEI=%.3f aII=%.3f dE=%.3f dI=%.3f ' %(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i)
            filepath += 'P=%.3f Q=%.3f sEE=%.3f sIE=%.3f sEI=%.3f sII=%.3f D=%.3f tE=%.3f tI=%.3f/'%(P,Q,sigma_EE,sigma_IE,sigma_EI,sigma_II,D,tau_e,tau_i) 
        else:
            filepath=Filepath
                       
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
    if one_dim==True:
        eigvals,eigvecs = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)
     
    
    #analyze fluctuations about the steady state
    if E_total is not None:
        E_total_fluctuations = E_total-Ess
        beta_E_total = np.dot(eigvecs.T,E_total_fluctuations)
    else:
        #needed only to calculate FC, which is only rarely done with 
        #whole connectome simulation (most common use case of beta, linearized sims)
        if compute_FC:
            E_total = np.dot(eigvecs,beta_E_total)
        
    PS = Spatial_scaling[0]*np.mean(np.abs(beta_E_total)**2, axis=1)+Spatial_scaling[1]
    print("Simulation SPS obtained.")
    
    TPS = signal.periodogram(beta_E_total[:,::temporal_downsampling], fs=1/(temporal_downsampling*Delta_t), detrend='constant', scaling='density') 
    
    
    #full temporal spectrum
    FTPS = Temporal_scaling[0]*np.sum(TPS[1], axis=0)+Temporal_scaling[1]
    frequencies = TPS[0]#*(2*np.pi)#for angular frequency
    print("Simulation TPS obtained.")

    HRF=False
    
    if HRF==True:
    
        def hrf(t):
            "A simple hemodynamic response function"
            k=1
            return (k*t) ** 8.6 * np.exp(-(k*t) / 0.547)
    
        hrf_times = np.arange(0, 20, 0.1)
        hrf_signal=hrf(hrf_times)
        for ts in range(np.shape(E_total_fluctuations)[1]):
            E_total_fluctuations[:,ts]=np.convolve(E_total[:,ts],hrf_signal,mode='same')
 
    if compute_FC==True:   
        #covariance = np.cov(E_total_fluctuations)
        #FC=np.dot(np.diag(np.power(np.diag(covariance),-0.5)),np.dot(covariance,np.diag(np.power(np.diag(covariance),-0.5))))
        FC = np.corrcoef(E_total_fluctuations)
    
    print("All simulation activity measures completed.")
    if prediction==True:
         print("Obtaining analytic predictions...")
         PS_prediction, PS_prediction_I = Graph_WC_Spatiotemporal_PowerSpectrum(eigvals, Graph_Kernel, Ess, Iss,
                                                       alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                                       sigma_EE, sigma_IE, sigma_EI, sigma_II, D, 
                                                       tau_e, tau_i,

                                                        aDW_EE, aDW_IE, aDW_EI, aDW_II,
                                                        bDW_EE, bDW_IE, bDW_EI, bDW_II,    
                                                       sigma_noise_e, sigma_noise_i, min_omega, max_omega, delta_omega,
                                                       Spatial_Spectrum_Only=False, Visual=False)
         
         PS_prediction_spatial = Graph_WC_Spatiotemporal_PowerSpectrum(eigvals, Graph_Kernel, Ess, Iss,
                                                       alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                                       sigma_EE, sigma_IE, sigma_EI, sigma_II, D, 
                                                       tau_e, tau_i,
                                                 aDW_EE, aDW_IE, aDW_EI, aDW_II,
                                                 bDW_EE, bDW_IE, bDW_EI, bDW_II, 
                                                       sigma_noise_e, sigma_noise_i,
                                                       Spatial_Spectrum_Only=True, Visual=False)         
         
         predicted_PS=Spatial_scaling[0]*PS_prediction_spatial[:,0,0]+Spatial_scaling[1]#delta_omega*np.sum(PS_prediction, axis=0)/np.pi
         predicted_TPS=Temporal_scaling[0]*2*np.sum(PS_prediction, axis=1)+Temporal_scaling[1]
         
         if compute_FC==True:
             predicted_FC = Functional_Connectivity(eigvecs, predicted_PS, False)
         
     
    if Visual==True:
        if Save_Results==True:
            figpath1=opj(filepath, "SPS E0=%.5f Sim #%d.pdf"%(Ess, NSim))
            figpath2=opj(filepath, "TPS E0=%.5f Sim #%d.pdf"%(Ess, NSim))
            figpath3=opj(filepath, "Predicted_FC E0=%.5f Sim #%d.pdf"%(Ess, NSim))
            figpath4=opj(filepath, "Simulated_FC E0=%.5f Sim #%d.pdf"%(Ess, NSim))
             
        plt.ion()
        plt.figure(figsize=(8,8))
        plt.xlabel("Harmonic Eigenmode ($k$)")
        plt.title("Harmonic Power Spectrum")
        #ax.set_xlim(-0.1, 20000)
        #plt.ylim(1e-16, 1e-13)
        plt.xlim(0.8,len(eigvals)+1)
        
        line2, = plt.loglog(np.arange(1,len(eigvals)), PS[1:], '-r', label='Simulation')     
        if prediction==True:
            line1, = plt.loglog(np.arange(1,len(eigvals)), predicted_PS[1:], '--k', label='Prediction')
        plt.legend()
        if Save_Results==True:    
            plt.savefig(figpath1, dpi=600, bbox_inches='tight')
        
        plt.figure(figsize=(8,8))
        plt.xlabel("Temporal Frequency (Hz)")
        plt.title("Temporal Power Spectrum")
        line3, = plt.loglog(frequencies, FTPS, '-r', label='Simulation')#'rs', label='Simulation', mec='black')        
        #plt.ylim(8*1e-16,3*1e-13)
        plt.xlim(0.5,100)
        
        if prediction:
            line4, = plt.loglog(np.arange(min_omega,max_omega,delta_omega)/(2*np.pi),predicted_TPS, '--k', label='Prediction')
        plt.legend()
        if Save_Results==True:    
            plt.savefig(figpath2, dpi=600, bbox_inches='tight')
        
        if compute_FC:
            if prediction:
                fig3 = plt.figure(figsize=(8,8))
                ax = fig3.add_subplot(111)
                ax.set_title("FC Matrix (Prediction)", pad=15)
                plot_pred_FC = ax.imshow(predicted_FC, cmap='inferno')
                fig3.colorbar(plot_pred_FC)

                if Save_Results==True:    
                    plt.savefig(figpath3, dpi=600, bbox_inches='tight')               
                
            fig4 = plt.figure(figsize=(8,8))
            ax2 = fig4.add_subplot(111)
            ax2.set_title("FC Matrix (Simulation)", pad=15)
            plot_actual_FC = ax2.imshow(FC, cmap='inferno')
            fig4.colorbar(plot_actual_FC)
            
            if Save_Results==True:    
                plt.savefig(figpath4, dpi=600, bbox_inches='tight')
#        else:
#            from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
#            import plotly.graph_objs as go
#            init_notebook_mode()    
#            if prediction==True:
#                trace0 = go.Heatmap(z=predicted_FC)           
#                data = [trace0]            
#                fig = dict(data=data)
#                plot(fig, filename='predicted_FC.html')
#            
#            trace1 = go.Heatmap(z=FC)           
#            data = [trace1]            
#            figz = dict(data=data)
#            plot(figz, filename='FC.html')
        
        
    # if Save_Results==True:                    
                
    #     with h5py.File(filepath+"PSD E0=%.5f Sim #%d.h5"%(Ess, NSim)) as hf:
    #         if "PSD" not in list(hf.keys()):
    #             hf.create_dataset("PSD",  data=PS)
    #         else:
    #             del hf["PSD"]
    #             hf.create_dataset("PSD",  data=PS)
                
    if compute_FC:
        return PS, TPS, FC
    else:    
        return PS, TPS