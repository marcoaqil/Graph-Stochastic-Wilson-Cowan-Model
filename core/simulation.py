import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import signal
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
from analysis import *
#written for python 3.6 
####################################################################################################
####################################################################################################    
####GRAPH SIMULATIONS
####################################################################################################
####################################################################################################    
####################################################################################################   
#testing any time-evolution propagator defined by a kernel on the graph
####################################################################################################   

def graph_propagator_test(u_0, Time, Delta_t, kernel_param, Graph_Kernel, sigma_noise=0,
                          one_dim=False, syn=0, gridsize=1000,  h=0.01, eigvals=None, eigvecs=None,                         
                          Visual=False, SaveActivity=False, Filepath=' ', NSim=0):
       
    if one_dim==True:
        s,U = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)

    else:
        s=eigvals
        U=eigvecs
    #note that s is a vector of eigenvalues, not the diagonal matrix of eigenvalues
    #s_matrix=sp.sparse.diags(s).toarray()
    if Graph_Kernel!='Damped Wave':
        kernel_matrix=sp.sparse.diags(GraphKernel(s,kernel_param, type=Graph_Kernel)).toarray()
        Laplacian_based_propagator = np.dot(U, np.dot(kernel_matrix, U.T))
    else:
        kernel_matrix, kernel_matrix_prime=GraphKernel(s,kernel_param, type=Graph_Kernel, prime=True)
        Laplacian_based_propagator = np.dot(U, np.dot(sp.sparse.diags(kernel_matrix).toarray(), U.T))
        Laplacian_based_propagator_prime = np.dot(U, np.dot(sp.sparse.diags(kernel_matrix_prime).toarray(), U.T))

        
        
    Timesteps = int(round(Time/Delta_t))
    u_Delta_t = np.zeros_like(u_0)
    u_prime = np.zeros_like(u_0)
    
    if SaveActivity==True:
        u_total = np.zeros((len(u_0),Timesteps-1000))   
    
    if Visual==True:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, len(u_0))
        ax.set_ylim(0, 11)
        #line2, = ax.plot(np.arange(len(I_0)), I_0, 'b-')
        #line1, = ax.plot(np.arange(len(E_0)), E_0, 'r-')        
        ax.plot(u_0, 'b-')     
        fig.canvas.draw()
    
    for i in range(Timesteps):
        
        if sigma_noise!=0:
            Noise = sigma_noise * np.array([gauss(0.0, 1.0) for k in range(len(u_0))])
        else:
            Noise = 0
            
        #impulse response    
        # if i==500:
        #     E_Delta_t[600:650]=0.9*np.ones(50)
        #     I_Delta_t[580:620]=0.9*np.ones(40)        
        if Graph_Kernel!='Damped Wave':
            u_Delta_t = np.dot(Laplacian_based_propagator,u_0)+np.sqrt(Delta_t)*Noise
        else:
            u_Delta_t = np.dot(Laplacian_based_propagator,u_0)+np.dot(Laplacian_based_propagator_prime,u_prime)+np.sqrt(Delta_t)*Noise
            u_prime=(u_Delta_t-u_0)/kernel_param
            
 
        if SaveActivity==True and i>=1000:
            u_total[:,i-1000]=np.copy(u_Delta_t)
            
        if Visual==True:# and i%10 == 0:
            print(i)
            ax.clear()
            ax.set_xlim(0, len(u_0))
            ax.set_ylim(-0.05,1.6)
            #line2.set_ydata(I_Delta_t)
            #line1.set_ydata(E_Delta_t)
            ax.plot(u_Delta_t, 'b-')           
            fig.canvas.draw()
            fig.canvas.flush_events()
           
        u_0 = np.copy(u_Delta_t)   
    
    if SaveActivity==True:
                    
        if Filepath==' ':
            filepath = 'G:/Macbook Stuff/Simulation Results/'+Graph_Kernel+' Kernel Test t=%.f/'%(kernel_param)
        else:
            filepath=Filepath
                        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
               
        with h5py.File(filepath+"%d# Sim Activity.h5"%(NSim)) as hf:
            if "Activity" not in list(hf.keys()):
                hf.create_dataset("Activity",  data=u_total)
            else:
                hf["Activity"]=u_total    
    
    return u_Delta_t

####################################################################################################
####################################################################################################
##################################################################
##GRAPH STOCHASTIC WILSON COWAN
##################################################################
#compute the four diffusion operators beforehand
def graph_WCM_propagators(alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1,
                          sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1,
                          Graph_Kernel='Gaussian', one_dim=False, syn=0, gridsize=1000, h=0.01,
                          eigvals=None, eigvecs=None):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D
    
    ForceParallel=True
                
    if one_dim==True:
        s,U = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)
        V=U.T
    else:
        s=eigvals
        U=eigvecs
        V=eigvecs.T
           
     
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
    #print(propagator_EE.shape)
    E_Delta_t = E_0 + time_E*(-d_e*E_0 + 1/(1+np.exp(-np.dot(propagator_EE,E_0) + np.dot(propagator_IE,I_0) - P)))+ Noise_E*np.sqrt(Delta_t)/tau_e 
    I_Delta_t = I_0 + time_I*(-d_i*I_0 + 1/(1+np.exp(-np.dot(propagator_EI,E_0) + np.dot(propagator_II,I_0) - Q)))+ Noise_I*np.sqrt(Delta_t)/tau_i 
    #print(E_Delta_t.shape)
    return E_Delta_t, I_Delta_t

#Wilson Cowan model 
def Graph_Wilson_Cowan_Model(Ess, Iss, Time, Delta_t,
                          alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1,
                          sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                          d_e=1, d_i=1, P=0, Q=0, tau_e=1, tau_i=1, sigma_noise_e=1, sigma_noise_i=1,
                          Graph_Kernel='Gaussian', one_dim=False, syn=0, gridsize=1000, h=0.01, eigvals=None, eigvecs=None,
                          Visual=False, SaveActivity=False, Filepath=' ', NSim=0):
    
    
    
    propagator_EE, propagator_IE, propagator_EI, propagator_II = graph_WCM_propagators(
                       alpha_EE, alpha_IE, alpha_EI, alpha_II,
                       sigma_EE, sigma_IE, sigma_EI, sigma_II, D,
                       Graph_Kernel, one_dim, syn, gridsize, h, eigvals,eigvecs)
    
    

        
    if one_dim==True:    
        E_0=Ess*np.ones(gridsize)
        I_0=Iss*np.ones(gridsize)
    else:
        E_0=Ess*np.ones(len(eigvals))
        I_0=Iss*np.ones(len(eigvals))
        
    
    E_Delta_t = np.zeros_like(E_0)
    I_Delta_t = np.zeros_like(I_0)
    
    Timesteps = int(round(Time/Delta_t))
    
    
    E_total = np.zeros((len(E_0),Timesteps-1000))
    
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
    
    
    for i in range(Timesteps):
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
         
        if i>=1000:
            E_total[:,i-1000]=np.copy(E_Delta_t)
            

        if Visual==True and i%10 == 0:
            print(i)
            ax.clear()
            ax.set_ylim(Ess-sigma_noise_e, Ess+sigma_noise_e)
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
                data=hf["Activity"]
                data[...]=E_total
    
                        
    return E_total

#################################################################################
#
# LINEARIZED MODEL
#
##################################################################################
def Linearized_GLDomain_Wilson_Cowan_Model(Ess, Iss, Time, Delta_t,
                          alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1,
                          sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                          d_e=1, d_i=1, P=0, Q=0, tau_e=1, tau_i=1, sigma_noise_e=1, sigma_noise_i=1,
                          Graph_Kernel='Gaussian', one_dim=False, syn=0, gridsize=1000, h=0.01, eigvals=None, eigvecs=None,
                          Visual=False, SaveActivity=False, Filepath=' ', NSim=0):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D   
    
    a = d_e*Ess*(1-d_e*Ess)
    b = d_i*Iss*(1-d_i*Iss)
    
   
    if one_dim==True:
        s,U = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)
    else:
        s=eigvals
        U=eigvecs
    
    beta_E_0 = np.zeros(len(s))
    beta_I_0 = np.zeros(len(s))
    
    beta_Ess = np.dot(U.T[0], Ess*np.ones(len(s)))
    beta_Iss = np.dot(U.T[0], Iss*np.ones(len(s)))
    
    beta_E_0[0] = beta_Ess
    beta_I_0[0] = beta_Iss    
            
    prop_EE = alpha_EE * GraphKernel(s, t_EE, Graph_Kernel)
    prop_IE = alpha_IE * GraphKernel(s, t_IE, Graph_Kernel)
    prop_EI = alpha_EI * GraphKernel(s, t_EI, Graph_Kernel)
    prop_II = alpha_II * GraphKernel(s, t_II, Graph_Kernel) 
    
    #beta_E_Delta_t = np.zeros_like(beta_E_0)
    #beta_I_Delta_t = np.zeros_like(beta_I_0)
    
    Timesteps = int(round(Time/Delta_t))    
    time_E = Delta_t/tau_e 
    time_I = Delta_t/tau_i 
    
    beta_E_total = np.zeros((len(beta_E_0),Timesteps-1000))    
    
    for i in range(Timesteps):
        if Visual==True and i%10 == 0:
            print(i)
            
        if sigma_noise_e!=0 or sigma_noise_i!=0:
            Noise_E = sigma_noise_e * np.array([gauss(0.0, 1.0) for k in range(len(beta_E_0))])
            Noise_I = sigma_noise_i * np.array([gauss(0.0, 1.0) for k in range(len(beta_I_0))])
        else:
            Noise_E = 0
            Noise_I = 0
               
        beta_E_Delta_t = beta_E_0 + time_E*((-d_e+a*prop_EE)*beta_E_0 - a*prop_IE*beta_I_0) + Noise_E*np.sqrt(Delta_t)/tau_e
        beta_I_Delta_t = beta_I_0 + time_I*(b*prop_EI*beta_E_0 - (d_i+b*prop_II)*beta_I_0) + Noise_I*np.sqrt(Delta_t)/tau_i
         
        if i>=1000:
            beta_E_total[:,i-1000]=np.copy(beta_E_Delta_t)
            
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
                data=hf["Beta_Activity"]
                data[...]=beta_E_total    

    return beta_E_total


#################################################################################
#
# activity analysis
#
##################################################################################   
def Activity_Analysis(Ess, Iss, Delta_t, beta=False, E_total=None, beta_E_total=None,
                      alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1,
                      sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                      d_e=1, d_i=1, P=0, Q=0, tau_e=1, tau_i=1, sigma_noise_e=1, sigma_noise_i=1,
                      Graph_Kernel='Gaussian', prediction=False, max_omega=100, delta_omega=0.1,
                      one_dim=True, syn=0, gridsize=1000, h=0.01, eigvals=None, eigvecs=None, Visual=True, Save_Results=False, Filepath=' ', NSim=0):
        
    if Save_Results==True:                    
        if Filepath==' ':
            filepath = 'G:/Macbook Stuff/Results/'+Graph_Kernel+' Kernel/aEE=%.3f aIE=%.3f aEI=%.3f aII=%.3f dE=%.3f dI=%.3f ' %(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i)
            filepath += 'P=%.3f Q=%.3f sEE=%.3f sIE=%.3f sEI=%.3f sII=%.3f D=%.3f tE=%.3f tI=%.3f/'%(P,Q,sigma_EE,sigma_IE,sigma_EI,sigma_II,D,tau_e,tau_i) 
        else:
            filepath=Filepath
                       
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
    if one_dim==True:
        s,U = one_dim_Laplacian_eigenvalues(gridsize, h, syn, vecs=True)
    else:
        s=eigvals 
        U=eigvecs       
    
    #analyze fluctuations about the steady state
    if beta==False:
        activity = E_total-Ess
        beta_activity = np.dot(U.T,activity)
    else:
        beta_E_total[0] = beta_E_total[0] - np.dot(U.T[0],Ess*np.ones(len(s)))
        beta_activity = beta_E_total
        activity = np.dot(U,beta_activity)
        
    PS = np.var(beta_activity, axis=1)
    print("Simulation SPS obtained.")
    TPS = signal.periodogram(beta_activity, fs=1/Delta_t, detrend='constant', scaling='density') 
    
    #TPS = np.abs(np.fft.fft(np.dot(U.T,E_total-Ess)))**2
    
    #full temporal spectrum
    FTPS = np.sum(TPS[1], axis=0)
    frequencies = TPS[0]*(2*np.pi)
    print("Simulation TPS obtained.")
    #FTPS = np.sum(TPS, axis=0)
  
    #time_step = 0.01
    #freqs = np.fft.fftfreq(FTPS.size, time_step)
    #idx = np.argsort(freqs)
    
    def hrf(t):
        "A hemodynamic response function"
        k=1
        return (k*t) ** 8.6 * np.exp(-(k*t) / 0.547)

    hrf_times = np.arange(0, 20, 0.1)
    hrf_signal=hrf(hrf_times)
    for ts in range(np.shape(activity)[1]):
        activity[:,ts]=np.convolve(activity[:,ts],hrf_signal,mode='same')
        
    covariance = np.cov(activity)
    FC=np.dot(np.diag(np.power(np.diag(covariance),-0.5)),np.dot(covariance,np.diag(np.power(np.diag(covariance),-0.5))))    
    print("All simulation activity measures completed.")
    if prediction==True:
         print("Obtaining analytic predictions...")
         PS_prediction = Graph_WC_Spatiotemporal_PowerSpectrum(s, Graph_Kernel, Ess, Iss,
                                                       alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                                       sigma_EE, sigma_IE, sigma_EI, sigma_II, D, 
                                                       tau_e, tau_i,
                                                       sigma_noise_e, sigma_noise_i, max_omega, delta_omega,
                                                       Spatial_Spectrum_Only=False, Visual=False)
         
         predicted_PS=delta_omega*np.sum(PS_prediction, axis=0)/np.pi
         predicted_TPS=2*np.sum(PS_prediction, axis=1)
         
         predicted_FC = Functional_Connectivity(U, predicted_PS, False, False)
         
     
    if Visual==True:
        if Save_Results==True:
            figpath1=filepath+"SPS E0=%.5f Sim #%d.png"%(Ess, NSim)
            figpath2=filepath+"TPS E0=%.5f Sim #%d.png"%(Ess, NSim)
            figpath3=filepath+"FC E0=%.5f Sim #%d.png"%(Ess, NSim)
             
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Spatial Eigenmode ($k$)")
        ax.set_title("Spatial Power Spectrum")
        #ax.set_xlim(-0.1, 20000)
        #ax.set_ylim(0, 20)
        
        line2, = plt.loglog(np.arange(1,len(s)+1), PS, '-r')     
        if prediction==True:
            line1, = plt.loglog(np.arange(1,len(s)+1), predicted_PS, '--k')
        if Save_Results==True:    
            plt.savefig(figpath1)
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.set_xlabel("Angular Frequency ($\omega$)")
        ax.set_title("Temporal Power Spectrum")
        line3, = plt.semilogx(frequencies, FTPS, '-r')       
        #line3, = plt.semilogx(freqs[idx], FTPS[idx])
        
        if prediction==True:
            line4, = plt.semilogx(np.arange(0,max_omega,delta_omega),predicted_TPS, '--k')
        if Save_Results==True:    
            plt.savefig(figpath2)
        
        if beta==False:
            if prediction==True:
                fig3 = plt.figure()
                plt.pcolormesh(predicted_FC)
                
            fig4 = plt.figure()
            plt.pcolormesh(FC)
            if Save_Results==True:    
                plt.savefig(figpath3)
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
        
        
    if Save_Results==True:                    
                
        with h5py.File(filepath+"PSD E0=%.5f Sim #%d.h5"%(Ess, NSim)) as hf:
            if "PSD" not in list(hf.keys()):
                hf.create_dataset("PSD",  data=PS)
            else:
                del hf["PSD"]
                hf.create_dataset("PSD",  data=PS)
                
        
    return PS, TPS, FC