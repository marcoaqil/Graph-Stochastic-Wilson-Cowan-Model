import numpy as np
import scipy as sp
from scipy import special
import similaritymeasures as sm
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
from scipy import sparse
plt.rcParams.update({'font.size': 20})
plt.tight_layout()
####################################################################################################
####################################################################################################
#written for python 3.6 

def GraphKernel(x,t,type='Gaussian', a=10**3, b=10, c=0, prime=False):
    if t<0:
        #print("Need positive kernel parameter")
        return
    else:
        if type=='Gaussian':
            #add prefactor to make kernel of unitary height
            return np.exp(t*x)#*2*np.sqrt(t*np.pi)
        elif type=='Exponential':
            return t/(t-x) #*2*t
        elif type=='Pyramid':
            #return t*(np.sinc(t*np.sqrt(-x)/(2*np.pi)))**2
            return np.sinc(np.sqrt(-x)/(2*np.pi*t))**2
        elif type=='Rectangle':   
            #Note: significant Gibbs effect makes this not advisable
            rect=np.sinc(np.sqrt(-x)/(2*np.pi*t))
            #rect[-990:]=0
            return rect
        elif type=='Mexican Hat':
            return -x*np.exp(t*x) #*2*np.sqrt(t*np.pi)  #*2*t  
        elif type=='Damped Wave':
            r_1=(-b+sp.sqrt(b**2 + 4*a*(x-c)))/(2*a)
            r_2=(-b-sp.sqrt(b**2 + 4*a*(x-c)))/(2*a)
            Damped_Wave_Kernel=(r_1*sp.exp(r_2*t)-r_2*sp.exp(r_1*t))/(r_1-r_2)
            Damped_Wave_Kernel_prime=(sp.exp(r_1*t)-sp.exp(r_2*t))/(r_1-r_2)
            if np.any(Damped_Wave_Kernel.imag!=0) or np.any(Damped_Wave_Kernel_prime.imag!=0):
                #print("Imaginary value in kernel")
                return Damped_Wave_Kernel.real
            else:
                if prime==True:
                    return Damped_Wave_Kernel.real, Damped_Wave_Kernel_prime.real
                else:
                    return Damped_Wave_Kernel.real


####################################################################################################
####################################################################################################
def one_dim_Laplacian_eigenvalues(gridsize, h, syn=0, vecs=False):
     
    diagonals = [np.ones(gridsize-1),np.zeros(gridsize+1),np.ones(gridsize-1)]
    #1D grid-graph
    AdjMatrix = sp.sparse.diags(diagonals,[-1,0,1]).toarray() 
    #periodic boundary
    #AdjMatrix[0,gridsize-1]=1
    #AdjMatrix[gridsize-1,0]=1
    
    AdjMatrix/=(h**2)
    
    #"synapses" (nonlocal connections) 
    if syn!=0:

        #simulatig hemispheric separation
        #AdjMatrix[499,500]=0
        #AdjMatrix[500,499]=0
        
        speed_factor=200
        indices1=np.arange(250,250+syn)#(gridsize*np.random.rand(syn)).astype(int)#
        indices2=np.arange(750,750+syn)#(gridsize*np.random.rand(syn)).astype(int)#
        
        #all-to-all or one-by-one connections
        alltoall=False
        
        for count, k1 in enumerate(indices1):
            
            if alltoall==True:
                for k2 in indices2:
                    if k1!=k2:                
                        dist=h*np.abs(k1-k2)                
                        AdjMatrix[k1,k2]=(speed_factor/dist)**2
                        AdjMatrix[k2,k1]=(speed_factor/dist)**2
                        
            else:

                dist=h*np.abs(k1-indices2[count])                
                AdjMatrix[k1,indices2[count]]=(speed_factor/dist)**2
                AdjMatrix[indices2[count],k1]=(speed_factor/dist)**2    
                
                #cross 
                dist_2=h*np.abs(k1-indices2[-count])                         
                AdjMatrix[k1,indices2[-count]]=(speed_factor/dist_2)**2
                AdjMatrix[indices2[-count],k1]=(speed_factor/dist_2)**2   
                
    
    #PLOT ADJ MATRIX
    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    ad_plot=ax.pcolormesh(AdjMatrix)
    fig3.colorbar(ad_plot)
    
    Deg=np.sum(AdjMatrix, axis=0)
    #sqrt_Deg=np.power(Deg,-0.5)
    Degree_Matrix=sp.sparse.diags(Deg)
    #sqrt_Degree_Matrix=sp.sparse.diags(sqrt_Deg)
    regLap = Degree_Matrix - sp.sparse.csc_matrix(AdjMatrix)
    #Laplacian = -(sp.sparse.csc_matrix.dot(sqrt_Degree_Matrix,sp.sparse.csc_matrix.dot(regLap,sqrt_Degree_Matrix))).toarray()
    #Laplacian[Laplacian>1]=1
    
    #unnormalized laplacian
    Laplacian=-regLap.toarray()
    
    #fig = plt.figure()
    #plt.pcolormesh(Laplacian)
    
    #Laplacian/=(h**2)
    
    if vecs==False:    
        s=np.linalg.eigvalsh(Laplacian)
        s[-1]=0
        #s[-1]=-np.abs(s[-1])

        return s[::-1]
    else:
        s,U=np.linalg.eigh(Laplacian)
        #NOTE: manually setting the first eigenvalue to zero seems to improve SPS numerical-analytic agreement at first eigenmode
        #(actually there is no difference as long as first eigenvalue is negative as it should be. Sometimes it comes out as a small positive value)
        s[-1]=0
        #this also works
        #s[-1]=-np.abs(s[-1])
   
        #for i in range(len(s)):
        #    U[:,i] = U[:,i]/np.sum(np.abs(U[:,i]))
        
        #NOTE: by contrast, manually setting the first eigenvector to zero seems to lead to errors in (nonlinear) numerical simulations
        #do not do this
        #U[:,-1]=np.zeros(len(s))
        #note that the vectors come out 2-normalized. seems to be fine as it is
        return s[::-1], U[:,::-1]
        
####################################################################################################
####################################################################################################
####################################################################################################
#Implementing a solver for the homogeneous steady state, in the simple case with space-independent parameters
#Implementation with multiple initial condition to find all steady states numerically
#thresholding unique steady state if norm(x1-x2)<0.01    
def H_Simple_Steady_State(alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1, d_e=1, d_i=1, P=0, Q=0):
    #generate multiple initial conditions to find all steady states
    initial_guesses = 5
    ##print("%.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g"%(alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i, P, Q))

    x0 = np.zeros((2,initial_guesses))
    x0[:,1] = np.array([1/(2*d_e), 1/(2*d_i)])
    x0[:,2] = np.random.rand(2)
    x0[:,3] = np.random.rand(2)
    x0[:,4] = np.random.rand(2)
    
    success = False
    results = np.zeros((2,initial_guesses))
    
    def f(x, alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i, P, Q):       
        d = np.array([[d_e,0],[0,d_i]], dtype=float)
        alpha = np.array([[alpha_EE,-alpha_IE],[alpha_EI,-alpha_II]], dtype=float)
        X = np.array([P,Q], dtype=float)
        
        SS_EQ = - np.dot(d,x) + sp.special.expit(np.dot(alpha,x) + X)
        return SS_EQ
    
    for i in range(initial_guesses):
        steady_state = sp.optimize.fsolve(f,x0[:,i],args=(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i,P,Q),
                                          #xtol=1e-9, 
                                          full_output=True) 
        
     #important line: conditions for success of SS calculation
        if steady_state[0][0]>=0 and steady_state[0][1]>=0: # and np.linalg.norm(steady_state[1]['fvec'],ord=1)<=1e-20:
            results[0,i]=steady_state[0][0]
            results[1,i]=steady_state[0][1]
            success=True
        else:
            results[0,i]= np.nan
            results[1,i]= np.nan
    
    #select and importantly sort the unique, acceptable results
    if success==True:

        results = results[:,~np.all(np.isnan(results), axis=0)]    
        uniques = np.unique(results, axis=1)
        
        #Further routine to select unique steady states up to some user-specified numerical tolerance   
        tolerance=0.001
        p=0
        countSS=1
        countoccurr=1
        finals=np.copy(uniques)
        
        while p < len(uniques[0])-1:
            diff = np.linalg.norm(uniques[:,p]-uniques[:,p+1], ord=2)
            if diff < tolerance:
                finals[:,p]+=uniques[:,p+1]
                uniques=np.delete(uniques, p+1, axis=1)
                finals=np.delete(finals,p+1,axis=1)
                countoccurr+=1
            else:
                finals[:,p]/=countoccurr
                countoccurr=1
                countSS+=1
                p+=1
                    
        finals[:,p]/=countoccurr        
        
    #    #print(str(len(finals[0]))+" unique steady states were found")        
       
        return finals, success
    else:
  #      #print("No positive, exact solutions were found")
        return None, success

####################################################################################################
####################################################################################################
#Implementing the explicit calculation of linearised Jacobian trace and determinant 
#for the space-independent homogeneous case
#given only the Wilson-Cowan model parameters and Laplacian eigenspectrum
def GraphWC_Jacobian_TrDet(Laplacian_eigenvalues, Graph_Kernel='Gaussian', Ess=None, Iss=None, 
                       alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1, d_e=1, d_i=1, 
                       sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                       tau_e=1, tau_i=1, 
                       aDW_EE=1, aDW_IE=1, aDW_EI=1, aDW_II=1,
                       bDW_EE=1, bDW_IE=1, bDW_EI=1, bDW_II=1,
                       Visual=False):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D
        
    eigs=Laplacian_eigenvalues
    
    Jacobian_eigenvalues=np.zeros((len(eigs),2),dtype=complex)
    SStype=0
    suitable=False

    
    if Ess == None or Iss == None:
        ##print("Steady state solution not provided. Using simplest approximation.")
        Ess = 1/(2*d_e)
        Iss = 1/(2*d_i)
    
    a = d_e*Ess*(1-d_e*Ess)
    b = d_i*Iss*(1-d_i*Iss)

    K_EE = GraphKernel(eigs,t_EE,type=Graph_Kernel,a=aDW_EE,b=bDW_EE)
    K_IE = GraphKernel(eigs,t_IE,type=Graph_Kernel,a=aDW_IE,b=bDW_IE)
    K_EI = GraphKernel(eigs,t_EI,type=Graph_Kernel,a=aDW_EI,b=bDW_EI)
    K_II = GraphKernel(eigs,t_II,type=Graph_Kernel,a=aDW_II,b=bDW_II)
    
    
    Trace = -(d_e/tau_e+d_i/tau_i) + alpha_EE*a*K_EE/tau_e - alpha_II*b*K_II/tau_i
    
    Determinant = -alpha_EE*alpha_II*a*b*K_EE*K_II/(tau_e*tau_i) - alpha_EE*a*d_i*K_EE/(tau_e*tau_i) + alpha_II*b*d_e*K_II/(tau_e*tau_i) + alpha_IE*alpha_EI*a*b*K_EI*K_IE + d_e*d_i/(tau_e*tau_i)
    
    
    Jacobian_eigenvalues[:,0]= (Trace + sp.sqrt(Trace**2 - 4*Determinant))/2.0
    Jacobian_eigenvalues[:,1]= (Trace - sp.sqrt(Trace**2 - 4*Determinant))/2.0
    
    if Visual==True:
        plt.ion()
        fig = plt.figure()
        plt.title("Jacobian Eigenspectrum")
        plt.xlabel("Re[x]")
        plt.ylabel("Im[x]")
        color=np.repeat(np.linspace(0,1,len(eigs)),2)[::-1]
        #ax = fig.add_subplot(111)
        #ax.set_xlim(-0.1, 20000)
        #ax.set_ylim(0, 20)
        plt.scatter(np.ravel(Jacobian_eigenvalues).real,np.ravel(Jacobian_eigenvalues).imag, marker='o', s=2, c=color, cmap='nipy_spectral')#, edgecolor='black', linewidth=0.1)
    
    if np.any(Jacobian_eigenvalues.real>=0) or np.any(np.isnan(Jacobian_eigenvalues)):
        ##print("E*=%.4f, I*=%.4f: unstable"%(Ess,Iss))
        SStype=0
        suitable = True
    else:
                  
        if np.all(Jacobian_eigenvalues.real<0) and np.all(Jacobian_eigenvalues.imag==0):
            ##print("E*=%.4f, I*=%.4f: strictly stable"%(Ess,Iss))
            SStype=1
            suitable = True
      #all or any in the line below for imaginary? ask rikkert #do they all need imaginary parts?
        elif np.all(Jacobian_eigenvalues.real<0) and np.any(Jacobian_eigenvalues.imag != 0):
            ##print("E*=%.4f, I*=%.4f: stable, with nonzero imaginary components"%(Ess,Iss))
            SStype=2
            suitable = True
                #same question here. what if some imaginary are zero and some nonzero?
        elif np.all(Jacobian_eigenvalues.real==0) and np.all(Jacobian_eigenvalues.imag != 0):
            ##print("E*=%.4f, I*=%.4f: all purely imaginary eigenvalues (potential Hopf)"%(Ess,Iss))
            SStype=3
            suitable = True
          
        else:
 
#cases with BOTH zero AND (negative) nonzero real components of eigens. 
#first subcase: all imaginary parts are zero (stable/undetermined?)
#second subcase: some imaginary parts are nonzero (need to consider overlap? does it matter if there are eigenvalues with magnitude zero?)
#third subcase: all imaginary parts are nonzero (same question)
            #print("E*=%.4f, I*=%.4f: undetermined"%(Ess,Iss))
            SStype=4
            suitable = True
            
            
        
    return SStype, suitable, Jacobian_eigenvalues #Trace, Determinant, Jacobian_eigenvalues


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#This method calculates and returns (if needed) the 2x2 linearised graph WC Jacobian of each eigenmode separately;
#and subsequently uses Ornstein-Uhlenbeck statistics for stochastic differential equations
#to calculate the G matrix for each mode, containing Power Spectral Density of that specific mode, and other quantities
def Graph_WC_Spatiotemporal_PowerSpectrum(Laplacian_eigenvalues, Graph_Kernel='Gaussian', Ess=None, Iss=None,  
                       alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1, d_e=1, d_i=1,
                       sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1,
                       tau_e=1, tau_i=1,
                       aDW_EE=1, aDW_IE=1, aDW_EI=1, aDW_II=1,
                       bDW_EE=1, bDW_IE=1, bDW_EI=1, bDW_II=1,
                       sigma_noise_e=1, sigma_noise_i=1, min_omega=0, max_omega=100, delta_omega=0.1,
                       Spatial_Spectrum_Only=True, Visual=False):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D
    eigs=Laplacian_eigenvalues

    K_EE = GraphKernel(eigs,t_EE,type=Graph_Kernel,a=aDW_EE,b=bDW_EE)
    K_IE = GraphKernel(eigs,t_IE,type=Graph_Kernel,a=aDW_IE,b=bDW_IE)
    K_EI = GraphKernel(eigs,t_EI,type=Graph_Kernel,a=aDW_EI,b=bDW_EI)
    K_II = GraphKernel(eigs,t_II,type=Graph_Kernel,a=aDW_II,b=bDW_II)
   
    
    if Ess == None or Iss == None:
        #print("Steady state solution not provided. Using simplest approximation.")
        Ess = 1/(2*d_e)
        Iss = 1/(2*d_i)
    
    a = d_e*Ess*(1-d_e*Ess)
    b = d_i*Iss*(1-d_i*Iss)
    
    Dmatrix=np.array([[sigma_noise_e/tau_e,0],[0,sigma_noise_i/tau_i]])**2
       
    A = np.stack([[d_e/tau_e - a*alpha_EE*K_EE/tau_e, a*alpha_IE*K_IE/tau_e],[-b*alpha_EI*K_EI/tau_i, d_i/tau_i + b*alpha_II*K_II/tau_i ]])
    A = np.moveaxis(A,-1,0)
        
    if Spatial_Spectrum_Only==True:
        
        Gmatrix2 = np.zeros((len(eigs),2,2), dtype=float)
  
        
        Gmatrix2[:,0,0] = 0.5*((sigma_noise_e**2)/(tau_i*d_e-tau_i*a*alpha_EE*K_EE+d_i*tau_e+b*tau_e*alpha_II*K_II))*((tau_i/tau_e) + ((a**2)*((alpha_IE*K_IE)**2)+ (d_i + b*alpha_II*K_II)**2)/(d_e*d_i+ (d_e*b*alpha_II*K_II) - (a*d_i*alpha_EE*K_EE) - a*b*(alpha_EE*K_EE*alpha_II*K_II-alpha_EI*K_EI*alpha_IE*K_IE)))
        
        #i cannot currently be bothered to write this down in terms of the parameters explicitly
        Gmatrix2[:,1,1] = 0.5*(Dmatrix[1,1] + (Dmatrix[0,0]*A[:,1,0]**2+Dmatrix[1,1]*A[:,0,0]**2) / (A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]))/(A[:,0,0]+A[:,1,1])
        
        if Visual==True:
            plt.ion()
            plt.figure()
            plt.xlabel("Harmonic Eigenmode ($k$)")
            plt.title("Harmonic Power Spectrum $H_E(k)$")
            #ax = fig.add_subplot(111)
            #ax.set_xlim(-0.1, 20000)
            #ax.set_ylim(0, 20)
            #line2, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,1,1], 'b-')
            #line1, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,0,0], 'r-')
            line3, = plt.loglog(np.arange(1,len(eigs)),np.abs(Gmatrix2[1:,0,0]), '--k')   
            
        return np.abs(Gmatrix2)                 
    else:
        omegas=np.arange(min_omega ,max_omega,delta_omega)

        E_Full_Spectrum=np.zeros((len(eigs),len(omegas)), dtype=float)
        I_Full_Spectrum=np.zeros((len(eigs),len(omegas)), dtype=float)
           
        for i, omega in enumerate(omegas):
            E_Full_Spectrum[:,i] = (Dmatrix[0,0]*(A[:,1,1]**2+omega**2) + Dmatrix[1,1]*A[:,0,1]**2)/((A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]-omega**2)**2 + ((A[:,0,0]+A[:,1,1])*omega)**2)
            I_Full_Spectrum[:,i] = (Dmatrix[1,1]*(A[:,0,0]**2+omega**2) + Dmatrix[0,0]*A[:,1,0]**2)/((A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]-omega**2)**2 + ((A[:,0,0]+A[:,1,1])*omega)**2)

            
        if Visual==True:
            
            # fig = plt.figure()
            # plt.ion()
            #ax.set_xlim(-0.1, 20000)
            #ax.set_ylim(omega_range[1], max_omega)
            #########use the /2pi rescaling if want temporal frequency
            #ax.set_ylim(omega_range[1]/(2*np.pi), max_omega/(2*np.pi))
            #line2, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,1,1], 'b-')
            #line1, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,0,0], 'r-')
            

            surf_plot=False
            if surf_plot==True:
                fig = plt.figure()
                plt.ion()
                ax = fig.add_subplot(121,projection='3d')
                              
                X, Y = np.meshgrid(np.arange(1,len(eigs)+1),omegas/(2*np.pi))
                ax.plot_surface(X,Y,E_Full_Spectrum.T)#,norm=pltcolors.LogNorm())
                ax_2 = fig.add_subplot(122,projection='3d')
                ax_2.plot_surface(X,Y,I_Full_Spectrum.T)

            else:
                plt.figure()                

                plt.xscale('log')            
                plt.yscale('log')

                plt.xlabel("Harmonic Eigenmode ($k$)")
                plt.ylabel("Temporal Frequency (Hz)")           
                plt.title("Excitatory Harmonic-Temporal Power Spectrum", pad=15) 
                plt.xlim(1, len(eigs))
                plt.minorticks_off()

                pc=plt.pcolormesh(np.arange(1,len(eigs)),omegas/(2*np.pi),E_Full_Spectrum.T[:,1:],norm=pltcolors.LogNorm(), cmap='jet')
                
                plt.yticks(ticks=[1,10,20,30,40], labels=['1','10','20','30','40'])

                plt.colorbar(pc)

                plt.figure()             

                plt.xscale('log')            
                plt.yscale('log')

                plt.xlabel("Harmonic Eigenmode ($k$)")
                plt.ylabel("Temporal Frequency (Hz)")           
                plt.title("Inhibitory Harmonic-temporal Power Spectrum", pad=15) 
                plt.xlim(1, len(eigs))
                plt.minorticks_off()

                pc_2=plt.pcolormesh(np.arange(1,len(eigs)),omegas/(2*np.pi),I_Full_Spectrum.T[:,1:],norm=pltcolors.LogNorm(), cmap='jet')
                
                plt.yticks(ticks=[1,10,20,30,40], labels=['1','10','20','30','40'])

                plt.colorbar(pc_2)        
                
                plt.figure()
                plt.xlabel("Temporal Frequency (Hz)")
                plt.title("Temporal Power Spectrum")
                plt.loglog(omegas/(2*np.pi), 2*np.sum(E_Full_Spectrum.T, axis=1),'--k')
            
            
        return E_Full_Spectrum.T, I_Full_Spectrum.T
   

    
   


def Functional_Connectivity(eigvecs, PS, Visual=False):
    U=eigvecs
    covariance = np.dot(U,np.dot(np.diag(PS),U.T))
    FC=np.dot(np.diag(np.power(np.diag(covariance),-0.5)),np.dot(covariance,np.diag(np.power(np.diag(covariance),-0.5))))
    if Visual==True:
         
        fig3 = plt.figure()
        ax = fig3.add_subplot(111)
        ax.set_title("Functional Connectivity (CHAOSS prediction)", pad=15)
        fc_plot=ax.imshow(FC, vmin=0.0, vmax=0.2, cmap='inferno')
        fig3.colorbar(fc_plot)
        
    
#        else:
#            from plotly.offline import init_notebook_mode,  plot
#            import plotly.graph_objs as go
#            init_notebook_mode()    
#        
#            trace1 = go.Heatmap(z=FC)           
#            data = [trace1]            
#            figz = dict(data=data)
#            plot(figz, filename='FC.html')
        
    
    return FC
####################################################################################################
####################################################################################################    
def NF_to_empirical(x, e_s, i_s):
    
    #c_s = e_s
    #c_s =  x[0]*e_s +x[1]
    c_s =  x[0]*e_s+x[1]*i_s+x[2]
    #c_s =  x[0]*(e_s*i_s)+x[1]
    #c_s =  x[0]*(e_s+i_s)+x[1]*(e_s*i_s)+x[2]
    
    #c_s = x[0]*e_s+x[1]*i_s+x[2]*(e_s*i_s)+x[3]
    
    #to avoid log10 throwing tantrums. but of course no "good" spectrum should have negative values
    c_s[c_s<=0] = 1e-10
    return c_s
    

def find_scaling(x, e_s, i_s, t_s):
    
    c_s = NF_to_empirical(x, e_s, i_s)    

    return np.linalg.norm(np.log10(c_s)-np.log10(t_s),ord=2)
####################################################################################################
#Loop for all semi-analytic calculations given parameter set and eigenvalues: HSS, LSA, PSD
####################################################################################################    

def Full_Analysis(Parameters, Laplacian_eigenvalues, Graph_Kernel, True_Temporal_Spectrum=None, min_omega=0, max_omega=300, delta_omega=0.5,
                  True_Spatial_Spectrum=None, first_k=2, last_k=None, bins=None, LSA=True, Visual=False, SaveFiles=False, Filepath=' ',
                  best_minDist = 800, disp_print=False):
   
    alpha_EE=Parameters[0]
    alpha_IE=Parameters[1]
    alpha_EI=Parameters[2]
    alpha_II=Parameters[3]
    d_e=Parameters[4]
    d_i=Parameters[5]
    P=Parameters[6]
    Q=Parameters[7]
    sigma_EE=Parameters[8]
    sigma_IE=Parameters[9]
    sigma_EI=Parameters[10]
    sigma_II=Parameters[11]
    D=1
    tau_e=Parameters[12] 
    tau_i=Parameters[13]

    if Graph_Kernel == 'Damped Wave':
        aDW_EE=Parameters[14]
        aDW_IE=Parameters[15]
        aDW_EI=Parameters[16]
        aDW_II=Parameters[17]
        bDW_EE=Parameters[18]
        bDW_IE=Parameters[19]
        bDW_EI=Parameters[20]
        bDW_II=Parameters[21]
    else:
        aDW_EE=0
        aDW_IE=0
        aDW_EI=0
        aDW_II=0
        bDW_EE=0
        bDW_IE=0
        bDW_EI=0
        bDW_II=0

    #sigma_noise_e=Parameters[15] 
    #sigma_noise_i=Parameters[16]   #only one sigma noise=scale_param
    
    #D=1.0
    #tau_e=1.0
    #tau_i=1.0
    #sigma_noise_e=1.0
    #sigma_noise_i=1.0
    
    eigs=Laplacian_eigenvalues
    
    if last_k is None:
        last_k=len(eigs)
    
    success = False
    #beginning of calculations
    steady_states, success = H_Simple_Steady_State(alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i, P, Q)
    found_suitable = False
    
    if success==True:
        nrSS=len(steady_states[0])

        #see linear stability analysis method for types
        SStypes=np.zeros(nrSS)
        
        if True_Spatial_Spectrum is not None:
            all_spatial_spectra = np.empty((nrSS,len(eigs),2,2), dtype=float)
            rescaled_spatial_spectra = np.empty((nrSS,len(True_Spatial_Spectrum)), dtype=float)
            
            
        #distance between eachSS power spectrum and true
        dist_spatial=np.zeros(nrSS)
        dist_temporal=np.zeros(nrSS)
        ###number of scaling params
        scale_params_spatial=np.zeros((nrSS,1))
        scale_params_temporal=np.zeros((nrSS,1))
               
            
        if True_Temporal_Spectrum is not None:

            E_temporal_spectrum = np.empty((nrSS,len(True_Temporal_Spectrum)), dtype=float)
            I_temporal_spectrum = np.empty((nrSS,len(True_Temporal_Spectrum)), dtype=float)
            rescaled_temporal_spectra = np.empty((nrSS,len(True_Temporal_Spectrum)), dtype=float)
            
            
            
            
            
        allJacEigs = np.empty((nrSS, len(eigs), 2), dtype=complex)
        
        for ss in range(len(steady_states[0])):
            
            Ess = steady_states[0,ss]
            Iss = steady_states[1,ss]
            
            
            if LSA==True:
                SStypes[ss], found_suitable, allJacEigs[ss,:,:] = GraphWC_Jacobian_TrDet(eigs, Graph_Kernel, Ess, Iss,                        
                                                 alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                                 sigma_EE, sigma_IE, sigma_EI, sigma_II, D, 
                                                 tau_e, tau_i,
                                                 aDW_EE, aDW_IE, aDW_EI, aDW_II,
                                                 bDW_EE, bDW_IE, bDW_EI, bDW_II, 
                                                 False)    
                
        
        
        if np.any(SStypes!=10):  

            for ss in range(len(steady_states[0])):
                
                Ess = steady_states[0,ss]
                Iss = steady_states[1,ss]          
                sigma_noise = 1
                        
                if True_Spatial_Spectrum is not None:
                    all_spatial_spectra[ss,:,:,:] = Graph_WC_Spatiotemporal_PowerSpectrum(eigs, Graph_Kernel, Ess, Iss, 
                                                  alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                                  sigma_EE, sigma_IE, sigma_EI, sigma_II, D,                      
                                                  tau_e, tau_i,                                                 
                                                  aDW_EE, aDW_IE, aDW_EI, aDW_II,
                                                  bDW_EE, bDW_IE, bDW_EI, bDW_II, 
                                                  sigma_noise_e=sigma_noise, sigma_noise_i=sigma_noise,
                                                  Spatial_Spectrum_Only=True, Visual=False)
                    
                    
                    if bins is None:
                        E_spatial_spectrum = all_spatial_spectra[ss,first_k:last_k,0,0]
                        I_spatial_spectrum = all_spatial_spectra[ss,first_k:last_k,1,1]
                        SPS_points = np.arange(first_k,last_k)
                    else:
                        E_spatial_spectrum = np.array([np.median(elem) for elem in np.array_split(all_spatial_spectra[ss,first_k:last_k,0,0], bins)])
                        I_spatial_spectrum = np.array([np.median(elem) for elem in np.array_split(all_spatial_spectra[ss,first_k:last_k,1,1], bins)])                                           
                        SPS_points = np.array([elem.mean() for elem in np.array_split(np.arange(first_k,last_k), bins)])
                            
                    # a_matrix_spatial = np.vstack((E_spatial_spectrum,
                    #                              I_spatial_spectrum,
                    #                              #E_spatial_spectrum*I_spatial_spectrum,
                    #                              np.ones_like(True_Spatial_Spectrum))).T
                    #scale_params_spatial[ss,:] = np.linalg.lstsq(a_matrix_spatial, True_Spatial_Spectrum)[0]
                    
                    #normally use this
                    scale_params_spatial[ss,:] = sp.optimize.fmin(find_scaling, x0=[1,0,0], ftol=1e-5, xtol=1e-5, args=(E_spatial_spectrum,I_spatial_spectrum,True_Spatial_Spectrum), disp=0)
                    #scale_params_spatial[ss,:] = (True_Spatial_Spectrum.mean())/(E_spatial_spectrum.mean())
#                    n_spatial = len(True_Spatial_Spectrum)    
#                    a_spatial = (n_spatial*np.dot(E_spatial_spectrum,True_Spatial_Spectrum)-np.sum(True_Spatial_Spectrum)*np.sum(E_spatial_spectrum))/(n_spatial*np.dot(E_spatial_spectrum,E_spatial_spectrum)-np.sum(E_spatial_spectrum)**2)
#                    b_spatial = (np.sum(True_Spatial_Spectrum)-a_spatial*np.sum(E_spatial_spectrum))/n_spatial
#                    scale_params_spatial[ss,:] = np.array([a_spatial,b_spatial])
#                    
                    current_spatial_spectrum = NF_to_empirical(scale_params_spatial[ss,:],
                                                               E_spatial_spectrum,
                                                               I_spatial_spectrum)
                    
                    
                    rescaled_spatial_spectra[ss,:] = np.copy(current_spatial_spectrum)
                    
                    data_1=np.vstack((SPS_points, np.log10(True_Spatial_Spectrum))).T
                    data_2=np.vstack((SPS_points, np.log10(current_spatial_spectrum))).T   

                    dist_spatial[ss] = (1+sm.area_between_two_curves(data_1,data_2))**1

                    #np.linalg.norm(np.log10(True_Spatial_Spectrum) - np.log10(current_spatial_spectrum), ord=1)#sm.area_between_two_curves(data_1,data_2)#np.linalg.norm((np.log10(True_Spatial_Spectrum) - np.log10(current_spatial_spectrum), ord=1)#np.linalg.norm(True_Spatial_Spectrum - a_spatial*current_spatial_spectrum-b_spatial, ord=2)#1-sp.stats.ks_2samp(True_Spatial_Spectrum, current_spatial_spectrum*a_spatial+b_spatial)[1]#1-np.ma.corrcoef(True_Spatial_Spectrum, current_spatial_spectrum)[0,1]#
                    
                    dist_spatial[ss] += (1+np.linalg.norm(np.log10(True_Spatial_Spectrum)-np.log10(current_spatial_spectrum), ord=2))**2

                    dist_spatial[ss] += (1+np.linalg.norm((np.log10(True_Spatial_Spectrum[1:])-np.log10(True_Spatial_Spectrum[:-1]))-(np.log10(current_spatial_spectrum[1:])-np.log10(current_spatial_spectrum[:-1])), ord=2))**2

                    #dist_spatial[ss] = np.corrcoef(np.log10(current_spatial_spectrum),np.log10(True_Spatial_Spectrum))[0,1]
                    
                    
                if True_Temporal_Spectrum is not None:
                    E_Spectrum, I_Spectrum = Graph_WC_Spatiotemporal_PowerSpectrum(eigs, Graph_Kernel, Ess, Iss, 
                                                  alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                                  sigma_EE, sigma_IE, sigma_EI, sigma_II, D,                      
                                                  tau_e, tau_i,                                                 
                                                  aDW_EE, aDW_IE, aDW_EI, aDW_II,
                                                  bDW_EE, bDW_IE, bDW_EI, bDW_II, 
                                                  sigma_noise_e=1, sigma_noise_i=1,
                                                  min_omega=min_omega, max_omega=max_omega, delta_omega=delta_omega,
                                                  Spatial_Spectrum_Only=False, Visual=False)
                    
                    E_temporal_spectrum[ss,:] = 2*np.sum(E_Spectrum,axis=1)
                    I_temporal_spectrum[ss,:] = 2*np.sum(I_Spectrum,axis=1)
                                        
                    # a_matrix_temporal = np.vstack((E_temporal_spectrum[ss,:],
                    #                                I_temporal_spectrum[ss,:],
                    #                                #E_temporal_spectrum[ss,:]*I_temporal_spectrum[ss,:],
                    #                                np.ones_like(True_Temporal_Spectrum))).T
                    # scale_params_temporal[ss,:] = np.linalg.lstsq(a_matrix_temporal, True_Temporal_Spectrum)[0]
                    
                    #normally use this
                    scale_params_temporal[ss,:] = sp.optimize.fmin(find_scaling, x0=[1,0,0], ftol=1e-5, xtol=1e-5, args=(E_temporal_spectrum[ss,:],I_temporal_spectrum[ss,:],True_Temporal_Spectrum), disp=0)
                    #scale_params_temporal[ss,:] = (True_Temporal_Spectrum.mean())/(E_temporal_spectrum.mean())
                    
                    current_temporal_spectrum = NF_to_empirical(scale_params_temporal[ss,:],
                                                                E_temporal_spectrum[ss,:],
                                                                I_temporal_spectrum[ss,:])
                                               
                    
                    rescaled_temporal_spectra[ss,:] = np.copy(current_temporal_spectrum)
                    
                    data_3=np.vstack((np.arange(min_omega,max_omega,delta_omega),np.log10(True_Temporal_Spectrum))).T
                    data_4=np.vstack((np.arange(min_omega,max_omega,delta_omega),np.log10(current_temporal_spectrum))).T        
                    dist_temporal[ss] = (1+sm.area_between_two_curves(data_3,data_4))**3#np.linalg.norm(np.log10(True_Temporal_Spectrum) - np.log10(current_temporal_spectrum), ord=1)#sm.area_between_two_curves(data_3,data_4)#np.linalg.norm(True_Temporal_Spectrum - current_temporal_spectrum, ord=1)##np.linalg.norm(True_Temporal_Spectrum - a_temporal*current_temporal_spectrum-b_temporal, ord=2)#1-sp.stats.ks_2samp(True_Temporal_Spectrum, current_temporal_spectrum*a_temporal+b_temporal)[1]#1-np.ma.corrcoef(True_Temporal_Spectrum, current_temporal_spectrum)[0,1]#
                    dist_temporal[ss] += (1+np.linalg.norm(np.log10(True_Temporal_Spectrum)-np.log10(current_temporal_spectrum), ord=2))**3

                    dist_temporal[ss] += (1+np.linalg.norm((np.log10(True_Temporal_Spectrum[1:])-np.log10(True_Temporal_Spectrum[:-1]))-(np.log10(current_temporal_spectrum[1:])-np.log10(current_temporal_spectrum[:-1])), ord=2))**3

                    #dist_temporal[ss] = np.corrcoef(np.log10(True_Temporal_Spectrum),np.log10(current_temporal_spectrum))[0,1]
                            
            ########*******######
            #important: insert here a metric to quantify distance between true spectrum and calculated
            ######****######
            
            #currently giving a stronger weight to the temporal distance
            if (scale_params_spatial+scale_params_temporal).sum()<10**6:
                Dist=dist_temporal+dist_spatial#(3*dist_temporal)**2+10**(dist_spatial)
            else:
                Dist=10**9*np.ones_like(dist_temporal)

            
  
            mask = np.argwhere((SStypes!=10)) #* (scale_params_spatial[:,0]>0)) 
            if ~np.all(np.isnan(Dist[mask])):# and np.abs((a_temporal+np.abs(b_temporal)+a_spatial+np.abs(b_spatial)))<1e13:
                Dist[SStypes==0] **= 2
                bestSSS = mask[np.nanargmin(Dist[mask])][0]
                minDist=Dist[bestSSS]

                if minDist<best_minDist:
                    best_minDist = np.copy(minDist)
                    print(repr(Parameters))
                

                 
                if True_Spatial_Spectrum is not None:
                    best_spatial_spectrum = rescaled_spatial_spectra[bestSSS,:]
                
                if True_Temporal_Spectrum is not None:
                    best_temporal_spectrum = rescaled_temporal_spectra[bestSSS,:]


                if disp_print:
                    print(f"Best suitable steady state: {bestSSS}, with Ess={steady_states[0,bestSSS]:.4g} Iss={steady_states[1,bestSSS]:.4g}. \
                      \nDist spatial: {dist_spatial[bestSSS]:.4g}, scale params: {scale_params_spatial[bestSSS]}  \
                      \nDist temporal: {dist_temporal[bestSSS]:.4g}, scale params: {scale_params_temporal[bestSSS]}\n\
                       total dist: {minDist:.4g}")

                
                plt.ioff()
                if Visual==True:
                    plt.ion()
                    fig = plt.figure()
                    #ax = fig.add_subplot(111)
                    #ax.set_xlim(-0.1, 20000)
                    #ax.set_ylim(0, 20)
                    plt.scatter(np.ravel(allJacEigs[bestSSS,:,:]).real,np.ravel(allJacEigs[bestSSS,:,:]).imag, s=2, c='black')                   
                    
                    
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    #ax.set_xlim(-0.1, 20000)
                    ax.set_ylim(1E-2, 1E2)
                    #line2, = plt.loglog(np.arange(1,len(eigs)+1),bestG[:,1,1], 'b-')
                    ax.set_title("Spatial Power Spectrum")
                    ax.set_xlabel("Spatial Eigenmode ($k$)")
                    line1, = plt.loglog(np.arange(1,len(eigs)+1),0.8*best_spatial_spectrum[:,0,0], linewidth=2)#, 'b-')
                    line3, = plt.loglog(np.arange(first_k+1,last_k+1),True_Spatial_Spectrum, 'b--', linewidth=2)
                 
                  
                    
                
               
                if SaveFiles==True:
                    
                    if Filepath==' ':
                        filepath = 'G:/Macbook Stuff/Results/'+Graph_Kernel+' Kernel/aEE=%.3f aIE=%.3f aEI=%.3f aII=%.3f dE=%.3f dI=%.3f ' %(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i)
                        filepath += 'P=%.3f Q=%.3f sEE=%.3f sIE=%.3f sEI=%.3f sII=%.3f D=%.3f tE=%.3f tI=%.3f/'%(P,Q,sigma_EE,sigma_IE,sigma_EI,sigma_II,D,tau_e,tau_i) 
                    else:
                        filepath=Filepath
                        
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    
                    file = open(filepath+'Parameters.dat', 'w+')        
                    file.write("Alpha_EE=%f \n"%alpha_EE)
                    file.write("Alpha_IE=%f \n"%alpha_IE)
                    file.write("Alpha_EI=%f \n"%alpha_EI)
                    file.write("Alpha_II=%f \n"%alpha_II)
                    file.write("d_E=%f \n"%d_e)
                    file.write("d_I=%f \n"%d_i)
                    file.write("P=%f \n"%P)
                    file.write("Q=%f \n"%Q)
                    file.write("Sigma_EE=%f \n"%sigma_EE)
                    file.write("Sigma_IE=%f \n"%sigma_IE)
                    file.write("Sigma_EI=%f \n"%sigma_EI)
                    file.write("Sigma_II=%f \n"%sigma_II)
                    file.write("D=%f \n"%D)
                    file.write("Tau_E=%f \n"%tau_e)
                    file.write("Tau_I=%f \n"%tau_i)
                    #file.write("Sigma_Noise_E=%f \n"%sigma_noise_e)
                    #file.write("Sigma_Noise_I=%f \n"%sigma_noise_i)       
                    file.close
                    
                    with h5py.File(filepath+"analysis.h5") as hf:
                        if "Steady States" not in list(hf.keys()):
                            hf.create_dataset("Steady States",  data=steady_states)
                            hf.create_dataset("Distance",  data=Dist)
                            hf.create_dataset("Scale", data=scale_params)
                            hf.create_dataset("Type",  data=SStypes)
                            hf.create_dataset("AllG", data=allG)
                        else:
                            data=hf["Steady States"]
                            data[...]=steady_states
                            if "AllG" in list(hf.keys()):                            
                                data=hf["Distance"]
                                data[...]=Dist
                                data=hf["Scale"]
                                data[...]=scale_params
                                data=hf["AllG"]
                                data[...]=allG
                            
                        
                    
                    if Visual==True:
                        plt.savefig(filepath+"Power Spectrum.pdf")   
                    
                        
                #if G[3,0,0]-G[-3,0,0]<=1:
                # minDist=10000*np.random.rand()+1000
                
                                   
                return minDist
            else:
                #nans in spectra
                #print("Unrealistic spectra or scaling")
                return 1e9+np.random.rand()#1e10+np.max(allJacEigs.real)
        else:
            #all unstable SS
            #print("No suitable (LSA) steady states found")
            return 1e9+np.random.rand()#1000.0+500*np.random.rand()#float('Inf')
    
    else:
        #case where no positive/exact solutions found (can #print from SS method)
        return 1e9+np.random.rand()
    
    
#utility functions to read data in python format from Selen's files
def construct_fibers_from_data(filepath_data,
                               filepath_Fibers,
                               savefiles=True,
                               bundle_size=1,
                               output_filepath_fiber_edges=None,
                               output_filepath_fiber_lengths=None,
                               output_filepath_fiber_dist_starts=None,
                               output_filepath_fiber_dist_ends=None,
                               output_filepath_fiber_ends=None):
    
    #important: this sets which fiber dataset is used for computation of edges
    if filepath_Fibers.endswith('.npy'):
        Fibers = np.load(filepath_Fibers,allow_pickle=True)
    else:
        with h5py.File(filepath_Fibers, 'r') as file:
            Fibers=[file[element][:] for element in file['fgCC']['fibers'][0]]
        
    with h5py.File(filepath_data, 'r') as file:    
        AllVet=np.asarray(file['vertices']['all'])    
        
     
    fiber_lengths=np.zeros(len(Fibers),dtype=float)
    fiber_start=np.zeros((len(Fibers),3),dtype=float)
    fiber_end=np.zeros((len(Fibers),3),dtype=float)
    for i in range(len(Fibers)):
        fiber_start[i] = Fibers[i][0]
        fiber_end[i] = Fibers[i][-1]   
        for j in range(len(Fibers[i])-1):    
            fiber_lengths[i] += np.linalg.norm(Fibers[i][j+1]-Fibers[i][j], ord=2)
            
    #can also create bundles instead of single edges if bundle_size>1
    
    mesh_fiber_nodes = np.zeros((len(Fibers),2,bundle_size))
    
    dist_starts=[]
    dist_ends=[]
    
    for i in range(fiber_start.shape[0]):
        #print(i)
        dist_start=np.Inf
        dist_end=np.Inf
        
        all_dists_end=[]
        all_dists_start=[]
        
        for j in range(AllVet.shape[1]):
            dist_start_new = np.linalg.norm(AllVet[:,j] - fiber_start[i,:], ord=2)
            dist_end_new = np.linalg.norm(AllVet[:,j] - fiber_end[i,:], ord=2)
            
            all_dists_end.append(dist_end_new)
            all_dists_start.append(dist_start_new)
    
            if dist_start_new<dist_start:
                dist_start=dist_start_new
                mesh_start=j
            if dist_end_new<dist_end:  
                dist_end=dist_end_new
                mesh_end=j
        
        #create fiber bundles instead of just fiber edges
        idx_start=np.argpartition(np.array(all_dists_start), bundle_size)[:bundle_size]
        idx_end=np.argpartition(np.array(all_dists_end), bundle_size)[:bundle_size]
        
        #if mesh_start not in idx_start or mesh_end not in idx_end:
            #print("sanity check failed. double check")
                   
        mesh_fiber_nodes[i,0,:]=idx_start#mesh_start
        mesh_fiber_nodes[i,1,:]=idx_end#mesh_end
        dist_starts.append(dist_start)
        dist_ends.append(dist_end)

    
    mesh_fiber_nodes=mesh_fiber_nodes.astype(int)
    
    if savefiles==True:                           
        np.save(output_filepath_fiber_edges, mesh_fiber_nodes)
        np.save(output_filepath_fiber_lengths, fiber_lengths)   
        np.save(output_filepath_fiber_dist_starts, dist_starts)  
        np.save(output_filepath_fiber_dist_ends, dist_ends)
        np.save(output_filepath_fiber_ends, fiber_end)
        return
    else:
        return mesh_fiber_nodes, fiber_lengths, dist_starts, dist_ends, fiber_end
    
    
    
def construct_adjacency_matrix_from_data(filepath_data,
                                        
                                       filepath_fiber_edges,
                                       filepath_fiber_lengths,
                                       filepath_fiber_ends,
                                       

                                       include_subcortex=False,
                                       add_DTI=True,
                                       fiber_speed_factor=100,
                                       threshold=False,
                                       max_dist=10,
                                       filepath_fiber_dist_starts=None,
                                       filepath_fiber_dist_ends=None,
                                       
                                       visual=True,
                                       plot_subcortex=False,
                                       plot_DTI_edges=False
                                       ):

    with h5py.File(filepath_data, 'r') as file:
        ##print(list(file.keys()))

        faces=np.asarray(file['faces']['all'], dtype=int)
        AllVet=np.asarray(file['vertices']['all'])
        AllVet_comp=np.asarray(file['vertices']['all'])
                
        if plot_subcortex==False:
            CC = np.asarray(file['CC']['restInds'], dtype=int)
            indices = np.array([elem[0] for elem in CC])-1
            AllVet=AllVet[:,indices]
    
        
    Xn=AllVet[0,:]
    Yn=AllVet[1,:]
    Zn=AllVet[2,:]
    
    iN=[]
    jN=[]
    kN=[]

    mesh_adjacency = sparse.lil_matrix(np.zeros((20484,20484)))
    
    #print("Constructing mesh adjacency matrix...")
    for p in range(faces.shape[1]):
        i=faces[0,p]-1
        j=faces[1,p]-1
        k=faces[2,p]-1
        
        #constructing the mesh adjacency matrix from the faces data. This still includes subcortical structures, but no DTI yet
        #edges are weighted according to the inverse squared distance (to obtain a metric graph Laplacian)
        mesh_adjacency[i,j]=1/np.linalg.norm(AllVet_comp[:,i] - AllVet_comp[:,j], ord=2)**2
        mesh_adjacency[j,k]=1/np.linalg.norm(AllVet_comp[:,j] - AllVet_comp[:,k], ord=2)**2
        mesh_adjacency[k,i]=1/np.linalg.norm(AllVet_comp[:,k] - AllVet_comp[:,i], ord=2)**2
        mesh_adjacency[j,i]=mesh_adjacency[i,j]
        mesh_adjacency[k,j]=mesh_adjacency[j,k]
        mesh_adjacency[i,k]=mesh_adjacency[k,i]
        
        #set up mesh data for later plotting with plotly. can choose to include subcortex or not.
        if visual==True:
            if plot_subcortex==True:
                iN.append(i)
                jN.append(j)
                kN.append(k)
            else:
                if i in indices and j in indices and k in indices:
                    node1 = np.where(indices==i)
                    node2 = np.where(indices==j)
                    node3 = np.where(indices==k)
                    iN.append(node1)
                    jN.append(node2)
                    kN.append(node3)
    
    #set up graph mesh edges for plotly visualization
    if visual==True and plot_DTI_edges==False:
        if plot_subcortex==False:
            mesh_adjacency_no_subcortex=mesh_adjacency[:,indices]
            mesh_adjacency_no_subcortex=mesh_adjacency_no_subcortex[indices,:]       
            Adj_mesh=sparse.triu(mesh_adjacency_no_subcortex)
        else:
            Adj_mesh=sparse.triu(mesh_adjacency)

        Edges_mesh=Adj_mesh.nonzero()
        Edges_mesh_starts=Edges_mesh[0]
        Edges_mesh_ends=Edges_mesh[1]

        Xe=[]
        Ye=[]
        Ze=[]

        for i in range(len(Edges_mesh_starts)):
            Xe+=[AllVet[0,Edges_mesh_starts[i]],AllVet[0,Edges_mesh_ends[i]], None]# x-coordinates of edge ends
            Ye+=[AllVet[1,Edges_mesh_starts[i]],AllVet[1,Edges_mesh_ends[i]], None]
            Ze+=[AllVet[2,Edges_mesh_starts[i]],AllVet[2,Edges_mesh_ends[i]], None]
        
    
    #previously, we calculated the 3D lengths of DTI fibers, and the nodes on the mesh nearest to each fiber beginning/end
    #(see the relevant function "construct_fibers_from_data" for details)
    DTI_edges=np.load(filepath_fiber_edges[0])
    fiber_lengths=np.load(filepath_fiber_lengths[0])
    #fiber_end=np.load(filepath_fiber_ends[0])
    
    #also, we calculated the distance between the fiber beginning/end and the mesh.
    #this data can be read and used to apply a threshold to the fibers based on the distance from the mesh.
    fiber_dist_starts=np.load(filepath_fiber_dist_starts[0])
    fiber_dist_ends=np.load(filepath_fiber_dist_ends[0])
    
    #if threshold is set to false, simply set the maximum allowed distance to infinity s.t. all fibers are included
    if threshold==False:
        max_dist=np.inf
    
    if add_DTI==True:
        #print("Now adding DTI fibers from "+filepath_fiber_edges[0]+"...")                
        #loop over all fibers
        for i in range(DTI_edges.shape[0]):      
            #threshold loop. trivial if threshold is set to false
            if fiber_dist_starts[i]<=max_dist and fiber_dist_ends[i]<=max_dist:
                #fibers vs bundles
                if DTI_edges.shape[-1]>2:
                    #avoid auto-edges in the graph
                    if np.any(DTI_edges[i,0,:] in DTI_edges[i,1,:]):
                        #print("Degenerate fiber. Skipping.")
                        pass
                    else:
                        #create all-to-all bundles
                        for j in range(DTI_edges.shape[-1]):                            
                            mesh_adjacency[DTI_edges[i,0,j],DTI_edges[i,1,:]]=np.ones(DTI_edges.shape[-1])*(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2    
                            mesh_adjacency[DTI_edges[i,1,:],DTI_edges[i,0,j]]=np.ones((DTI_edges.shape[-1],1))*(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2  
                else:
                    #fiber edges not bundles
                    #avoid auto edges
                    if DTI_edges[i,1]!=DTI_edges[i,0]:
                        mesh_adjacency[DTI_edges[i,0],DTI_edges[i,1]]=(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2    
                        mesh_adjacency[DTI_edges[i,1],DTI_edges[i,0]]=(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2 

    
    if len(filepath_fiber_edges)>1:
        #Repeat for the two fiber paths, obtained from fg and fgCC datasets respectively
        DTI_edges=np.load(filepath_fiber_edges[1])
        fiber_lengths=np.load(filepath_fiber_lengths[1])
        #fiber_end=np.load(filepath_fiber_ends[1])
        
        #also, we calculated the distance between the fiber beginning/end and the mesh.
        #this data can be read and used to apply a threshold to the fibers based on the distance from the mesh.
        fiber_dist_starts=np.load(filepath_fiber_dist_starts[1])
        fiber_dist_ends=np.load(filepath_fiber_dist_ends[1])
        
        
        if add_DTI==True:
            #print("Now adding DTI fibers from "+filepath_fiber_edges[1]+"...")                
            #loop over all fibers
            for i in range(DTI_edges.shape[0]):      
                #threshold loop. trivial if threshold is set to false
                if fiber_dist_starts[i]<=max_dist and fiber_dist_ends[i]<=max_dist:
                    #fibers vs bundles
                    if DTI_edges.shape[-1]>2:
                        #avoid auto-edges in the graph
                        if np.any(DTI_edges[i,0,:] in DTI_edges[i,1,:]):
                            #print("Degenerate fiber. Skipping.")
                            pass
                        else:
                            #create all-to-all bundles
                            for j in range(DTI_edges.shape[-1]):                            
                                mesh_adjacency[DTI_edges[i,0,j],DTI_edges[i,1,:]]=np.ones(DTI_edges.shape[-1])*(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2    
                                mesh_adjacency[DTI_edges[i,1,:],DTI_edges[i,0,j]]=np.ones((DTI_edges.shape[-1],1))*(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2  
                    else:
                        #fiber edges not bundles
                        #avoid auto edges
                        if DTI_edges[i,1]!=DTI_edges[i,0]:
                            mesh_adjacency[DTI_edges[i,0],DTI_edges[i,1]]=(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2    
                            mesh_adjacency[DTI_edges[i,1],DTI_edges[i,0]]=(fiber_speed_factor/(fiber_lengths[i]+fiber_dist_starts[i]+fiber_dist_ends[i]))**2 
    

    
    if visual==True and plot_DTI_edges==True:
        if plot_subcortex==False:
            mesh_adjacency_no_subcortex=mesh_adjacency[:,indices]
            mesh_adjacency_no_subcortex=mesh_adjacency_no_subcortex[indices,:]       
            Adj_mesh=sparse.triu(mesh_adjacency_no_subcortex)
        else:
            Adj_mesh=sparse.triu(mesh_adjacency)

        Edges_mesh=Adj_mesh.nonzero()
        Edges_mesh_starts=Edges_mesh[0]
        Edges_mesh_ends=Edges_mesh[1]

        Xe=[]
        Ye=[]
        Ze=[]

        for i in range(len(Edges_mesh_starts)):
            Xe+=[AllVet[0,Edges_mesh_starts[i]],AllVet[0,Edges_mesh_ends[i]], None]# x-coordinates of edge ends
            Ye+=[AllVet[1,Edges_mesh_starts[i]],AllVet[1,Edges_mesh_ends[i]], None]
            Ze+=[AllVet[2,Edges_mesh_starts[i]],AllVet[2,Edges_mesh_ends[i]], None]
        
    
    
    #only cortical nodes and edges in output
    if include_subcortex==False:
        mesh_adjacency=mesh_adjacency[indices,:]
        mesh_adjacency=mesh_adjacency[:,indices]
    
    if visual==True:
        return mesh_adjacency, Xn, Yn, Zn, iN, jN, kN,  Xe, Ye, Ze 
    else:
        return mesh_adjacency