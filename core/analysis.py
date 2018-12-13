import numpy as np
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import h5py
import os
from scipy import sparse
plt.rcParams.update({'font.size': 20})
plt.tight_layout()
####################################################################################################
####################################################################################################
#written for python 3.6 

def GraphKernel(x,t,type='Gaussian', a=1, b=1, prime=False):
    if t<0:
        print("Need positive kernel parameter")
        return
    else:
        if type=='Gaussian':
            #add prefactor to make kernel of unitary height
            return np.exp(t*x)#*2*np.sqrt(t*np.pi)
        elif type=='Exponential':
            return 2*t/(t**2-x)
        elif type=='Pyramid':
            #return t*(np.sinc(t*np.sqrt(-x)/(2*np.pi)))**2
            return np.sinc(np.sqrt(-x)*t/(2*np.pi))**2
        elif type=='Rectangle':   
            #Note: significant Gibbs effect makes this not advisable
            rect=t*(np.sinc(t*np.sqrt(-x)/(2*np.pi)))
            #rect[-990:]=0
            return rect
        elif type=='Mexican Hat':
            return -x*np.exp(t*x) #*2*np.sqrt(t*np.pi)  #*2*t  
        elif type=='Damped Wave':
            #make a smaller: wave travels faster
            a=0.3
            #make b larger: more diffusion
            b=0.001
            c=0
            r_1=(-b+sp.sqrt(b**2 + 4*a*(x-c)))/(2*a)
            r_2=(-b-sp.sqrt(b**2 + 4*a*(x-c)))/(2*a)
            Damped_Wave_Kernel=(r_1*sp.exp(r_2*t)-r_2*sp.exp(r_1*t))/(r_1-r_2)
            Damped_Wave_Kernel_prime=(sp.exp(r_1*t)-sp.exp(r_2*t))/(r_1-r_2)
            if np.any(Damped_Wave_Kernel.imag!=0) or np.any(Damped_Wave_Kernel_prime.imag!=0):
                print("Imaginary value in kernel")
                return
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
        indices1=np.arange(500,500+syn)
        indices2=np.arange(700,700+syn)
        for k1 in indices1:
            for k2 in indices2:
                if k1!=k2:                
                    dist=h*np.abs(k1-k2)                
                    AdjMatrix[k1,k2]=1/(dist**2)
                    AdjMatrix[k2,k1]=1/(dist**2)
    
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
        return s[::-1]
    else:
        s,U=np.linalg.eigh(Laplacian)
        s[-1]=0
        U[:,-1]=np.zeros(len(s))
        #note that the vectors come out 2-normalized
        return s[::-1], U[:,::-1]
        
####################################################################################################
####################################################################################################
####################################################################################################
#Implementing a solver for the homogeneous steady state, in the simple case with space-independent parameters
#Implementation with multiple initial condition to find all steady states numerically
#thresholding unique steady state if norm(x1-x2)<0.01    
def H_Simple_Steady_State(alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1, d_e=1, d_i=1, P=0, Q=0):
    #generate multiple initial conditions to find all steady states
    initial_guesses = 10
    #print("%.3g %.3g %.3g %.3g %.3g %.3g %.3g %.3g"%(alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i, P, Q))

    x0 = np.zeros((2,initial_guesses))
    x0[:,1] = np.array([1/(2*d_e), 1/(2*d_i)])
    x0[:,2] = np.random.rand(2)
    x0[:,3] = np.random.rand(2)
    x0[:,4] = 5*np.random.rand(2)
    x0[:,5] = 10*np.random.rand(2)
    x0[:,6] = np.ones(2)
    x0[:,7] = 2*np.random.rand(2)
    x0[:,8] = 20*np.random.rand(2)
    x0[:,9] = 20*np.ones(2)
    
    success = False
    results = np.zeros((2,initial_guesses))
    
    def f(x, alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i):       
        d = np.array([[d_e,0],[0,d_i]], dtype=float)
        alpha = np.array([[alpha_EE,-alpha_IE],[alpha_EI,-alpha_II]], dtype=float)
        X = np.array([P,Q], dtype=float)
        
        SS_EQ = - np.dot(d,x) + sp.special.expit(np.dot(alpha,x) + X)
        return SS_EQ
    
    for i in range(initial_guesses):
        steady_state = sp.optimize.fsolve(f,x0[:,i],args=(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i),
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
        tolerance=0.01
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
        
    #    print(str(len(finals[0]))+" unique steady states were found")        
       
        return finals, success
    else:
  #      print("No positive, exact solutions were found")
        return None, success

####################################################################################################
####################################################################################################
#Implementing the explicit calculation of linearised Jacobian trace and determinant 
#for the space-independent homogeneous case
#given only the Wilson-Cowan model parameters and Laplacian eigenspectrum
def GraphWC_Jacobian_TrDet(Laplacian_eigenvalues, Graph_Kernel='Gaussian', Ess=None, Iss=None, 
                       alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1, d_e=1, d_i=1, 
                       sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1, 
                       tau_e=1, tau_i=1, Visual=False):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D
        
    eigs=Laplacian_eigenvalues
    
    Jacobian_eigenvalues=np.zeros((len(eigs),2),dtype=complex)
    SStype=0
    suitable=False

    
    if Ess == None or Iss == None:
        print("Steady state solution not provided. Using simplest approximation.")
        Ess = 1/(2*d_e)
        Iss = 1/(2*d_i)
    
    a = d_e*Ess*(1-d_e*Ess)
    b = d_i*Iss*(1-d_i*Iss)
    
 
    
    Trace = -(d_e/tau_e+d_i/tau_i) + alpha_EE*a*GraphKernel(eigs,t_EE,type=Graph_Kernel)/tau_e - alpha_II*b*GraphKernel(eigs,t_II,type=Graph_Kernel)/tau_i
    
    Determinant = -alpha_EE*alpha_II*a*b*GraphKernel(eigs,t_EE+t_II,type=Graph_Kernel)/(tau_e*tau_i) - alpha_EE*a*d_i*GraphKernel(eigs,t_EE,type=Graph_Kernel)/(tau_e*tau_i) + alpha_II*b*d_e*GraphKernel(eigs,t_II,type=Graph_Kernel)/(tau_e*tau_i) + alpha_IE*alpha_EI*a*b*GraphKernel(eigs,t_EI+t_IE,type=Graph_Kernel) + d_e*d_i/(tau_e*tau_i)
    
    
    Jacobian_eigenvalues[:,0]= (Trace + sp.sqrt(Trace**2 - 4*Determinant))/2.0
    Jacobian_eigenvalues[:,1]= (Trace - sp.sqrt(Trace**2 - 4*Determinant))/2.0
    
    if Visual==True:
        plt.ion()
        fig = plt.figure()
        color=np.repeat(np.linspace(0,1,len(eigs)),2)[::-1]
        #ax = fig.add_subplot(111)
        #ax.set_xlim(-0.1, 20000)
        #ax.set_ylim(0, 20)
        plt.scatter(np.ravel(Jacobian_eigenvalues).real,np.ravel(Jacobian_eigenvalues).imag, marker='o', s=2, c=color, cmap='nipy_spectral')#, edgecolor='black', linewidth=0.1)
    
    if np.any(Jacobian_eigenvalues.real>-0.1) or np.any(np.isnan(Jacobian_eigenvalues)):
        print("E*=%.4f, I*=%.4f: unstable"%(Ess,Iss))
        SStype=0
        suitable = False
    else:
                  
        if np.all(Jacobian_eigenvalues.real<0) and np.all(Jacobian_eigenvalues.imag==0):
            print("E*=%.4f, I*=%.4f: strictly stable"%(Ess,Iss))
            SStype=1
            suitable = True
      #all or any in the line below for imaginary? ask rikkert #do they all need imaginary parts?
        elif np.all(Jacobian_eigenvalues.real<0) and np.any(Jacobian_eigenvalues.imag != 0):
            print("E*=%.4f, I*=%.4f: stable, with nonzero imaginary components"%(Ess,Iss))
            SStype=2
            suitable = True
                #same question here. what if some imaginary are zero and some nonzero?
        elif np.all(Jacobian_eigenvalues.real==0) and np.all(Jacobian_eigenvalues.imag != 0):
            print("E*=%.4f, I*=%.4f: all purely imaginary eigenvalues (potential Hopf)"%(Ess,Iss))
            SStype=3
            suitable = True
          
        else:
 
#cases with BOTH zero AND (negative) nonzero real components of eigens. 
#first subcase: all imaginary parts are zero (stable/undetermined?)
#second subcase: some imaginary parts are nonzero (need to consider overlap? does it matter if there are eigenvalues with magnitude zero?)
#third subcase: all imaginary parts are nonzero (same question)
            print("E*=%.4f, I*=%.4f: undetermined"%(Ess,Iss))
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
                       sigma_noise_e=1, sigma_noise_i=1, max_omega=100, delta_omega=0.1,
                       Spatial_Spectrum_Only=True, Visual=False):
    
    t_EE = (0.5*sigma_EE**2)/D
    t_IE = (0.5*sigma_IE**2)/D
    t_EI = (0.5*sigma_EI**2)/D    
    t_II = (0.5*sigma_II**2)/D
    eigs=Laplacian_eigenvalues
   
    
    if Ess == None or Iss == None:
        print("Steady state solution not provided. Using simplest approximation.")
        Ess = 1/(2*d_e)
        Iss = 1/(2*d_i)
    
    a = d_e*Ess*(1-d_e*Ess)
    b = d_i*Iss*(1-d_i*Iss)
        
    if Spatial_Spectrum_Only==True:
        
        Gmatrix2 = np.zeros((len(eigs),2,2), dtype=float)
  
        G_EE = alpha_EE*GraphKernel(eigs,t_EE,type=Graph_Kernel)
        G_IE = alpha_IE*GraphKernel(eigs,t_IE,type=Graph_Kernel)
        G_EI = alpha_EI*GraphKernel(eigs,t_EI,type=Graph_Kernel)
        G_II = alpha_II*GraphKernel(eigs,t_II,type=Graph_Kernel)
        
        Gmatrix2[:,0,0] = 0.5*((sigma_noise_e**2)/(tau_i*d_e-tau_i*a*G_EE+d_i*tau_e+b*tau_e*G_II))*((tau_i/tau_e) + ((a**2)*(G_IE**2)+ (d_i + b*G_II)**2)/(d_e*d_i+ (d_e*b*G_II) - (a*d_i*G_EE) - a*b*(G_EE*G_II-G_EI*G_IE)))
        
        
        if Visual==True:
            plt.ion()
            fig = plt.figure()
            #ax = fig.add_subplot(111)
            #ax.set_xlim(-0.1, 20000)
            #ax.set_ylim(0, 20)
            #line2, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,1,1], 'b-')
            #line1, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,0,0], 'r-')
            line3, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix2[:,0,0], 'r-')   
            
        return Gmatrix2                 
    else:
        omegas=int(max_omega/delta_omega)
        Full_Spectrum=np.zeros((len(eigs),omegas), dtype=float)
        
        Dmatrix=np.array([[sigma_noise_e/tau_e,0],[0,sigma_noise_i/tau_i]])**2
       
        A = np.stack([[d_e/tau_e - a*alpha_EE*GraphKernel(eigs,t_EE,type=Graph_Kernel)/tau_e, a*alpha_IE*GraphKernel(eigs,t_IE,type=Graph_Kernel)/tau_e],[-b*alpha_EI*GraphKernel(eigs,t_EI,type=Graph_Kernel)/tau_i, d_i/tau_i + b*alpha_II*GraphKernel(eigs,t_II,type=Graph_Kernel)/tau_i ]])
        A = np.moveaxis(A,-1,0)
        
#        Gmatrix = np.zeros((len(eigs),2,2), dtype=float)
#        detA = np.linalg.det(A)      
#        A_resc=np.copy(A)
#        A_resc[:,0,0]=-A[:,1,1]
#        A_resc[:,1,1]=-A[:,0,0]
#        A_resc_T=np.moveaxis(A_resc,1,2)
#        A-np.trace(A,axis1=1,axis2=2)[:,np.newaxis,np.newaxis]*np.eye(2)
#        trdet = detA*np.trace(A,axis1=1,axis2=2)  
#        for i in range(len(eigs)):
#            Gmatrix[i,:,:] = 0.5*(detA[i]*Dmatrix + np.dot(A_resc[i,:,:], np.dot(Dmatrix, A_resc_T[i,:,:])))/trdet[i]              
        
        i=0
        omega_range=np.linspace(0,max_omega,np.shape(Full_Spectrum)[1])
        for omega in omega_range:
            Full_Spectrum[:,i] = (Dmatrix[0,0]*(A[:,1,1]**2+omega**2) + Dmatrix[1,1]*A[:,0,1]**2)/((A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]-omega**2)**2 + ((A[:,0,0]+A[:,1,1])*omega)**2)
            #Full_Spectrum[:,i] = (Dmatrix[0,0]*(A[:,1,1]**2+omega**2) + Dmatrix[1,1]*A[:,0,1]**2)/(omega**4 + (A[:,0,0]**2+A[:,1,1]**2)*omega**2 + A[:,0,0]**2 * A[:,1,1]**2)
        
            #for k in range(np.shape(A)[0]):
            #    M=np.matrix(np.linalg.inv(1j*np.eye(2)*omega + A[k,:,:]))
            #    Full_Spectrum[k,i]=np.dot(M,np.dot(Dmatrix,M.H))[0,0].real
            
            #print(i)
            i+=1
            
        if Visual==True:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax.set_xlim(-0.1, 20000)
            ax.set_ylim(omega_range[1], max_omega)
            #########use the /2pi rescaling if want temporal frequency
            #ax.set_ylim(omega_range[1]/(2*np.pi), max_omega/(2*np.pi))
            #line2, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,1,1], 'b-')
            #line1, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,0,0], 'r-')
            ax.set_xscale('log')            
            ax.set_yscale('log')
            ax.set_xlabel("Spatial Eigenmode ($k$)")
            ax.set_ylabel("Angular Frequency ($\omega$)")
            ax.set_title("Spatiotemporal Power Spectrum")                 
            pc=ax.pcolormesh(np.arange(1,len(eigs)+1),omega_range,Full_Spectrum.T,norm=pltcolors.LogNorm())
            fig.colorbar(pc)
            #ax.pcolormesh(np.arange(1,len(eigs)+1),omega_range/(2*np.pi),Full_Spectrum.T)
            
        return Full_Spectrum.T
   

    
   


def Functional_Connectivity(eigvecs, PS, one_dim=True, Visual=False):
    U=eigvecs
    covariance = np.dot(U,np.dot(np.diag(PS),U.T))
    FC=np.dot(np.diag(np.power(np.diag(covariance),-0.5)),np.dot(covariance,np.diag(np.power(np.diag(covariance),-0.5))))
    if Visual==True:
        if one_dim==True:            
            fig3 = plt.figure()
            plt.pcolormesh(FC)
    
        else:
            from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
            import plotly.graph_objs as go
            init_notebook_mode()    
        
            trace1 = go.Heatmap(z=FC)           
            data = [trace1]            
            figz = dict(data=data)
            plot(figz, filename='FC.html')
        
    
    return FC
####################################################################################################
####################################################################################################    
####################################################################################################
#Loop for all semi-analytic calculations given parameter set and eigenvalues: HSS, LSA, PSD
####################################################################################################    

def Full_Analysis(Parameters, Laplacian_eigenvalues, Graph_Kernel, True_Spectrum, first_k=2, LSA=True, Visual=False, SaveFiles=False, Filepath=' '):
   
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
    #sigma_noise_e=Parameters[15] 
    #sigma_noise_i=Parameters[16]   #only one sigma noise=scale_param
    
    #D=1.0
    #tau_e=1.0
    #tau_i=1.0
    #sigma_noise_e=1.0
    #sigma_noise_i=1.0
    
    eigs=Laplacian_eigenvalues
    
    last_k=first_k+len(True_Spectrum)
    
    success = False
    #beginning of calculations
    steady_states, success = H_Simple_Steady_State(alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i, P, Q)
    found_suitable = False
    
    if success==True:
        nrSS=len(steady_states[0])
        #distance between eachSS power spectrum and true
        Dist=np.zeros(nrSS)
        scale_params=np.zeros(nrSS)
        #see linear stability analysis method for types
        SStypes=np.zeros(nrSS)
        
        allG = np.empty((nrSS,len(eigs),2,2), dtype=float)
        allJacEigs = np.empty((nrSS, len(eigs), 2), dtype=complex)
        
        for ss in range(len(steady_states[0])):
            
            Ess = steady_states[0,ss]
            Iss = steady_states[1,ss]
            
            
            if LSA==True:
                SStypes[ss], found_suitable, allJacEigs[ss,:,:] = GraphWC_Jacobian_TrDet(eigs, Graph_Kernel, Ess, Iss,                        
                                                 alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                                 sigma_EE, sigma_IE, sigma_EI, sigma_II, D, 
                                                 tau_e, tau_i,True)    
                
              
                    
            
            allG[ss,:,:,:] = Graph_WC_Spatiotemporal_PowerSpectrum(eigs, Graph_Kernel, Ess, Iss, 
                                              alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                              sigma_EE, sigma_IE, sigma_EI, sigma_II, D,                      
                                              tau_e, tau_i, 
                                              sigma_noise_e=1, sigma_noise_i=1,
                                              max_omega=100, delta_omega=0.1,
                                              Spatial_Spectrum_Only=True, Visual=False)
            
            ########*******######
            #important: insert here a metric to quantify distance between true spectrum and calculated
            ######****######
            #Dist[ss] = np.max(np.abs(G[first_k:last_k,0,0]- True_Spectrum))
            scale_params[ss] = np.dot(True_Spectrum,allG[ss,first_k:last_k,0,0])/(np.linalg.norm(allG[ss,first_k:last_k,0,0], ord=2))**2            
            
            Dist[ss] = np.linalg.norm(True_Spectrum - scale_params[ss] * allG[ss,first_k:last_k,0,0], ord=2)
            #Dist[ss] = -stats.ks_2samp( Gmatrix[first_k:last_k,0,0], True_Spectrum )[1]
        
        if np.any(SStypes!=0):    
            
            mask = np.argwhere(SStypes!=0)       
            if ~np.all(np.isnan(Dist[mask])):
                bestSSS = mask[np.nanargmin(Dist[mask])][0]
                minDist=Dist[bestSSS]
                scale_param=scale_params[bestSSS]
                print("Best suitable steady state: %d, with Ess=%.4f Iss=%.4f, Distance: %.4f, Scale: %.4f"%(bestSSS, steady_states[0,bestSSS], steady_states[1,bestSSS], minDist, scale_param))
                bestG=scale_param * allG[bestSSS,:,:,:]
                
                
                    
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
                    line1, = plt.loglog(np.arange(1,len(eigs)+1),0.8*bestG[:,0,0], linewidth=2)#, 'b-')
                    line3, = plt.loglog(np.arange(first_k+1,last_k+1),True_Spectrum, 'b--', linewidth=2)
                 
                  
                    
                
               
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
                        plt.savefig(filepath+"Power Spectrum.png")   
                    
                        
                #if G[3,0,0]-G[-3,0,0]<=1:
                # minDist=10000*np.random.rand()+1000
                
                                   
                return minDist
            else:
                print("Unrealistic spectra")
                return float('Inf')
        else:
            print("No suitable (LSA) steady states found")
            return float('Inf')
    
    else:
        #case where no positive/exact solutions found (can print from SS method)
        return float('Inf')
    
    
#utility functions to read data in python format from Selen's files
def construct_fibers_from_data(filepath_data,
                               filepath_Fibers,
                               savefiles=True,
                               output_filepath_fiber_edges=None,
                               output_filepath_fiber_lengths=None,
                               output_filepath_fiber_dist_starts=None,
                               output_filepath_fiber_dist_ends=None,
                               output_filepath_fiber_ends=None):

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

    mesh_fiber_nodes = np.zeros((len(Fibers),2))
    dist_starts=[]
    dist_ends=[]
    for i in range(fiber_start.shape[0]):
        print(i)
        dist_start=np.Inf
        dist_end=np.Inf
        
        for j in range(AllVet.shape[1]):
            dist_start_new = np.linalg.norm(AllVet[:,j] - fiber_start[i,:], ord=2)
            dist_end_new = np.linalg.norm(AllVet[:,j] - fiber_end[i,:], ord=2)
    
            if dist_start_new<dist_start:
                dist_start=dist_start_new
                mesh_start=j
            if dist_end_new<dist_end:  
                dist_end=dist_end_new
                mesh_end=j
                
            
        mesh_fiber_nodes[i,0]=mesh_start
        mesh_fiber_nodes[i,1]=mesh_end
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
                                       threshold=False,
                                       max_dist=10,
                                       filepath_fiber_dist_starts=None,
                                       filepath_fiber_dist_ends=None,
                                       
                                       visual=True,
                                       plot_subcortex=False,
                                       plot_DTI_edges=False
                                       ):

    with h5py.File(filepath_data, 'r') as file:
        #print(list(file.keys()))

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
    
    print("Constructing mesh adjacency matrix...")
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
    fiber_end=np.load(filepath_fiber_ends[0])
    
    #also, we calculated the distance between the fiber beginning/end and the mesh.
    #this data can be read and used to apply a threshold to the fibers based on the distance from the mesh.
    fiber_dist_starts=np.load(filepath_fiber_dist_starts[0])
    fiber_dist_ends=np.load(filepath_fiber_dist_ends[0])
    
    #if threshold is set to false, simply set the maximum allowed distance to infinity s.t. all fibers are included
    if threshold==False:
        max_dist=np.inf
    
    if add_DTI==True:
        print("Now adding DTI fibers from "+filepath_fiber_edges[0]+"...")                
        #loop over all fibers
        for i in range(DTI_edges.shape[0]):
        
            #threshold loop. trivial if threshold is set to false
            if fiber_dist_starts[i]<=max_dist and fiber_dist_ends[i]<=max_dist:
            
                #in some cases, the fiber's beginning and end happen on the same node. 
                #to avoid auto-edges in the graph, we switch to the second-nearest-neighbor in those cases
                if DTI_edges[i,1]==DTI_edges[i,0]:
                    dist=np.inf
#                for j in Edges_mesh_ends[np.where(Edges_mesh_starts==DTI_edges[i,0])]:
#                    new_dist = np.linalg.norm(AllVet_comp[:,j] - fiber_end[i,:], ord=2)
#                    if new_dist<dist:
#                        dist=new_dist
#                        new_end=j
#                
#                print(new_dist)
#                if mesh_adjacency[DTI_edges[i,0],new_end] != 0:
#                    print("DTI fiber %g =cortical edge?"%i)
#                    
#                mesh_adjacency[DTI_edges[i,0],new_end]=1/fiber_lengths[i]**2    
#                mesh_adjacency[new_end,DTI_edges[i,0]]=1/fiber_lengths[i]**2
                else:  
                    #avoid double counting
                    if mesh_adjacency[DTI_edges[i,0],DTI_edges[i,1]] == 0:
                        mesh_adjacency[DTI_edges[i,0],DTI_edges[i,1]]=40000/fiber_lengths[i]**2    
                        mesh_adjacency[DTI_edges[i,1],DTI_edges[i,0]]=40000/fiber_lengths[i]**2  



    #Repeat for the two fiber paths, obtained from fg and fgCC datasets respectively
    DTI_edges=np.load(filepath_fiber_edges[1])
    fiber_lengths=np.load(filepath_fiber_lengths[1])
    fiber_end=np.load(filepath_fiber_ends[1])
    
    #also, we calculated the distance between the fiber beginning/end and the mesh.
    #this data can be read and used to apply a threshold to the fibers based on the distance from the mesh.
    fiber_dist_starts=np.load(filepath_fiber_dist_starts[1])
    fiber_dist_ends=np.load(filepath_fiber_dist_ends[1])
    
    
    if add_DTI==True:
        print("Now adding DTI fibers from "+filepath_fiber_edges[1]+"...")                
        #loop over all fibers
        for i in range(DTI_edges.shape[0]):
        
            #threshold loop. trivial if threshold is set to false
            if fiber_dist_starts[i]<=max_dist and fiber_dist_ends[i]<=max_dist:
            
                #in some cases, the fiber's beginning and end happen on the same node. 
                #to avoid auto-edges in the graph, we switch to the second-nearest-neighbor in those cases
                if DTI_edges[i,1]==DTI_edges[i,0]:
                    dist=np.inf
#                for j in Edges_mesh_ends[np.where(Edges_mesh_starts==DTI_edges[i,0])]:
#                    new_dist = np.linalg.norm(AllVet_comp[:,j] - fiber_end[i,:], ord=2)
#                    if new_dist<dist:
#                        dist=new_dist
#                        new_end=j
#                
#                print(new_dist)
#                if mesh_adjacency[DTI_edges[i,0],new_end] != 0:
#                    print("DTI fiber %g =cortical edge?"%i)
#                    
#                mesh_adjacency[DTI_edges[i,0],new_end]=1/fiber_lengths[i]**2    
#                mesh_adjacency[new_end,DTI_edges[i,0]]=1/fiber_lengths[i]**2
                else:  
                    #avoid double counting
                    if mesh_adjacency[DTI_edges[i,0],DTI_edges[i,1]] == 0:
                        mesh_adjacency[DTI_edges[i,0],DTI_edges[i,1]]=40000/fiber_lengths[i]**2    
                        mesh_adjacency[DTI_edges[i,1],DTI_edges[i,0]]=40000/fiber_lengths[i]**2

    
    
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