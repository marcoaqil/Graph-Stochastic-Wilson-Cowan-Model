import numpy as np
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt
import h5py
import os

####################################################################################################
####################################################################################################
def GraphKernel(x,t,type='Gaussian'):
    if t<0:
        print("Need positive kernel parameter")
        return
    else:
        if type=='Gaussian':
            return np.exp(-t*x)
        elif type=='Exponential':
            return t/(t+x)
        elif type=='Pyramid':
            return (np.sinc(np.sqrt(x)))**2


####################################################################################################
####################################################################################################
def one_dim_Laplacian_eigenvalues(gridsize, h, syn=0, vecs=False):
     
    diagonals = [np.ones(gridsize-1),np.zeros(gridsize+1),np.ones(gridsize-1)]
    #1D grid-graph
    AdjMatrix = sp.sparse.diags(diagonals,[-1,0,1]).toarray() 
    #periodic boundary
    #AdjMatrix[0,gridsize-1]=1
    #AdjMatrix[gridsize-1,0]=1
    
    #"synapses" (nonlocal connections) 
    for p in range(syn):        
        k1=int(np.floor(gridsize*np.random.rand()))
        k2=int(np.floor(gridsize*np.random.rand()))
        AdjMatrix[k1,k2]=-1
        AdjMatrix[k2,k1]=-1
    
    Deg=np.sum(AdjMatrix, axis=0)
    sqrt_Deg=np.power(Deg,-0.5)
    Degree_Matrix=sp.sparse.diags(Deg)
    sqrt_Degree_Matrix=sp.sparse.diags(sqrt_Deg)
    regLap = Degree_Matrix - sp.sparse.csc_matrix(AdjMatrix)
    Laplacian = (sp.sparse.csc_matrix.dot(sqrt_Degree_Matrix,sp.sparse.csc_matrix.dot(regLap,sqrt_Degree_Matrix))).toarray()
    #Laplacian[Laplacian>1]=1
    
    #unnormalized laplacian
    Laplacian=regLap.toarray()
    
    Laplacian/=(h**2)
    
    if vecs==False:    
        return np.linalg.eigvalsh(Laplacian)
    else:
        return np.linalg.eigh(Laplacian)
        
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
        #ax = fig.add_subplot(111)
        #ax.set_xlim(-0.1, 20000)
        #ax.set_ylim(0, 20)
        plt.scatter(np.ravel(Jacobian_eigenvalues).real,np.ravel(Jacobian_eigenvalues).imag, s=2, c='black')
    
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
def Graph_WC_SpatialPowerSpectrum(Laplacian_eigenvalues, Graph_Kernel='Gaussian', Ess=None, Iss=None,  
                       alpha_EE=1, alpha_IE=1, alpha_EI=1, alpha_II=1, d_e=1, d_i=1,
                       sigma_EE=10, sigma_IE=10, sigma_EI=10, sigma_II=10, D=1,
                       tau_e=1, tau_i=1,
                       sigma_noise_e=1, sigma_noise_i=1,
                       Visual=False):
    
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
    
    Exc_Spectrum_Only=True
    
    if Exc_Spectrum_Only==True:
        Gmatrix2 = np.zeros((len(eigs),2,2), dtype=float)
  
        G_EE = alpha_EE*GraphKernel(eigs,t_EE,type=Graph_Kernel)
        G_IE = alpha_IE*GraphKernel(eigs,t_IE,type=Graph_Kernel)
        G_EI = alpha_EI*GraphKernel(eigs,t_EI,type=Graph_Kernel)
        G_II = alpha_II*GraphKernel(eigs,t_II,type=Graph_Kernel)
        
        Gmatrix2[:,0,0] = 0.5*((sigma_noise_e**2)/(tau_i*d_e-tau_i*a*G_EE+d_i*tau_e+b*tau_e*G_II))*((tau_i/tau_e) + ((a**2)*(G_IE**2)+ (d_i + b*G_II)**2)/(d_e*d_i+ (d_e*b*G_II) - (a*d_i*G_EE) - a*b*(G_EE*G_II-G_EI*G_IE)))
                            
    #else:    
    
    
        Dmatrix=np.array([[sigma_noise_e/tau_e,0],[0,sigma_noise_i/tau_i]])**2
       
        A = np.stack([[d_e/tau_e - a*alpha_EE*GraphKernel(eigs,t_EE,type=Graph_Kernel)/tau_e, a*alpha_IE*GraphKernel(eigs,t_IE,type=Graph_Kernel)/tau_e],[-b*alpha_EI*GraphKernel(eigs,t_EI,type=Graph_Kernel)/tau_i, d_i/tau_i + b*alpha_II*GraphKernel(eigs,t_II,type=Graph_Kernel)/tau_i ]])
        A = np.moveaxis(A,-1,0)
        detA = np.linalg.det(A)
       
        A_resc=np.copy(A)
        A_resc[:,0,0]=-A[:,1,1]
        A_resc[:,1,1]=-A[:,0,0]
        A_resc_T=np.moveaxis(A_resc,1,2)
        #A-np.trace(A,axis1=1,axis2=2)[:,np.newaxis,np.newaxis]*np.eye(2)
        trdet = detA*np.trace(A,axis1=1,axis2=2)  
        
        Gmatrix = 0.5*(detA[:,np.newaxis,np.newaxis]*Dmatrix + A_resc*Dmatrix*A_resc_T)/trdet[:,np.newaxis,np.newaxis]
        
    if Visual==True:
        plt.ion()
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_xlim(-0.1, 20000)
        #ax.set_ylim(0, 20)
        line2, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,1,1], 'b-')
        line1, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix[:,0,0], 'r-')
        line3, = plt.loglog(np.arange(1,len(eigs)+1),Gmatrix2[:,0,0], 'k--')
    
    return Gmatrix



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
    D=Parameters[12]
    tau_e=Parameters[13] 
    tau_i=Parameters[14] 
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
                                                 tau_e, tau_i)    
                
              
                    
    
            allG[ss,:,:,:] = Graph_WC_SpatialPowerSpectrum(eigs, Graph_Kernel, Ess, Iss, 
                                              alpha_EE, alpha_IE, alpha_EI, alpha_II, d_e, d_i,
                                              sigma_EE, sigma_IE, sigma_EI, sigma_II, D,                      
                                              tau_e, tau_i, 
                                              sigma_noise_e=1, sigma_noise_i=1)
            
            ########*******######
            #important: insert here a metric to quantify distance between true spectrum and calculated
            ######****######
            #Dist[ss] = np.max(np.abs(G[first_k:last_k,0,0]- True_Spectrum))
            scale_params[ss] = np.dot(True_Spectrum,allG[ss,first_k:last_k,0,0])/(np.linalg.norm(allG[ss,first_k:last_k,0,0], ord=2))**2            
            
            Dist[ss] = np.linalg.norm(True_Spectrum - scale_params[ss] * allG[ss,first_k:last_k,0,0], ord=2)
            #Dist[ss] = -stats.ks_2samp( Gmatrix[first_k:last_k,0,0], True_Spectrum )[1]
        
        if found_suitable==True:    
            
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
                    #ax = fig.add_subplot(111)
                    #ax.set_xlim(-0.1, 20000)
                    #ax.set_ylim(0, 20)
                    line2, = plt.loglog(np.arange(1,len(eigs)+1),bestG[:,1,1], 'b-')
                    line1, = plt.loglog(np.arange(1,len(eigs)+1),bestG[:,0,0], 'r-')
                    line3, = plt.loglog(np.arange(first_k+1,last_k+1),True_Spectrum, 'k--')
                 
                  
                    
                
               
                if SaveFiles==True:
                    
                    if Filepath==' ':
                        filepath = 'G:/Macbook Stuff/Analysis Results/'+Graph_Kernel+' Kernel/aEE=%.3g aIE=%.3g aEI=%.3g aII=%.3g dE=%.3g dI=%.3g ' %(alpha_EE,alpha_IE,alpha_EI,alpha_II,d_e,d_i)
                        filepath += 'P=%.3g Q=%.3g sEE=%.3g sIE=%.3g sEI=%.3g sII=%.3g D=%.3g tE=%.3g tI=%.3g/'%(P,Q,sigma_EE,sigma_IE,sigma_EI,sigma_II,D,tau_e,tau_i) 
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
                        hf.create_dataset("Steady States",  data=steady_states)
                        hf.create_dataset("Distance",  data=Dist)
                        hf.create_dataset("Scale", data=scale_params)
                        hf.create_dataset("Type",  data=SStypes)
                        hf.create_dataset("AllG", data=allG)
                    
                    if Visual==True:
                        plt.savefig(fig, filepath+"Power Spectrum.png")   
                    
                        
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