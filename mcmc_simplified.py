import numpy as np
import math
import os
#import zeus
#from zeus import ChainManager
import emcee

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

from likelihood_simplified import * 

os.environ["PATH"]='/home/ngrewal/flask/flask/bin:'+os.environ["PATH"]

# inputs (there are default values in the likelihood function)
smoothing, nside, thr_ct, sky_frac,m_type, save_L = 5,256,10,1,'c+l',False

# initialise zeus MCMC
cosmo_params = np.array([0.3,0.8])
nsteps, nwalkers, ndim, nchains = 1000, 4*len(cosmo_params), len(cosmo_params), 1
start = np.random.randn(nwalkers, ndim)*cosmo_params*0.000001 + np.tile(cosmo_params,(nwalkers,1))

# save empty likelihood function
if save_L==True:
    np.save('L',np.zeros(0))

# run sampler without MPI
sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_s, args=[smoothing,nside,thr_ct,sky_frac,m_type,save_L])

sampler.run_mcmc(start, nsteps)

# save chain without using a loop
chain = sampler.get_chain(flat=False) #3D
np.save(f'chain_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_Cl_{m_type}_1map',chain)

'''
#The code below only stops when a time limit is set
while True:
    sampler.run_mcmc(start=sampler.get_last_sample(), nsteps=nsteps, log_prob0=sampler.get_last_log_prob())
    chain = sampler.get_chain(flat=False) #3D
    np.save(f'chain_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_Cl_l_1map',chain)
 



# run sampler and get chain
with ChainManager(nchains) as cm:
    rank = cm.get_rank
    
    sampler = zeus.EnsembleSampler(nwalkers, ndim, likelihood_s, args=[smoothing,nside,thr_ct,sky_frac,save_L],pool = cm.get_pool)
    sampler.run_mcmc(start, nsteps)

    #The code below only stops when a time limit is set
    while True:
        sampler.run_mcmc(start=sampler.get_last_sample(), nsteps=nsteps, log_prob0=sampler.get_last_log_prob())
        print('cycle_sampler')
        chain = sampler.get_chain(flat=False) #3D
        print('cycle_chain')
        np.save(f'chain_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_Cl_l_1map',chain)
        print('cycle_save')
        start = sampler.get_last_sample
        print('cycle_end')
 
'''
