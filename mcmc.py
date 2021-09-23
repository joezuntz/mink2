import numpy as np
import math
import os
#import zeus
#from zeus import ChainManager
import emcee
import time
import pickle

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

from likelihood import * 

os.environ["PATH"]='/home/ngrewal/flask/flask/bin:'+os.environ["PATH"]

# inputs (there are default values in the likelihood function)
smoothing, nside, thr_ct, sky_frac, a_type, m_type, source, source_file = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]
print(sys.argv)

bias = get_fiducial_bias(source_file)
print("Bias is:",bias)

if sky_frac==1:
    sky_frac = int(sky_frac)

# initialise zeus MCMC
nsteps, nwalkers, nchains = 10, 40, 1

# check if sampler exists
if os.path.exists(os.path.join(os.getcwd(),f'mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.p')):
    start = None
    sampler = pickle.load(open(f'mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.p',"rb"))
else:
    if m_type=='l':
        cosmo_params = np.array([0.048,0.3,0.7,0.96,0.8])
        ndim = len(cosmo_params)
        start = np.random.randn(nwalkers, ndim)*cosmo_params*0.01 + np.tile(cosmo_params,(nwalkers,1))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_lens, args=[bias,smoothing,nside,thr_ct,sky_frac,a_type,m_type,source,source_file])
    else:
        cosmo_params = np.concatenate((np.array([0.048,0.3,0.7,0.96,0.8]),bias))
        ndim = len(cosmo_params)
        start = np.random.randn(nwalkers, ndim)*cosmo_params*0.01 + np.tile(cosmo_params,(nwalkers,1))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=[smoothing,nside,thr_ct,sky_frac,a_type,m_type,source,source_file])


while True:
    # run sampler
    sampler.run_mcmc(start, nsteps)

    # set starting position after first run
    start = None
    
    # save sampler
    pickle.dump(sampler,open(f'mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.p',"wb"))

    # save chain
    chain = sampler.get_chain(flat=False) #3D
    np.save(f'chain_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}',chain)

    # save likelihood values
    np.save(f'L_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}',sampler.get_log_prob())
    
'''
# using MPI: 

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
