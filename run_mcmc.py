import numpy as np
import math
import os
import zeus
from zeus import ChainManager

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

from likelihood import * 

os.environ["PATH"]='/home/ngrewal/flask/flask/bin:'+os.environ["PATH"]

# inputs (there are default values in the likelihood function)
smoothing, nside, thr_ct, sky_frac, a_type = 5,256,10,1,'MF+Cl'

# initialise zeus MCMC
cosmo_params = np.array([0.048,0.3,0.7,0.96,0.8,1.42,1.65,1.6,1.92,2])
nsteps, nwalkers, ndim, nchains = 50, 4*len(cosmo_params), len(cosmo_params), 1
start = np.random.randn(nwalkers, ndim)*0.001 + np.tile(cosmo_params,(nwalkers,1))

# run sampler and get chain
with ChainManager(nchains) as cm:
    rank = cm.get_rank
    
    sampler = zeus.EnsembleSampler(nwalkers, ndim, likelihood, args=[smoothing,nside,thr_ct,sky_frac,a_type],pool = cm.get_pool)

    while True:
        sampler.run_mcmc(start, nsteps)
    
        chain = sampler.get_chain(flat=False)
        np.save('chain.npy',chain)


 


''' Without MPI
# run zeus MCMC
sampler = zeus.EnsembleSampler(nwalkers, ndim, likelihood, args=[smoothing, nside, thr_ct])
sampler.run_mcmc(start, nsteps)

# get chain from sampler
# cut burn-in phase off (discard first half of the chain)
chain = sampler.get_chain(flat=True, discard=np.int(nsteps/2))

np.save('mcmc_chain',chain)
'''
