import numpy as np
import math
import os
#import zeus
#from zeus import ChainManager
import emcee
import pickle

import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

from likelihood import * 

os.environ["PATH"]='/home/ngrewal/flask/flask/bin:'+os.environ["PATH"]

# inputs (there are default values in the likelihood function)
smoothing, nside, thr_ct, sky_frac, a_type, m_type = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5],sys.argv[6]
# 5,256,10,1,'Cl','l'

# initialise MCMC
cosmo_params = np.array([0.3,0.8])
nsteps, nwalkers, ndim, nchains = 10, 10*len(cosmo_params), len(cosmo_params), 1
   
# save empty likelihood function
#np.save(f'L_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}.npy',np.zeros(0))

# use observables and likelihood function with only omega_m and sigma_8 and all other parameters fixed
def likelihood_s(cosmo_params,smoothing,nside,thr_ct,sky_frac,a_type,m_type):
    cms = [0.048,cosmo_params[0],0.7,0.96,cosmo_params[1],1.42,1.65,1.6,1.92,2]
    return likelihood(cms,smoothing,nside,thr_ct,sky_frac,a_type,m_type)

# check if sampler exists
if os.path.exists(os.path.join(os.getcwd(),f'mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_2params.p')):
    start = None
    sampler = pickle.load(open(f'mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_2params.p',"rb"))
else:
    # set a random starting position
    start = np.random.randn(nwalkers, ndim)*cosmo_params*0.01 + np.tile(cosmo_params,(nwalkers,1))
    # build sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_s, args=[smoothing,nside,thr_ct,sky_frac,a_type,m_type])


# this will stop when the time runs out
while True:
    # run sampler
    sampler.run_mcmc(start, nsteps)

    # set starting position after first run
    start = None
    
    #force print
    sys.stdout.flush()
    
    # save sampler
    pickle.dump(sampler,open(f'mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_2params.p',"wb"))

    # save chain 
    chain = sampler.get_chain(flat=False) #3D
    np.save(f'chain_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_2params',chain)

    # save likelihood values
    np.save(f'L_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_2params',sampler.get_log_prob())


            
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
