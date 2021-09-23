import os
import numpy as np
import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *
from observables import observables, number_of_observables

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]
os.environ["PATH"]=os.path.abspath("../flask/bin") + ":" +os.environ["PATH"]

# define inputs
smoothing = int(sys.argv[1])
nside = int(sys.argv[2])
thr_ct = int(sys.argv[3])
sky_frac = float(sys.argv[4])
m_type = sys.argv[5]
source = sys.argv[6]
source_file = sys.argv[7]
itr = int(sys.argv[8]) # count of fiducial realisations in thousands

bias = get_fiducial_bias(source_file)

if sky_frac==1:
    sky_frac = int(sky_frac)

is_cori = os.environ.get('NERSC_HOST', '') in ['cori', 'perlmutter']

if is_cori:
    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

# path for given outputs. Different if on NERSC vs cuillin.
if is_cori:
    path_mf = '/global/cscratch1/sd/zuntz/nisha/MFs'
    path_cl = '/global/cscratch1/sd/zuntz/nisha/Cls'
else:
    path_mf = '/disk01/ngrewal/MFs'
    path_cl = '/disk01/ngrewal/Cls'

ncl, nmf = number_of_observables(thr_ct, nside, m_type, source_file)
    
for i in range(itr):
    
    # define index of simulation and output
    if is_cori:
        index = i*size + rank
    else:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])+1000*i

    # fiducial values
    omega_b, omega_m, h, n_s, sigma_8 = 0.048,0.3,0.7,0.96,0.8

    # Calculate observables
    obs = observables(omega_b, omega_m, h, n_s, sigma_8, bias, smoothing, nside, thr_ct, sky_frac, a_type = 'MF+Cl', m_type = m_type, seed = index, source_file = source_file)
    v_all = obs[:nmf]
    c = obs[nmf:]
    assert len(obs) == ncl + nmf, "Joe messed something up in number_of_observables"

    # save MFs and Cls
    np.save(os.path.join(path_mf, f'V_{index}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}_{source}'),v_all)
    np.save(os.path.join(path_cl, f'C_{index}_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}_{source}'),c)



'''
path = '/disk01/ngrewal/Fiducial_Simulations'

# load maps
cmaps = np.load(os.path.join(path, f'cmaps_{index}.npy'))  # clustering maps
lmaps = np.load(os.path.join(path, f'lmaps_{index}.npy'))  # lensing maps
 
## Simplify Maps ##

# reduce pixel size - do second set if using nside=1024 (opt)
cmaps1 = hp.ud_grade(cmaps,nside,power=-2)         
lmaps1 = hp.ud_grade(lmaps,nside)

# smooth maps
smoothing = np.radians(smoothing_arcmin/60)      # convert arcmin to degrees
cmaps2 = np.zeros((len(cmaps1),12*nside**2))   
lmaps2 = np.zeros((len(lmaps1),12*nside**2))

for i in range(len(cmaps1)):
    cmaps2[i] = hp.smoothing(cmaps1[i],fwhm=smoothing)
 
for i in range(len(lmaps1)):
    lmaps2[i] = hp.smoothing(lmaps1[i],fwhm=smoothing) 
'''
