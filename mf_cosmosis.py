from cosmosis.datablock import option_section
import numpy as np
from observables import *
import os

os.environ["PATH"]='/home/ngrewal/flask/flask/bin:'+os.environ["PATH"]



# likelihood function gets split up into two parts: one for loading and one for calculating likelihood

def setup(options):
    
    smoothing = options[option_section,"smoothing"]
    nside = options[option_section,"nside"]
    thr_ct = options[option_section,"thr_ct"]
    sky_frac = options[option_section,"sky_frac"]
    a_type = options[option_section,"a_type"]
    m_type = options[option_section,"m_type"]
    source = options[option_section,"source"]
    source_file = options[option_section,"source_file"]
    
    # load fiducial observables
    V = np.load(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.npy')
    print(V.shape)
    
    # find the covariance of the fiducial observables
    cov = np.cov(V.transpose()) 

    # find the mean of the fiducial observables
    fiducial_mean = np.mean(V,axis=0)        

    # testing covariance correction
    #itr = len(V)                                          # find number of iterations
    #N_ = itr-1                                            # number of iterations - 1
    #p = len(V[0])                                         # number of observable data points 
    #i_cov = ((N_)/(N_ - p - 1)) * np.linalg.inv(cov)      # find the inverse covariance with the Anderson-Hartlap correction
    i_cov = np.linalg.inv(cov) # regular covariance
    
    
    return {
        "smoothing": smoothing,
        "nside": nside,
        "thr_ct": thr_ct,
        "sky_frac": sky_frac,
        "a_type": a_type,
        "m_type": m_type,
        "source": source,
        "source_file": source_file,
        "fiducial_mean": fiducial_mean,
        "i_cov": i_cov,
        "cov": cov
    }


def execute(block, config):

    # config is whatever came from the setup function

    omega_b = block["cosmo_params","omega_b"]
    omega_m = block["cosmo_params","omega_m"]
    h = block["cosmo_params","h"]
    n_s = block["cosmo_params","n_s"]
    sigma_8 = block["cosmo_params","sigma_8"]
    bias = [block["cosmo_params", f"b{i}"] for i in range(1,6)]
    
    # fixed parameters
    smoothing = config["smoothing"]
    nside = config["nside"]
    thr_ct = config["thr_ct"]
    sky_frac = config["sky_frac"]
    a_type = config["a_type"]
    m_type = config["m_type"]
    source_file = config["source_file"]
    fiducial_mean = config["fiducial_mean"]
    i_cov = config["i_cov"]
    cov = config["cov"]

    # calculate observables given input parameters
    output = observables(omega_b, omega_m, h, n_s, sigma_8, bias, smoothing, nside, thr_ct, sky_frac, a_type, m_type, seed=29101995, source_file=source_file)
    print("observables: ",output.shape)
    
    print("fid mean: ",fiducial_mean.shape)

    # calculate likelihood      
    diff = output - fiducial_mean
    L = -0.5 * diff @ i_cov @ diff                 

    block["likelihoods", "mfcl_like"] = L
    block['data_vector','mfcl_theory'] = output
    block['data_vector','mfcl_data'] = fiducial_mean  
    block['data_vector','mfcl_icov'] = i_cov  
    block['data_vector','mfcl_cov'] = cov  


    return 0
