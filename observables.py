#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:47:48 2021

@author: ngrewal
"""
import numpy as np
import math
import os
from mf import calc_mf_2maps
from cl import Cl_2maps
import pymaster as nmt
import time
import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"]

def number_of_observables(thr_ct, nside, a_type, m_type, source_file):
    # Get the number of maps in the given source file
    f = fits.open(source_file)
    nsource = f["SOURCE"].header['NBIN']
    nlens = f["LENS"].header['NBIN']
    f.close()

    # Check if we are using source, lens, or both bins
    if m_type == 'c+l':
        nmap = nsource + nlens
    elif m_type == 'l':
        nmap = nsource
    elif m_type == 'c':
        nmap = nlens
    else:
        raise ValueError(f"Unknown m_type {m_type}")

    nmf = nmap * thr_ct * 3

    # Do the same bin generation as is done in cl.py
    b = nmt.NmtBin.from_lmax_linear(lmax=int(1.5*nside),nlb=50)
    cl_len = b.get_n_bands()
    ncl = nmap * cl_len
    return ncl, nmf



def observables(omega_b, omega_m, h, n_s, sigma_8, bias, smoothing, nside, thr_ct, sky_frac, a_type, m_type, seed, source_file):

    # start time
    t0 = time.perf_counter()
    
    # build new clustering and lensing maps
    cmaps,lmaps = simulate_general_maps(omega_b,omega_m,h,n_s,sigma_8,bias,smoothing,nside,seed,source_file)

    # map generation time
    t1 = time.perf_counter()
    print('Map generation time: ',t1-t0,'sec')
    
    # calculate sky fraction
    frac = int(math.floor(sky_frac*12*nside**2))
    
    # power spectrum (Cl) analysis only
    if a_type=='Cl':

        # clustering and lensing maps  
        if m_type=='c+l':                                                                                                                                                                                  
            output = Cl_2maps(cmaps,lmaps,nside,frac).flatten()
            
        # clustering only
        elif m_type=='c':
            output = Cl_2maps(cmaps,[],nside,frac).flatten()
            
        # lensing only
        elif m_type=='l':
            output = Cl_2maps([],lmaps,nside,frac).flatten()
    
        
    # Minkowski functional analysis only
    if a_type=='MF':
            
        # clustering and lensing maps
        if m_type=='c+l':                                                                                                                                                                                   
            v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)
            
        # clustering only
        elif m_type=='c':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)
                
        # lensing only
        elif m_type=='l':
            v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac) 
                 
        output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
    
    
    # Minkowski functional and power spectrum analysis
    if a_type=='MF+Cl':
                     
        # clustering and lensing maps  
        if m_type=='c+l':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)     # calculate MFs
            c = Cl_2maps(cmaps,lmaps,nside,frac)                    # calculate Cls
           
        # clustering only
        elif m_type=='c':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)        # calculate MFs
            c = Cl_2maps(cmaps,[],nside,frac)                           # calculate Cls
                
        # lensing only
        elif m_type=='l':
            v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac)        # calculate MFs
            c = Cl_2maps([],lmaps,nside,frac)                           # calculate Cls
            
        # concatenate MFs and Cls
        output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten(),c.flatten()))


    # observable calculation time
    t2 = time.perf_counter()
    print('Observable calculation time: ',t2-t1,'sec')

                
    return output
        
        
        
'''another try - less code, but takes more time
        
        # TYPES OF MAPS 
        # clustering and lensing maps  
        if m_type=='c+l':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,lmaps,thr_ct,frac)     # calculate MFs
            c = Cl_2maps(cmaps,lmaps,nside,frac)                    # calculate Cls
        
        # clustering only
        elif m_type=='c':
            v,v0,v1,v2 = calc_mf_2maps(cmaps,[],thr_ct,frac)        # calculate MFs
            c = Cl_2maps(cmaps,[],nside,frac)                       # calculate Cls
            
        # lensing only
        elif m_type=='l':
            v,v0,v1,v2 = calc_mf_2maps([],lmaps,thr_ct,frac)        # calculate MFs
            c = Cl_2maps([],lmaps,nside,frac)                       # calculate Cls
    
        # TYPES OF ANALYSIS 
        # Minkowski functional and power spectrum analysis
        if a_type=='MF+Cl':
            output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten(),c.flatten()))
        
        # Minkowski functional analysis only
        if a_type=='MF':
            output = np.concatenate((v0.flatten(),v1.flatten(),v2.flatten()))
        
        # power spectrum (Cl) analysis only
        if a_type=='Cl':
            output = c.flatten()
        '''
