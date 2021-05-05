import numpy as np
import pymaster as nmt

# define inputs
sky_frac = 0.44
a_type = 'MF+Cl'
m_type = 'l'
smoothing = 5
nside = 256
thr_ct = 10

# get Cl info
b = nmt.NmtBin.from_nside_linear(nside,50)
cl_len = b.get_n_bands()

# load all array
a = np.load(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}.npy')

if a_type=='Cl':
    
    # get the first clustering or lensing map
    if m_type=='c' or m_type=='l':
        first = a[:,0:cl_len]

    # get the first clustering map (index = 0) and the first lensing map (index = 5)
    if m_type=='c+l':
        q = cl_len
        first = np.concatenate((a[:,0:q],a[:,5*q:6*q]),axis=1)

# MFs are organised by V0,V1,V2 then map index
if a_type=='MF':

    # get the first clustering map
    if m_type=='c':
        first = np.concatenate((a[:,0:10],a[:,50:60],a[:,100:110]),axis=1)

    # get the first lensing map
    if m_type=='l':
        first = np.concatenate((a[:,0:10],a[:,40:50],a[:,80:90]),axis=1)

    # get the first clustering map and the first lensing map
    if m_type=='c+l':
        first = np.concatenate((a[:,0:10],a[:,50:60],a[:,90:100],a[:,140:150],a[:,180:190],a[:,230:240]),axis=1)

if a_type=='MF+Cl':

    # first clustering map
    if m_type=='c':
        mf_len = 150
        cl = a[:,mf_len:mf_len+cl_len]
        mf = np.concatenate((a[:,0:10],a[:,50:60],a[:,100:110]),axis=1)
        first = np.concatenate((mf,cl),axis=1)

    # first lensing map
    if m_type=='l':
        mf_len = 120
        cl = a[:,mf_len:mf_len+cl_len]
        mf = np.concatenate((a[:,0:10],a[:,40:50],a[:,80:90]),axis=1)
        first = np.concatenate((mf,cl),axis=1)
        
    if m_type=='c+l':
        mf_len = 270
        cl = np.concatenate((a[:,mf_len:mf_len+cl_len],a[:,mf_len+5*cl_len:mf_len+6*cl_len]),axis=1)
        mf = np.concatenate((a[:,0:10],a[:,50:60],a[:,90:100],a[:,140:150],a[:,180:190],a[:,230:240]),axis=1)
        first = np.concatenate((mf,cl),axis=1)

        
np.save(f'all_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_1map.npy',first)
print(first.shape)


