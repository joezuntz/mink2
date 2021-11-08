#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:36:21 2021

@author: ngrewal
"""
import sys
import subprocess
sys.path.append("./simulation_code/")
from simulate_des_maps import *
import os
dirname = os.path.dirname(__file__)

# SCRIPT THAT CREATES A SUITE OF SUBMISSION SCRIPTS

smoothing = sys.argv[1]
nside = sys.argv[2]
thr_ct = sys.argv[3]
sky_frac = sys.argv[4]
a_type = sys.argv[5]
m_type = sys.argv[6]
source = sys.argv[7]
source_file = os.path.abspath(os.path.join(dirname, "simulation_code/new_data", source+".fits"))
print(source_file)
#itr = 4 # number of fiducial iterations (in thousands)
#last_index = int(itr*1000)
#print(last_index)

# make worker nodes a variable bc they change frequently
node_list = 'worker[001-036],worker[038-066],worker[075-076],worker[078-084]'
                                     

# run an mcmc chain

# create a new script
mcmc_script = open(f'/home/ngrewal/mink2/sub/mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.sub',"w+")

mcmc_script.write(
f'''#!/bin/bash                    
#SBATCH --time=100:00:00     
#SBATCH --cpus-per-task=32
#SBATCH --exclude={node_list}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=scratchdisk 
#SBATCH --output=log.mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.txt

export LD_LIBRARY_PATH=/home/ngrewal/.conda/envs/nisha/lib:${{LD_LIBRARY_PATH}}
export OMP_NUM_THREADS=32                                                                                            
export TMPDIR=/scratch/ngrewal
mkdir -p $TMPDIR

SMOOTHING={smoothing} NSIDE={nside} THR_CT={thr_ct} SKY_FRAC={sky_frac} A_TYPE={a_type} M_TYPE={m_type} SOURCE={source} cosmosis mfcl.ini''')
# new cosmosis command

# old command
#time python /home/ngrewal/mink2/mcmc.py {smoothing} {nside} {thr_ct} {sky_frac} {a_type} {m_type} {source} {source_file}

mcmc_script.close()


# run script
s = subprocess.run(args = ['sbatch', f'/home/ngrewal/mink2/sub/mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source}.sub'], capture_output = True)
print(s)


# gets job id from the compute process instance
print(str(s.stdout).split()[-1][0:6])

