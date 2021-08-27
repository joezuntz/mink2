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
source_file = os.path.abspath(os.path.join(dirname,"new_data",source+".fits"))
bias = get_fiducial_bias(source_file)
itr = 1 # number of fiducial iterations (in thousands)

# make worker nodes a variable bc they change frequently
node_list = 'worker[001-036],worker[038-066],worker[075-076],worker[078-084]'

# RUN 3 ANALYSIS STEPS

# STEP 1: generate maps and measure observables
# STEP 2: combine observable iterations into a single array

# create a new script
calc_script = open(f'/home/ngrewal/mink2/sub/calc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source_file}.sub',"w+")

calc_script.write(          
f'''#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --array=1-1000
#SBATCH --cpus-per-task=8
#SBATCH --exclude={node_list}                                                                 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --output='/disk01/ngrewal/logs/log.mf_cl_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{m_type}_{source_file}.%a.txt'

export LD_LIBRARY_PATH=${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH}}
export OMP_NUM_THREADS=8
time srun -u -n1 python /home/ngrewal/mink2/calc_mf_cl.py {smoothing} {nside} {thr_ct} {sky_frac} {m_type} {bias} {source_file} {itr}

# concatenate fiducial observables
python /home/ngrewal/mink2/fid_concat.py {smoothing} {nside} {thr_ct} {sky_frac} {a_type} {m_type} {source_file} {itr}''')

calc_script.close()

# run script
s = subprocess.run(args = ['sbatch',f'/home/ngrewal/mink2/sub/calc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source_file}.sub'], capture_output = True)
print(s)

# gets job id from the compute process instance
#calc_job_id = (str(s.stdout)[-9:-3])
calc_job_id = str(s.stdout).split()[-1]
print(calc_job_id)


# STEP 3: run an mcmc chain

# create a new script
mcmc_script = open(f'/home/ngrewal/mink2/sub/mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{source_file}.sub',"w+")

mcmc_script.write(
f'''#!/bin/bash                    
#SBATCH --time=00:20:00     
#SBATCH --cpus-per-task=32
#SBATCH --exclude={node_list}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --output=log.mcmc_{a_type}_{m_type}.txt
#SBATCH --dependency=afterok:{calc_job_id}

export LD_LIBRARY_PATH=${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH}}
export OMP_NUM_THREADS=32                                                                                            
time python /home/ngrewal/mink2/mcmc.py {smoothing} {nside} {thr_ct} {sky_frac} {a_type} {m_type} {bias} {source_file}''')

mcmc_script.close()


# run script
s = subprocess.run(args = ['sbatch',f'/home/ngrewal/mink2/sub/mcmc_s{smoothing}_n{nside}_t{thr_ct}_f{sky_frac}_{a_type}_{m_type}_{sourcefile}.sub'], capture_output = True)
print(s)


# gets job id from the compute process instance
#calc_job_id = (str(s.stdout)[-9:-3])
calc_job_id = str(s.stdout).split()[-1]
print(calc_job_id)

                  
