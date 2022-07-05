#!/bin/bash
#SBATCH -o moh-xF-1.8.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=20    # number of threads
#SBATCH --mem-per-cpu=2200      # times ntasks-per-node gives total mem
#SBATCH -p cascade.p            # determines partition
#SBATCH --job-name='moh-xF-1.8'       #job name
#SBATCH --threads-per-core=1
#SBATCH -t 24:00:00

module purge

source /etc/profile.d/modules.sh
source /home/farrugma/.bash_profile

module load slurm/18-08-4-1-hits        #check if there is a newer version, this works but throws an error

g09 moh-xF-1.8.com
wait
