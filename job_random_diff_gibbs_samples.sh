#!/bin/bash
#SBATCH --job-name=array-job     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=6:10:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-9              # job array with index values 0, 1, 2, 3, 4
#SBATCH --mail-type=end          # send email on job start, end and fault
#SBATCH --mail-user=aditijha@princeton.edu

module load anaconda3/2020.11
conda activate al
python run_iohmm.py --input_selection random