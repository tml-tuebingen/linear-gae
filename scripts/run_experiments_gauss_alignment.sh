#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=g

#resources:

#SBATCH --cpus-per-task=2
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=50G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --time=3-00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=/home/sklepper/logs/g.err
# write the error output to job.*jobID*.err

#SBATCH --output=/home/sklepper/logs/g.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=solveig.klepper@uni-tuebingen.de
# your mail address

####
#c) Execute your code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####


cd ..

while read p; do
	singularity exec --nv /common/singularityImages/TCML-CUDA11_6_TF2_7_1_PT1_10_2.simg python3 /home/sklepper/code/run_batch.py $p
	echo ONE MORE DONE!
done <~/code/params/experiments_gauss_alignment.txt

echo ALL DONE!
