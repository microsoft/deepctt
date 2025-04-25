#!/bin/bash
#SBATCH -J tst                         # Job name
#SBATCH -o logs/tst_%j.out                  # output file (%j expands to jobID)
#SBATCH -e logs/tst_%j.err                  # error log file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=100GB                           # server memory (MBs) requested (per node)
#SBATCH --gres=gpu:a6000:1

#
# Script for running the Higgs two-sample experiment with n=4096 without poisoning.
# n: the number of observations from each of the two distributions.
# NOTE: that the total number of observations is 2n.
# P_POISONING: proportion of the second sample that is poisoned (i.e., drawn from the first distribution).
# DIMENSION: number of original features that we consider.
#
# Example usage:
# ./examples/higgs/slurm/power_n4096.sh OUTPUT_PATH
#

n=4096
P_POISONING=0
OUTPUT_PATH=$1

# Run deep kernel CTT with varying values of g (the oversampling parameter).
# The number of kernel evaluations is 4**g * n * num_bins with num_bins=16.
for g in {0..4}
do
    for kk in {0..9}
    do
        python run_power.py -m ctt -g $g -kk $kk -n $n -op $OUTPUT_PATH --p_poisoning $P_POISONING
    done
done

# Run wild boostrap block test with varying values of g.
# Here g is the thinning factor to determine the block size, not the oversampling parameter.
# The number of kernel evaluations is n**2 / 2**g.
# NOTE: g=0 corresponds to the exact (quadratic time) MMD test.
for g in 0 1 2 3 4 6 8
do
    for kk in {0..9}
    do
        python run_power.py -m wb_block -g $g -kk $kk -n $n -op $OUTPUT_PATH --p_poisoning $P_POISONING
    done
done

# Run wild boostrap subsampling test with varying values of g.
# Here g is the thinning factor to determine the thinned sample size, not the oversampling parameter.
# The number of kernel evaluations is (n/2**g)**2.
for g in 0 0.5 1 1.5 2 3 4
do
    for kk in {0..9}
    do
        python run_power.py -m subsampling -g $g -kk $kk -n $n -op $OUTPUT_PATH --p_poisoning $P_POISONING
    done
done
