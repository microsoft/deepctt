#!/bin/bash

#
# Script for reproducing the Higgs experiments of Low-Rank Thinning (Section 6.2). 
# n: the number of observations from each of the two distributions.
# NOTE: that the total number of observations is 2n.
# P_POISONING: proportion of the second sample that is poisoned (i.e., drawn from the first distribution).
# DIMENSION: number of original features that we consider.
#
# Example usage:
# ./examples/higgs/slurm/power_n16384_mixture.sh
#

n=16384
P_POISONING=0.5
DIMENSION=2
OUTPUT_PATH="out-${n}-d${DIMENSION}-${P_POISONING}"

# Run deep kernel CTT with varying values of g (the oversampling parameter).
# The number of kernel evaluations is 4**g * n * num_bins with num_bins=16.
for g in {0..4}
do
    for kk in {0..9}
    do
        python run_power.py -m ctt -g $g -kk $kk -n $n -op $OUTPUT_PATH --p_poisoning $P_POISONING -d $DIMENSION
    done
done

# Run wild boostrap block test with varying values of g.
# Here g is the thinning factor to determine the block size, not the oversampling parameter.
# The number of kernel evaluations is n**2 / 2**g.
# NOTE: g=0 corresponds to the exact (quadratic time) MMD test.
for g in 10 8 6 4 2 0
do
    for kk in {0..9}
    do
        python run_power.py -m wb_block -g $g -kk $kk -n $n -op $OUTPUT_PATH --p_poisoning $P_POISONING -d $DIMENSION
    done
done

# Run wild boostrap subsampling test with varying values of g.
# Here g is the thinning factor to determine the thinned sample size, not the oversampling parameter.
# The number of kernel evaluations is (n/2**g)**2.
for g in 5 4 3 2 1
do
    for kk in {0..9}
    do
        python run_power.py -m subsampling -g $g -kk $kk -n $n -op $OUTPUT_PATH --p_poisoning $P_POISONING -d $DIMENSION
    done
done
