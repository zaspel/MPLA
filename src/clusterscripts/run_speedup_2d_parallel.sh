#!/bin/bash

for i in 1 2 4 8 16 32 64 128 256
do
	qsub -l nodes=$i -t $i -N h_matrix_speedup_2d_parallel_$i script_h_matrix_speedup_2d_parallel.pbs
done
