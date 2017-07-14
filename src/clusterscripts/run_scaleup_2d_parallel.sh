#!/bin/bash

for i in 64 128 256 1024
do
	qsub -l nodes=$i -t $i -N h_matrix_scaleup_2d_parallel_$i script_h_matrix_scaleup_2d_parallel.pbs
done
