#!/bin/bash

for i in 1 2 4 8 16 32
do
	qsub -l nodes=$i -t $i -N h_matrix_scaleup_$i script_h_matrix_scaleup.pbs
done
