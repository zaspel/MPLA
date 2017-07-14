# Copyright (C) 2016 Peter Zaspel
#
# This file is part of MPLA.
#
# MPLA is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# MPLA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MPLA.  If not, see <http://www.gnu.org/licenses/>.

# default: test
#
#benchmark: shared
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o benchmark_dgemv.o benchmark_dgemv.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o benchmark_dgemv benchmark_dgemv.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o benchmark_cg.o benchmark_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o benchmark_cg benchmark_cg.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	nvcc -O3 -c -arch=sm_30 -I/home/zaspel/libs/openmpi/include kernel.cu -o kernel.o
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o benchmark_kernel_cg.o benchmark_kernel_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o benchmark_kernel_cg benchmark_kernel_cg.o kernel.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#
#
# shared:
#	g++ -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -pthread  -L/usr/local/cuda/lib64 -lcudart -fPIC -o libmpla.so --shared mpla.cpp
#
#titan:
# 	CC -g -lcudart -fPIC -o libmpla.so --shared mpla.cpp
##	CC -lcurand -L. -lmpla -lcublas -o benchmark_dgemv benchmark_dgemv.cpp
##	CC -lcurand -L. -lmpla -lcublas -o benchmark_cg benchmark_cg.cpp
##	nvcc -g -O3 -c -arch=sm_35 -I$(MPICH_DIR)/include kernel.cu -o kernel.o
##	CC -lcurand -L. -lmpla -lcublas kernel.o -o benchmark_kernel_cg benchmark_kernel_cg.cpp
##	CC -g -lcurand -L. -lmpla -lcublas ${GSL_LIB} kernel.o -o generate_quadrature_weights generate_quadrature_weights.cpp
#	nvcc -O3 -c -arch=sm_35 -I$(MPICH_DIR)/include hmglib_adapter.cu -I/lustre/atlas/proj-shared/csc238/hmglib/src -o hmglib_adapter.o
#	nvcc -O3 -arch=sm_35 -lcurand -L. -L/lustre/atlas/proj-shared/csc238/hmglib/src -lmpla -lhmglib -lcublas ${GSL_LIB} -I$(MPICH_DIR)/include -I/lustre/atlas/proj-shared/csc238/hmglib/src hmglib_adapter.o -o h_matrix_test h_matrix_test.cu
#
#clean:
#	rm *.o *.so test_library benchmark_dgemv
#
#test: shared
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_library.o test_library.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_library test_library.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_cg.o test_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_cg test_cg.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_generic_cg.o test_generic_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_generic_cg test_generic_cg.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	nvcc -c -arch=sm_30 -I/home/zaspel/libs/openmpi/include kernel.cu -o kernel.o
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_kernel_cg.o test_kernel_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_kernel_cg test_kernel_cg.o kernel.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas

all: h_matrix_test h_matrix_test_mrhs h_matrix_test_2d_parallel generate_quadrature_weights

libmpla.so: mpla.cpp mpla.h
	CC -g -lcudart -fPIC -o libmpla.so --shared mpla.cpp

hmglib_adapter.o: hmglib_adapter.h hmglib_adapter.cu
	nvcc -g -O3 -c -arch=sm_35 -I$(MPICH_DIR)/include hmglib_adapter.cu -I/lustre/atlas/proj-shared/csc238/hmglib/src -o hmglib_adapter.o

kernel.o: kernel.h kernel.cu
	nvcc -g -O3 -c -arch=sm_35 -I$(MPICH_DIR)/include kernel.cu -I/lustre/atlas/proj-shared/csc238/hmglib/src -o kernel.o

generate_quadrature_weights: generate_quadrature_weights.cu kernel.o libmpla.so
	nvcc -g -O3 -arch=sm_35 -lcurand -L. -L/lustre/atlas/proj-shared/csc238/hmglib/src -lmpla -lhmglib -lcublas ${GSL_LIB} -I$(MPICH_DIR)/include -I/lustre/atlas/proj-shared/csc238/hmglib/src kernel.o -o generate_quadrature_weights generate_quadrature_weights.cu

h_matrix_test: h_matrix_test.cu hmglib_adapter.o libmpla.so
	nvcc -g -O3 -arch=sm_35 -lcurand -L. -L/lustre/atlas/proj-shared/csc238/hmglib/src -lmpla -lhmglib -lcublas ${GSL_LIB} -I$(MPICH_DIR)/include -I/lustre/atlas/proj-shared/csc238/hmglib/src hmglib_adapter.o -o h_matrix_test h_matrix_test.cu

h_matrix_test_mrhs: h_matrix_test_mrhs.cu hmglib_adapter.o libmpla.so
	nvcc -g -O3 -arch=sm_35 -lcurand -L. -L/lustre/atlas/proj-shared/csc238/hmglib/src -lmpla -lhmglib -lcublas ${GSL_LIB} -I$(MPICH_DIR)/include -I/lustre/atlas/proj-shared/csc238/hmglib/src hmglib_adapter.o -o h_matrix_test_mrhs h_matrix_test_mrhs.cu




h_matrix_test_2d_parallel: h_matrix_test_2d_parallel.cu hmglib_adapter.o libmpla.so
	nvcc -g -O3 -arch=sm_35 -lcurand -L. -L/lustre/atlas/proj-shared/csc238/hmglib/src -lmpla -lhmglib -lcublas ${GSL_LIB} -I$(MPICH_DIR)/include -I/lustre/atlas/proj-shared/csc238/hmglib/src hmglib_adapter.o -o h_matrix_test_2d_parallel h_matrix_test_2d_parallel.cu



